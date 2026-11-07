import os
import sys
import re
import time
import shutil
import warnings
warnings.filterwarnings("ignore")

_T_E2E_START = time.time()

from glob import glob
from typing import Optional, Dict
from omegaconf import OmegaConf
import click
import json
import subprocess
if ('CUDA_VISIBLE_DEVICES' not in os.environ
        and not {'RANK', 'LOCAL_RANK', 'WORLD_SIZE'} & set(os.environ)):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.distributed as dist
import torch
import torch.utils.data
import PIL
import grid
import time
from PIL import Image
from accelerate import Accelerator
import warnings
import deepseek
from contextlib import contextmanager
warnings.filterwarnings("ignore", message="Failed to load image Python extension")
from transformers import AutoTokenizer, CLIPTextModel
from einops import rearrange
import torchvision.transforms as T
# torchvision>=0.16 removed 'functional_tensor'; basicsr (used by controlnet annotators) still
# imports it. Alias it to T.functional (rgb_to_grayscale lives there) so basicsr loads cleanly.
import sys as _sys
try:
    _sys.modules.setdefault('torchvision.transforms.functional_tensor', T.functional)
except Exception:
    pass
import transformers
import utils.constants as const
from video_diffusion.data.dataset import ImageSequenceDataset
from video_diffusion.common.util import get_time_string, get_function_args
from video_diffusion.common.instantiate_from_config import instantiate_from_config
from video_diffusion.pipelines.p2p_validation_loop import P2pSampleLogger
from video_diffusion import elve_timing
import numpy as np
import utils.preprocesser_utils as pu
import utils.image_process_utils as ipu
import utils.feature_utils as fu


def setup_diffusers_path(method):
    if method == "ControlNet":
        local_diffusers_path = '/mnt/fspace/hproject/ELVE/diffusersC/'
    elif method in ("FateZero", "MotionEditor"):
        return None
    else:
        raise ValueError(f"Unknown method: {method}")
    if local_diffusers_path in sys.path:
        sys.path.remove(local_diffusers_path)
    sys.path.insert(0, local_diffusers_path)
    return local_diffusers_path

def clear_diffusers_cache():
    modules_to_remove = []
    for module_name in list(sys.modules.keys()):
        if (module_name.startswith('diffusers') or 
            module_name.startswith('transformers')):
            if module_name != 'transformers' and not module_name.startswith('transformers.'):
                modules_to_remove.append(module_name)
    for module_name in modules_to_remove:
        del sys.modules[module_name]

    import logging
    logging.getLogger('diffusers').handlers.clear()

@contextmanager
def diffusers_context(method):
    local_diffusers_path = setup_diffusers_path(method)
    clear_diffusers_cache()

    try:
        yield local_diffusers_path
    finally:
        pass


# ============================================================
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PRETRAINED = "/mnt/fspace/hproject/model/stable-diffusion-v1-5"
MOTIONEDITOR_PYTHON = "/home/user4/miniconda3/envs/motioneditor/bin/python3.10"
TUNE_DONE_MARKER = "tune_done.marker"

_DIST_ENV_KEYS = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                  "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID",
                  "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_ERROR_FILE",
                  "ACCELERATE_USE_DEEPSPEED", "ACCELERATE_USE_CPU",
                  "ACCELERATE_MIXED_PRECISION")


def _subproc_env_single_gpu():
    env = dict(os.environ)
    for k in _DIST_ENV_KEYS:
        env.pop(k, None)
    cvd = env.get("CUDA_VISIBLE_DEVICES", "")
    env["CUDA_VISIBLE_DEVICES"] = cvd.split(",")[0].strip() if cvd else "0"
    return env


def _subproc_env_multi_gpu():
    env = dict(os.environ)
    for k in _DIST_ENV_KEYS:
        env.pop(k, None)
    return env


def _pick_free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

_MOTION_CASE_RE = re.compile(r'[（(]\s*case[-_]?\s*([0-9A-Za-z][0-9A-Za-z_-]*)\s*[)）]', re.IGNORECASE)


def _is_main_process():
    for var in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        if var in os.environ:
            return int(os.environ[var]) == 0
    return True


def classify_adaption(adaption):
    text = str(adaption or "").lower().replace(" ", "")
    if "one-shot(motionmodule)" in text:
        return "motion"
    if "one-shot(temporalmodule)" in text:
        return "temporal"
    return "zero-shot"


def is_no_mask(adaption):
    return "no-mask" in str(adaption or "").lower().replace(" ", "")


def _read_llm_decision(yaml_path):
    try:
        d = OmegaConf.load(yaml_path).get("llm_decision", None)
    except Exception:
        return None
    return dict(d) if d is not None else None


def _strip_llm_decision(yaml_path):
    with open(yaml_path) as f:
        text = f.read()
    new_text = re.sub(r'(?ms)^llm_decision:.*?(?=^\S|\Z)', '', text).rstrip() + '\n'
    if new_text != text:
        with open(yaml_path, "w") as f:
            f.write(new_text)


def _write_llm_decision(yaml_path, adaption, paradigm):
    _strip_llm_decision(yaml_path)
    block = (f"llm_decision:  # auto-written by rank0's DeepSeek decision in multi-GPU runs (overwritten on rerun; ignored when Use_DeepSeek=False)\n"
             f"  adaption: '{adaption}'\n"
             f"  paradigm: {paradigm}\n"
             f"  written_at: {time.time()}\n")
    with open(yaml_path, "a") as f:
        f.write(block)


def _wait_for_file(path, timeout_s, what, fresh_after=None):
    t0 = time.time()
    while True:
        if os.path.exists(path) and os.path.getmtime(path) >= (fresh_after or 0):
            with open(path) as f:
                return f.read()
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"等待 {what} 超时: {path}")
        time.sleep(5)


def _find_tune_yaml(dataset_path):
    from glob import glob as _glob
    want = os.path.normpath(os.path.join(PROJECT_ROOT, str(dataset_path)))
    for cand in sorted(_glob(os.path.join(PROJECT_ROOT, "config", "tune", "*.yaml"))):
        try:
            tp = OmegaConf.load(cand)["train_data"]["video_path"]
        except Exception:
            continue
        have = os.path.normpath(os.path.join(PROJECT_ROOT, str(tp)))
        if have == want:
            rel = os.path.relpath(cand, PROJECT_ROOT)
            print(f"[tune] 未配置 tune-yaml，按数据集 {dataset_path} 自动匹配: {rel}")
            return rel
    raise SystemExit(
        f"[ELVE] temporal 路由需要 tune 配置，但 config/tune/ 下没有 train_data.video_path "
        f"匹配 {dataset_path} 的 yaml。请在编辑配置里显式加一行 tune-yaml: <path>，"
        f"或参照 config/tune/ 现有配置新建（train_data.video_path 需指向同一数据集）。")


def run_tune(tune_config_path):
    tune_cfg = OmegaConf.load(tune_config_path)
    output_dir = tune_cfg["output_dir"]
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)
    output_dir = os.path.normpath(output_dir)
    done_marker = os.path.join(output_dir, TUNE_DONE_MARKER)

    if os.path.isdir(output_dir) and "unet" in os.listdir(output_dir):
        print(f"[tune] 已存在微调模型 {output_dir}，跳过 tune（如需重训请删除该目录）")
    else:
        accelerate_bin = os.path.join(os.path.dirname(sys.executable), "accelerate")
        world = int(os.environ.get("WORLD_SIZE", "1") or 1)
        if world > 1:
            env = _subproc_env_multi_gpu()
            cvd = env.get("CUDA_VISIBLE_DEVICES", "")
            n_vis = len([x for x in cvd.split(",") if x.strip()]) if cvd \
                else torch.cuda.device_count()
            n_proc = max(1, min(world, n_vis))
        else:
            env = _subproc_env_single_gpu()
            n_proc = 1
        if os.path.exists(accelerate_bin):
            launcher = [accelerate_bin, "launch", f"--num_processes={n_proc}"]
            if n_proc > 1:
                launcher.append(f"--main_process_port={_pick_free_port()}")
        elif n_proc > 1:
            raise SystemExit(f"[tune] 多卡 tune 需要 accelerate launch，未找到: {accelerate_bin}")
        else:
            launcher = [sys.executable]
        cmd = launcher + ["run_tuning_1.py", "--config", tune_config_path]
        print(f"[tune] one-shot(temporal module) -> 启动微调({n_proc}进程): {' '.join(cmd)}"
              f"  (CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')})")
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)
        assert "unet" in os.listdir(output_dir), \
            f"tune 结束但 {output_dir} 下没有 unet/，微调产物不完整"
    with open(done_marker, "w") as f:
        f.write(tune_config_path)
    print(f"[tune] 微调完成，pipeline 将加载微调后模型: {output_dir}")
    return output_dir


def launch_motioneditor(me_config=None):
    me_root = os.path.join(PROJECT_ROOT, "MotionEditor")
    me_config = me_config or os.path.join("configs", "case-4", "eval-motion.yaml")
    me_config_abs = me_config if os.path.isabs(me_config) else os.path.join(me_root, me_config)
    if not os.path.exists(me_config_abs):
        raise SystemExit(
            f"[MotionEditor] 配置不存在: {me_config_abs}\n"
            f"  请检查 motioneditor-yaml 字段或 target prompt 里的 case 标记是否写对了")
    eval_cfg = OmegaConf.load(me_config_abs)

    def _me_abs(p):
        return p if os.path.isabs(str(p)) else os.path.normpath(os.path.join(me_root, str(p)))

    def _me_run(script, cfg_rel, hint):
        cmd = [MOTIONEDITOR_PYTHON, script, f"--config={cfg_rel}"]
        print(f"[MotionEditor] {hint}: {' '.join(cmd)}  (cwd={me_root})")
        subprocess.run(cmd, cwd=me_root, check=True, env=_subproc_env_single_gpu())

    video_dir = _me_abs(eval_cfg["input_data"]["video_dir"])
    if not os.path.isdir(video_dir):
        raise SystemExit(
            f"[MotionEditor] 数据目录不存在: {video_dir}\n"
            f"  原视频/姿态条件无法自动补齐，请先按 case 数据集结构准备 "
            f"（images/ source_condition/ target_condition/ frame_list.txt 等）")

    bg_ckpt = _me_abs(eval_cfg["resume_from_checkpoint"])
    adapter_pth = _me_abs(eval_cfg["adapter_weight_path"])
    bg_missing = not (os.path.isdir(bg_ckpt)
                      and any(f.startswith(("model.", "pytorch_model")) for f in os.listdir(bg_ckpt)))
    ad_missing = not os.path.isfile(adapter_pth)
    if not (bg_missing or ad_missing):
        print(f"[MotionEditor] 训练产物齐备:\n"
              f"  背景 checkpoint: {bg_ckpt}\n"
              f"  adapter 权重:    {adapter_pth}\n  直接推理")
    else:
        case_dir = os.path.dirname(me_config)
        train_yamls = {"train_bg.py": os.path.join(case_dir, "train-bg.yaml"),
                       "train_adaptor.py": os.path.join(case_dir, "train-motion.yaml")}
        need = []
        if bg_missing:
            need.append("train_bg.py")
        if bg_missing or ad_missing:
            need.append("train_adaptor.py")
        for script in need:
            yml_abs = train_yamls[script] if os.path.isabs(train_yamls[script]) \
                else os.path.join(me_root, train_yamls[script])
            if not os.path.exists(yml_abs):
                raise SystemExit(
                    f"[MotionEditor] 训练产物缺失且找不到训练配置 {yml_abs}，无法自动补训")
            print(f"[MotionEditor] {'背景 checkpoint' if script == 'train_bg.py' else 'adapter 权重'}"
                  f" 缺失/失效 -> 自动补训（约 5 分钟/级）")
            _me_run(script, train_yamls[script], f"自动补训 {script}")

    _me_run("inference.py", me_config, "one-shot(Motion module) 推理")

    out_dir = _me_abs(eval_cfg["output_dir"])
    if not os.path.isdir(out_dir):
        raise SystemExit(f"[MotionEditor] 推理结束但输出目录不存在: {out_dir}")
    result_root = os.path.join(PROJECT_ROOT, "result")
    dst = os.path.join(result_root, os.path.basename(out_dir))
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(out_dir, dst)
    print(f"[MotionEditor] 结果已复制到 {dst}（原产物仍在 {out_dir}）")


def _extract_motion_case(Omegadict):
    prompts = Omegadict.get("editing_config", {}).get("editing_prompts", []) or []
    for p in prompts:
        m = _MOTION_CASE_RE.search(str(p))
        if m:
            return m.group(1)
    return None


def llm_decision_routing(Omegadict, config):
    paradigm = None
    adaption = None
    motion_case = _extract_motion_case(Omegadict)
    multi_proc = any(v in os.environ for v in ("RANK", "LOCAL_RANK", "SLURM_PROCID"))
    forced = os.environ.get("ELVE_FORCE_PARADIGM", "").strip().lower()
    if forced:
        if forced not in ("zero-shot", "temporal", "motion"):
            raise SystemExit(f"[ELVE] ELVE_FORCE_PARADIGM 只接受 zero-shot/temporal/motion，got {forced!r}")
        print(f"[ELVE] ELVE_FORCE_PARADIGM={forced} 生效：跳过 DeepSeek，直接按该决策路由")
        paradigm = forced
    elif _is_main_process():
        if multi_proc:
            _strip_llm_decision(config)
        source_prompt = Omegadict["dataset_config"]["prompt"]
        target_prompt = list(Omegadict["editing_config"]["editing_prompts"])
        adaption, info = deepseek.LLM_stream(source_prompt, target_prompt)
        paradigm = classify_adaption(adaption)
        if motion_case is not None and paradigm != "motion":
            print(f"[ELVE] target prompt 带 MotionEditor case 标记（case-{motion_case}），"
                  f"LLM 判定为 {paradigm}，强制校正为 motion")
            paradigm = "motion"
            adaption = "no-mask, one-shot(Motion module)"
        if multi_proc:
            _write_llm_decision(config, adaption, paradigm)
    else:
        t_accept = time.time() - 30
        t0 = time.time()
        while True:
            d = _read_llm_decision(config)
            if d and d.get("written_at", 0) >= t_accept and d.get("paradigm"):
                paradigm = d["paradigm"]
                adaption = d.get("adaption", "")
                break
            if time.time() - t0 > 600:
                raise TimeoutError(f"等待 rank0 写入 yaml 的 LLM 决策超时: {config}")
            time.sleep(5)

    if paradigm == "motion":
        if _is_main_process():
            me_yaml = Omegadict.get("motioneditor-yaml", None)
            if me_yaml is None and motion_case is not None:
                me_yaml = os.path.join("configs", f"case-{motion_case}", "eval-motion.yaml")
            launch_motioneditor(me_yaml)
        return None

    overrides = {}
    if is_no_mask(adaption):
        overrides["memory_frames"] = 8
        p2p_entries = Omegadict.get("editing_config", {}).get("p2p_config", None) or {}
        if hasattr(p2p_entries, "keys") and len(p2p_entries):
            overrides["editing_config"] = {
                "p2p_config": {k: {"blend_words": None} for k in p2p_entries}
            }
        print("[ELVE] no-mask(全局编辑)：memory_frames=8（特征分级存储，单卡超 8 帧落盘）；"
              "blend_words 置 None 禁用 mask 机制（yaml 原值不生效，等效注释）")
    if paradigm == "temporal":
        overrides["method"] = "FateZero"
        tune_yaml = Omegadict.get("tune-yaml", None)
        if tune_yaml is None:
            tune_yaml = _find_tune_yaml(Omegadict["dataset_config"]["path"])
        if _is_main_process():
            tuned_dir = run_tune(tune_yaml)
        else:
            tune_output = OmegaConf.load(tune_yaml)["output_dir"]
            if not os.path.isabs(tune_output):
                tune_output = os.path.join(PROJECT_ROOT, tune_output)
            _wait_for_file(os.path.join(tune_output, TUNE_DONE_MARKER), timeout_s=4 * 3600,
                           what="rank0 的 tune 完成", fresh_after=_T_E2E_START - 5)
            tuned_dir = tune_output
        overrides["pretrained_model_path"] = tuned_dir
    else:
        overrides["pretrained_model_path"] = DEFAULT_PRETRAINED
    return overrides


def run_motioneditor(cfg):
    me_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MotionEditor')
    orig_cwd = os.getcwd()
    os.chdir(me_dir)
    sys.path.insert(0, me_dir)
    try:
        me_cfg = {k: v for k, v in dict(cfg).items() if k != 'method'}
        import diffusers
        print(f"[MotionEditor] diffusers = {diffusers.__file__}")
        from inference import main as me_main
        me_main(**me_cfg)
    finally:
        os.chdir(orig_cwd)
        if me_dir in sys.path:
            sys.path.remove(me_dir)


def control_preprocess(image_tensor, accelerator, controlnet, prepare_control_image,Control_image_path,control_class):
    if len(os.listdir(Control_image_path)) > 0:
                control_torch = torch.load(os.path.join(Control_image_path, 'control.pt')).to(accelerator.device)
    else:
        image_pil_list = []
        
        for i in range(image_tensor.shape[0]):
            img_tensor = image_tensor[i]
            
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.permute(1, 2, 0)
            
            if img_tensor.max() <= 1.0:
                img_array = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            else:
                img_array = img_tensor.cpu().numpy().astype(np.uint8)
            
            pil_image = Image.fromarray(img_array)
            image_pil_list.append(pil_image)
        
        control_torch_list = []
        for image_pil in image_pil_list:
            width, height = image_pil.size
            list_of_image_pils = fu.pil_grid_to_frames(image_pil, grid_size=[2,2]) # List[C, W, H] -> len = num_frames
            list_of_pils = [pu.pixel_perfect_process(np.array(frame_pil, dtype='uint8'), control_class) for frame_pil in list_of_image_pils]
            control_images = np.array(list_of_pils)
            control_img = ipu.create_grid_from_numpy(control_images, grid_size=[2,2])
            control_pil = PIL.Image.fromarray(control_img).convert("L")
            control_image = prepare_control_image(
                image=control_pil,
                width=width,
                height=height,
                device=accelerator.device,
                dtype=controlnet.dtype,
                batch_size=1,
                num_images_per_prompt=1
            )
            control_torch_list.append(control_image)

        control_torch = torch.cat(control_torch_list, dim=0).to(accelerator.device)
        torch.save(control_torch, os.path.join(Control_image_path, 'control.pt'))
        pu.unload_preprocess_models()
        print("control_torch.shape=",control_torch.shape)
    return control_torch

def init_model(pretrained_model_path, method, grid_sizeH, grid_sizeW, model_config,
               test_pipeline_config, video_length, accelerator, data_path, num_inference_steps, device,control_class,
               memory_frames: int = 16):
    
    import diffusers
    from diffusers import AutoencoderKL, DDIMScheduler
    from diffusers.utils.import_utils import is_xformers_available

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if method == "FateZero":
        from video_diffusion.models.unet_3d_condition import UNet3DConditionModel
        if accelerator.is_main_process: 
            print("pretrained_model_path=",pretrained_model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_path,
            subfolder="tokenizer",
            use_fast=False,
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_path,
            subfolder="text_encoder",
        )
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_path,
            subfolder="vae",
        )
        unet = UNet3DConditionModel.from_2d_model(
            os.path.join(pretrained_model_path, "unet"), model_config=model_config, 
            grid_sizeH=grid_sizeH, grid_sizeW=grid_sizeW
        )
        scheduler = DDIMScheduler.from_pretrained(
            pretrained_model_path,
            subfolder="scheduler",
        )

        controlnet = None
        prepare_control_image = None
        scheduler_config=scheduler.config
        pipeline = instantiate_from_config(
            test_pipeline_config,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            controlnet=controlnet, 
            method=method,
            control_class=control_class,
            disk_store=False,
            video_length=video_length,
            grid_sizeH=grid_sizeH,
            grid_sizeW=grid_sizeW,
            accelerator=accelerator,
            data_path=data_path,
            num_inference_steps=num_inference_steps,
            memory_frames=memory_frames
        )

    elif method == "ControlNet":
        print("ControlNet Model Path",const.PREPROCESSOR_DICT[control_class])
        from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
        controlnet = ControlNetModel.from_pretrained(
           const.PREPROCESSOR_DICT[control_class], #depth canny lineart softedge 
            torch_dtype=weight_dtype
        ).to(device, weight_dtype)
        SDModel_Path="/mnt/fspace/hproject/model/stable-diffusion-v1-5"
        print("SDModel_Path=",SDModel_Path)
        pipeline_cn = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            SDModel_Path, 
            controlnet=controlnet, 
            torch_dtype=weight_dtype
        ).to(device, weight_dtype)
        prepare_control_image = pipeline_cn.prepare_control_image
        vae = pipeline_cn.vae
        unet = pipeline_cn.unet
        text_encoder = pipeline_cn.text_encoder
        scheduler = pipeline_cn.scheduler
        tokenizer = pipeline_cn.tokenizer
        scheduler_config = pipeline_cn.scheduler.config
        pipeline_cn.set_progress_bar_config(disable=True)

        pipeline = instantiate_from_config(
            test_pipeline_config,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            controlnet=controlnet, 
            method=method,
            control_class=control_class,
            disk_store=False,
            video_length=video_length,
            grid_sizeH=grid_sizeH,
            grid_sizeW=grid_sizeW,
            accelerator=accelerator,
            data_path=data_path,
            num_inference_steps=num_inference_steps
        )
    
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    scheduler.set_timesteps(num_inference_steps)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if is_xformers_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print("xformers加载异常")  

    return pipeline, unet, vae, text_encoder, tokenizer, weight_dtype, controlnet, prepare_control_image,scheduler_config


def _save_edited_only_gif(logdir, step=0, idx=0):
    sample_dir = os.path.join(logdir, 'sample')
    per_gpu = []
    for p in glob(os.path.join(sample_dir, f'step_{step}_{idx}_0_GPU*.gif')):
        if 'atten' in os.path.basename(p):
            continue
        m = re.search(r'GPU(\d+)\.gif$', p)
        if m:
            per_gpu.append((int(m.group(1)), p))
    per_gpu.sort()
    frames_all = []
    duration = 100
    for _, p in per_gpu:
        gif = Image.open(p)
        n = 0
        try:
            while True:
                frames_all.append(gif.copy())
                n += 1
                gif.seek(n)
        except EOFError:
            pass
        duration = gif.info.get('duration', duration)
    if not frames_all:
        print(f"[ELVE] 未找到纯编辑帧 gif（step_{step}_{idx}_0_GPU*.gif），跳过 edited 拼接")
        return
    out = os.path.join(sample_dir, 'step1_edited.gif')
    frames_all[0].save(out, save_all=True, append_images=frames_all[1:],
                       loop=0, duration=duration)
    print(f"[ELVE] 纯编辑帧汇总已写出: {out}（{len(per_gpu)} 卡共 {len(frames_all)} 帧）")


def test(
    config: str,
    pretrained_model_path: str,
    method: str,
    control_class: str,
    Use_DeepSeek: bool,
    dataset_config: Dict,
    logdir: str = None,
    editing_config: Optional[Dict] = None,
    test_pipeline_config: Optional[Dict] = None,
    gradient_accumulation_steps: int = 1,
    seed: Optional[int] = None,
    mixed_precision: Optional[str] = "fp16",
    batch_size: int = 1,
    model_config: dict = {},
    verbose: bool = True,
    memory_frames: int = 16,
    **kwargs
):
    with diffusers_context(method):
        import diffusers
        #print("________________", diffusers.__file__)
    source_prompt=dataset_config["prompt"]
    target_prompt=editing_config["editing_prompts"]
    args = get_function_args()
    grid_sizeH = editing_config["grid_sizeH"]
    grid_sizeW = editing_config["grid_sizeW"]
    _target = test_pipeline_config.get("target", "") if test_pipeline_config else ""
    if method == "FateZero" and "spatial_temp" in _target:
        test_pipeline_config["target"] = "video_diffusion.pipelines.p2p_ddim_fatezero.P2pDDIMFateZeroPipeline"
        print("[ELVE] method=FateZero：target 已自动从 ControlNet pipeline 切换为 P2pDDIMFateZeroPipeline")
    elif method == "ControlNet" and "fatezero" in _target:
        test_pipeline_config["target"] = "video_diffusion.pipelines.p2p_ddim_spatial_temp.P2pDDIMSpatiotempPipeline"
        print("[ELVE] method=ControlNet：target 已自动从 FateZero pipeline 切换为 P2pDDIMSpatiotempPipeline")
    controlnet_conditioning_scale = 1.0
    video_length = dataset_config["n_sample_frame"]
    time_string = get_time_string()
    GPU_num=torch.cuda.device_count()
    if GPU_num>1:
        if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
            raise SystemExit(
                f"[ELVE] 检测到 {GPU_num} 张 GPU，但当前是裸 python 启动（无 RANK/WORLD_SIZE），"
                f"无法走 torch.distributed 路径。\n"
                f"  单卡运行: CUDA_VISIBLE_DEVICES=0 python test_fatezero.py --config <yaml>\n"
                f"  多卡运行: CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 "
                f"test_fatezero.py --config <yaml>")
        with elve_timing.timer('comm_init_group'):
            dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
        _lr = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', 0)))
        _ts_enc = torch.tensor([int(time_string[:6] + time_string[7:])], dtype=torch.int64,
                               device=f'cuda:{_lr}')
        with elve_timing.timer('comm_bcast_ts'):
            dist.broadcast(_ts_enc, src=0)
        _ts_v = int(_ts_enc.item())
        time_string = f"{_ts_v // 10000:06d}-{_ts_v % 10000:04d}"
    Control_image_path = None
    if method == "ControlNet":
        _parts = ['Control', f"{GPU_num}GPU", control_class]
        if GPU_num > 1:
            _parts.append(str(dist.get_rank()))
        _parts += [os.path.basename(dataset_config["path"]), str(video_length)]
        Control_image_path = os.path.join(*_parts)
        os.makedirs(Control_image_path, exist_ok=True)
    if logdir is None:
        logdir = config.replace('config', 'result').replace('.yml', '').replace('.yaml', '')
    logdir += f"_{time_string}"
    
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
  
    if accelerator.is_main_process:
        os.makedirs(logdir, exist_ok=True)
        OmegaConf.save(args, os.path.join(logdir, "config.yml"))
        


    pipeline, unet, vae, text_encoder, tokenizer, weight_dtype, controlnet, prepare_control_image,scheduler_config = init_model(
        pretrained_model_path, method, grid_sizeH, grid_sizeW, model_config,
        test_pipeline_config, video_length, accelerator, dataset_config["path"],
        editing_config['num_inference_steps'], accelerator.device,control_class,
        memory_frames)

    prompt_ids = tokenizer(
        dataset_config["prompt"],
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids

    video_dataset = ImageSequenceDataset(**dataset_config, prompt_ids=prompt_ids, GPU_num=torch.cuda.device_count())
    train_dataloader = torch.utils.data.DataLoader(
        video_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )
    
    unet, train_dataloader = accelerator.prepare(unet, train_dataloader)
    
    if accelerator.is_main_process:
        accelerator.init_trackers("video")  

    validation_sample_logger = P2pSampleLogger(
        **editing_config, 
        logdir=logdir, 
        source_prompt=dataset_config['prompt'],
        controlnet_conditioning_scale=controlnet_conditioning_scale
    )
   
    for step, batch in enumerate(train_dataloader):
        assert batch["images"].shape[0] == 1, "Only support, overfiting on a single video"
        print("frames_per_GPU:", batch['images'].shape[2])
        if batch['images'].shape[2] * torch.cuda.device_count() != video_length:
            print(f"[WARN] n_sample_frame={video_length} 不能被 GPU 数整除，"
                  f"数据集按 floor 切分已丢弃尾部帧（当前保留 {batch['images'].shape[2] * torch.cuda.device_count()} 帧）")
        
        vae.eval()
        text_encoder.eval()
        unet.eval()
        
        if method == "ControlNet":
            images=batch["images"]
            n_real_d = images.shape[2]
            n_pad_d = (-n_real_d) % (grid_sizeH * grid_sizeW)
            if n_pad_d:
                print(f'[ControlNet] 深度图生成：帧数 {n_real_d} 非 {grid_sizeH*grid_sizeW} 整数倍，'
                      f'复制末帧补齐到 {n_real_d + n_pad_d}')
                images = torch.cat([images, images[:, :, -1:, :, :].expand(-1, -1, n_pad_d, -1, -1)], dim=2)
            images=grid.combine_frames_in_grid(images,grid_sizeH,grid_sizeW)
            images_for_depth = rearrange(images.to(dtype=weight_dtype), "b c f h w -> (b f) c h w")
            time1=time.time()
            print("start control image make")
            depth_images = control_preprocess(images_for_depth,accelerator,controlnet,prepare_control_image,Control_image_path,control_class)
            time2=time.time()
            print("end control image make")
            print("control_image time cost=",time2-time1)
            depth_images=rearrange(depth_images,"(b f) c h w -> b c f h w", b=1)
            depth_images=grid.restore_frames_from_grid(depth_images,grid_sizeH,grid_sizeW)
            if n_pad_d:
                depth_images = depth_images[:, :, :n_real_d, :, :]
            depth_images=rearrange(depth_images,"b c f h w -> (b f) c h w")
            depth_images = depth_images.to(device=accelerator.device, dtype=weight_dtype)
        elif method == "FateZero":
            depth_images = None
        
        text_embeddings = pipeline._encode_prompt(
            dataset_config.prompt,
            device=accelerator.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=None
        )
        
        use_inversion_attention = editing_config.get('use_inversion_attention', False)
        controlnet_conditioning_scale = 1.0

        _t_inv0 = time.time()
        if method == "FateZero":
            p2p_cfg = editing_config['p2p_config'][0]
            _blend_words = p2p_cfg.get('blend_words', None)
            _blender_on = _blend_words is not None and _blend_words != 'None'
            editing_params = dict(
                source_prompt=dataset_config.prompt,
                prompt=editing_config['editing_prompts'][0],
                cross_replace_steps=p2p_cfg.get('cross_replace_steps'),
                self_replace_steps=p2p_cfg.get('self_replace_steps', 0.0),
                blend_latents=_blender_on and p2p_cfg.get('blend_latents', False),
                latent_replace_steps=p2p_cfg.get('latent_replace_steps', (0.2, 0.8)),
            )
            batch['ddim_init_latents'] = pipeline.prepare_latents_ddim_inverted(
                rearrange(batch["images"].to(dtype=weight_dtype), "b c f h w -> (b f) c h w"),
                text_embeddings=text_embeddings,
                scheduler_config=scheduler_config,
                editing_params=editing_params,
                num_inference_steps=editing_config['num_inference_steps'],
            )
        else:
            batch['ddim_init_latents'] = pipeline.prepare_latents_ddim_inverted(
                rearrange(batch["images"].to(dtype=weight_dtype), "b c f h w -> (b f) c h w"),
                batch_size=1,
                num_images_per_prompt=1,
                text_embeddings=text_embeddings,
                prompt=dataset_config.prompt,
                LOW_RESOURCE=True,
                save_path=logdir if verbose else None,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                control_image=depth_images,
                scheduler_config=scheduler_config
            )
        print(f"[TIMING] rank={accelerator.process_index} use_time_Inversion={time.time()-_t_inv0:.4f}", flush=True)
        
        vae.eval()
        text_encoder.eval()
        unet.eval()
        
        images = batch["images"].to(dtype=weight_dtype)
        images = rearrange(images, "b c f h w -> (b f) c h w")

        if accelerator is not None:
            _t_den0 = time.time()
            validation_sample_logger.log_sample_images(
                image=images,
                pipeline=pipeline,
                device=accelerator.device,
                step=0,
                latents=batch['ddim_init_latents'],
                save_dir=logdir if verbose else None,
                control_image=depth_images,
            )
            print(f"[TIMING] rank={accelerator.process_index} use_time_Denoising={time.time()-_t_den0:.4f}", flush=True)

        if elve_timing.ENABLED:
            print(elve_timing.format_report(f'rank={accelerator.process_index}'), flush=True)
            elve_timing.dump(os.path.join(logdir, f'elve_timing_rank{accelerator.process_index}.json'))

    if torch.cuda.device_count() > 1:
        dist.barrier()
        rank = dist.get_rank()
        if rank == 0:
            per_gpu = []
            for p in glob(f"{logdir}/sample/step_0_GPU*.gif"):
                m = re.search(r'GPU(\d+)\.gif$', p)
                if m:
                    per_gpu.append((int(m.group(1)), p))
            per_gpu.sort()

            combined_frames = []
            duration = 100
            for _, p in per_gpu:
                gif = Image.open(p)
                n = 0
                try:
                    while True:
                        combined_frames.append(gif.copy())
                        n += 1
                        gif.seek(n)
                except EOFError:
                    pass
                duration = gif.info.get('duration', duration)

            if not combined_frames:
                print(f"[ELVE] 未找到任何 step_0_GPU*.gif，跳过 step1.gif 拼接: {logdir}/sample")
            else:
                output_path = f"{logdir}/sample/step1.gif"
                combined_frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=combined_frames[1:],
                    loop=0,
                    duration=duration
                )
            _save_edited_only_gif(logdir)
    else:
        _save_edited_only_gif(logdir)

    accelerator.end_training()

@click.command()
@click.option("--config", type=str, default="config/sample.yml")
def run(config):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    Omegadict = OmegaConf.load(config)
    method = Omegadict.get('method')

    if not method:
        raise ValueError("Method parameter is required in config")

    # ============================================================
    # ============================================================
    llm_overrides = {}
    if method == "FateZero" and Omegadict.get('Use_DeepSeek', False):
        llm_overrides = llm_decision_routing(Omegadict, config)
        if llm_overrides is None:
            return
        if 'method' in llm_overrides:
            method = llm_overrides['method']

    with diffusers_context(method):
        Omegadict = OmegaConf.load(config)
        if llm_overrides:
            Omegadict.merge_with(OmegaConf.create(llm_overrides))

        if method == "MotionEditor":
            run_motioneditor(Omegadict)
            return

        if 'unet' in os.listdir(Omegadict['pretrained_model_path']):
            test(config=config, **Omegadict)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", module="torchvision.io.image")
    _code = 0
    try:
        run()
    except SystemExit as e:
        _code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    _rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))
    print(f"[TIMING] rank={_rank} use_time_EndToEnd={time.time()-_T_E2E_START:.4f}", flush=True)
    sys.exit(_code)