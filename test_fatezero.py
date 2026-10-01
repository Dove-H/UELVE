import os
import sys
import warnings
warnings.filterwarnings("ignore")
from glob import glob
from typing import Optional, Dict
from omegaconf import OmegaConf
import click
import json
import subprocess
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
import numpy as np
import utils.preprocesser_utils as pu
import utils.image_process_utils as ipu
import utils.feature_utils as fu
from mask_utils import gen_masks


# diffusers 路径管理函数
def setup_diffusers_path(method):
    """根据 method 参数设置不同的 diffusers 路径"""
    if method == "FateZero":
        local_diffusers_path = '/mnt/fspace/hproject/ELVE/diffusersF/'
    elif method == "ControlNet":
        local_diffusers_path = '/mnt/fspace/hproject/ELVE/diffusersC/'
    elif method == "MotionEditor":
        # MotionEditor 用 pip 装的标准 diffusers 0.15.1，不切 fork（独立 pipeline）
        return None
    else:
        raise ValueError(f"Unknown method: {method}")
    if local_diffusers_path in sys.path:
        sys.path.remove(local_diffusers_path)  
    sys.path.insert(0, local_diffusers_path)
    return local_diffusers_path

def clear_diffusers_cache():
    """只清理 diffusers 和 transformers 相关模块缓存，不清理 torch"""
    modules_to_remove = []
    for module_name in list(sys.modules.keys()):
        if (module_name.startswith('diffusers') or 
            module_name.startswith('transformers')):
            if module_name != 'transformers' and not module_name.startswith('transformers.'):
                modules_to_remove.append(module_name)
    for module_name in modules_to_remove:
        del sys.modules[module_name]

@contextmanager
def diffusers_context(method):
    """diffusers 版本切换的上下文管理器"""
    local_diffusers_path = setup_diffusers_path(method)
    clear_diffusers_cache()

    try:
        yield local_diffusers_path
    finally:
        pass


# ============================================================
# LLM 语义分析与模型适配路由（论文 Sec 3.3 / Figure 3）：
# pipeline 启动前，由 LLM 对比 source/target prompt 流式输出决策。
# model_adaption 前置 -> 不等 Thinking 生成完即可路由：
#   zero-shot                  -> 不跑 tune，直接加载预训练模型
#   one-shot(temporal module)  -> 自主启动 tune，跑完加载微调后的模型文件
#   one-shot(Motion module)    -> 启动外置 MotionEditor 模块
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PRETRAINED = "/mnt/fspace/hproject/model/stable-diffusion-v1-5"
# MotionEditor 独立 conda env（py3.10 + diffusers 0.15.1），与 ELVE env 隔离，
# 用绝对路径调用以避开 PATH 被 ELVE env 劫持的问题
MOTIONEDITOR_PYTHON = "/home/user4/miniconda3/envs/motioneditor/bin/python3.10"
TUNE_DONE_MARKER = "tune_done.marker"


def _is_main_process():
    """多进程(accelerate launch)时仅 RANK/LOCAL_RANK=0 做 LLM 决策与 tune，
    其余进程读它写出的决策文件并等待微调产物；单进程直跑时恒为 True。"""
    for var in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        if var in os.environ:
            return int(os.environ[var]) == 0
    return True


def classify_adaption(adaption):
    """把 model_adaption 文本归一化为三种路由之一：
    'motion' | 'temporal' | 'zero-shot'（对应论文 Figure 3a 的三种模型适配）。"""
    text = str(adaption or "").lower().replace(" ", "")
    if "one-shot(motionmodule)" in text:
        return "motion"
    if "one-shot(temporalmodule)" in text:
        return "temporal"
    return "zero-shot"


def _decision_path(config):
    """rank0 与其余进程交换 LLM 决策的确定性文件路径（多 GPU 时用）"""
    return config + ".llm_decision"


def _wait_for_file(path, timeout_s, what, fresh_after=None):
    """轮询等待文件出现并返回其内容；fresh_after 用于拒绝上一轮运行留下的旧文件。"""
    t0 = time.time()
    while True:
        if os.path.exists(path) and os.path.getmtime(path) >= (fresh_after or 0):
            with open(path) as f:
                return f.read()
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"等待 {what} 超时: {path}")
        time.sleep(5)


def run_tune(tune_config_path):
    """阶段2：one-shot(temporal module) —— 自主启动 tune（微调膨胀后的时空注意力，
    论文 Sec 3.3 "if a big shape change occurs, UELVE just finetunes the inflated
    spatial-temporal attention"）。跑完返回微调模型目录，pipeline 直接加载它。

    run_tuning_1.py 结束时会将完整 pipeline（含微调后的 unet）save_pretrained 到
    tune yaml 的 output_dir；沿用 read.sh 的启动方式（accelerate launch）。
    """
    tune_cfg = OmegaConf.load(tune_config_path)
    output_dir = tune_cfg["output_dir"]
    if not os.path.isabs(output_dir):
        # tune yaml 内的相对路径以项目根目录为基准
        output_dir = os.path.join(PROJECT_ROOT, output_dir)
    output_dir = os.path.normpath(output_dir)
    done_marker = os.path.join(output_dir, TUNE_DONE_MARKER)

    if os.path.isdir(output_dir) and "unet" in os.listdir(output_dir):
        print(f"[tune] 已存在微调模型 {output_dir}，跳过 tune（如需重训请删除该目录）")
    else:
        accelerate_bin = os.path.join(os.path.dirname(sys.executable), "accelerate")
        launcher = [accelerate_bin, "launch", "--num_processes=1"] \
            if os.path.exists(accelerate_bin) else [sys.executable]
        cmd = launcher + ["run_tuning_1.py", "--config", tune_config_path]
        # 自主启动固定单卡：run_tuning_1.py 的多卡分支（_set_static_graph /
        # alternate_multiGPU）要求分布式多进程启动，单进程 + 多卡可见会崩溃
        env = dict(os.environ)
        cvd = env.get("CUDA_VISIBLE_DEVICES", "")
        env["CUDA_VISIBLE_DEVICES"] = cvd.split(",")[0].strip() if cvd else "0"
        print(f"[tune] one-shot(temporal module) -> 启动微调: {' '.join(cmd)}"
              f"  (CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']})")
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)
        assert "unet" in os.listdir(output_dir), \
            f"tune 结束但 {output_dir} 下没有 unet/，微调产物不完整"
    with open(done_marker, "w") as f:  # 供其余 GPU 进程判断 tune 已完成
        f.write(tune_config_path)
    print(f"[tune] 微调完成，pipeline 将加载微调后模型: {output_dir}")
    return output_dir


def launch_motioneditor(me_config=None):
    """阶段3：one-shot(Motion module) —— 启动外置 MotionEditor 模块。

    MotionEditor 尚未融合进 ELVE pipeline（独立代码库，依赖 py3.10 +
    diffusers 0.15.1 的 conda env motioneditor），因此用子进程 + 绝对解释器
    路径启动。默认跑 case-4（checkpoints 已就绪，直接推理即可验证模块启动）。
    """
    me_root = os.path.join(PROJECT_ROOT, "MotionEditor")
    me_config = me_config or os.path.join("configs", "case-4", "eval-motion.yaml")
    cmd = [MOTIONEDITOR_PYTHON, "inference.py", f"--config={me_config}"]
    print(f"[MotionEditor] one-shot(Motion module) -> 外置模块启动: {' '.join(cmd)}  (cwd={me_root})")
    subprocess.run(cmd, cwd=me_root, check=True)


def llm_decision_routing(Omegadict, config):
    """pipeline 启动前的 LLM 语义分析入口。

    流式调用 DeepSeek（model_adaption 前置输出，不等 Thinking 生成完），
    按解析结果决定模型配置，返回需要覆盖到 edit 配置上的字段 dict；
    判定 one-shot(Motion module) 时直接启动外置 MotionEditor 并返回 None。
    """
    paradigm = None
    if _is_main_process():
        # 先清掉上一轮的决策文件，避免多 GPU 时其余进程读到旧决策
        if os.path.exists(_decision_path(config)):
            os.remove(_decision_path(config))
        source_prompt = Omegadict["dataset_config"]["prompt"]
        target_prompt = list(Omegadict["editing_config"]["editing_prompts"])
        adaption, info = deepseek.LLM_stream(source_prompt, target_prompt)
        paradigm = classify_adaption(adaption)
        print(f"[LLM] model_adaption = {adaption!r} -> 路由: {paradigm} "
              f"(较全量输出提前 {info['t_total_ms'] - info['t_adaption_ms']:.0f} ms 拿到决策)")
        with open(_decision_path(config), "w") as f:  # 多 GPU 时同步给其余进程
            json.dump({"adaption": adaption, "paradigm": paradigm}, f)
    else:
        decision = _wait_for_file(_decision_path(config), timeout_s=600,
                                  what="rank0 的 LLM 决策", fresh_after=time.time() - 2)
        paradigm = json.loads(decision)["paradigm"]
        print(f"[LLM] 其余进程读取 rank0 决策 -> 路由: {paradigm}")

    if paradigm == "motion":
        # 阶段3：MotionEditor 为外置模块，启动后即返回，不进入 ELVE pipeline
        if _is_main_process():
            launch_motioneditor(Omegadict.get("motioneditor-yaml", None))
        return None

    overrides = {}
    if paradigm == "temporal":
        # 阶段2：先自主跑 tune，再加载 tune 后的模型文件（本阶段固定走 FateZero 模式）
        overrides["method"] = "FateZero"
        tune_yaml = Omegadict.get("tune-yaml", os.path.join("config", "tune", "child-tune.yaml"))
        if _is_main_process():
            tuned_dir = run_tune(tune_yaml)
        else:
            tune_output = OmegaConf.load(tune_yaml)["output_dir"]
            if not os.path.isabs(tune_output):
                tune_output = os.path.join(PROJECT_ROOT, tune_output)
            _wait_for_file(os.path.join(tune_output, TUNE_DONE_MARKER), timeout_s=4 * 3600,
                           what="rank0 的 tune 完成", fresh_after=time.time() - 2)
            tuned_dir = tune_output
        overrides["pretrained_model_path"] = tuned_dir
    else:
        # zero-shot：无需 tune，直接加载默认预训练模型
        overrides["pretrained_model_path"] = DEFAULT_PRETRAINED
    return overrides


def run_motioneditor(cfg):
    """MotionEditor 推理：走独立 pipeline（pipeline_motion_editor），与 ELVE 的 fatezero/controlnet 不交叉。
    MotionEditor 用标准 diffusers 0.15.1（setup_diffusers_path 不切 fork；diffusers_context 已 clear cache）。
    在独立 accelerate launch 进程跑，与 ELVE 进程互不干扰。"""
    me_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MotionEditor')
    orig_cwd = os.getcwd()
    os.chdir(me_dir)  # MotionEditor 用相对路径（checkpoints/、data/），切到其目录
    sys.path.insert(0, me_dir)  # 让 `from inference import main` / `import motion_editor` 找到
    try:
        me_cfg = {k: v for k, v in dict(cfg).items() if k != 'method'}  # 去掉 ELVE 专属的 method 字段
        import diffusers  # 确认拿到 pip 0.15.1（未切 fork）
        print(f"[MotionEditor] diffusers = {diffusers.__file__}")
        from inference import main as me_main
        me_main(**me_cfg)
    finally:
        os.chdir(orig_cwd)
        if me_dir in sys.path:
            sys.path.remove(me_dir)



def control_preprocess(image_tensor, accelerator, controlnet, prepare_control_image,Control_image_path,control_class):
    """
    处理输入的四维张量，转换为PIL图像列表进行深度预处理
    image_tensor: 形状为 (batch_size * frames, channels, height, width) 的张量
    """
    if len(os.listdir(Control_image_path)) > 0:
                control_torch = torch.load(os.path.join(Control_image_path, 'control.pt')).to(accelerator.device)
    else:
        # 将张量转换为PIL图像列表
        image_pil_list = []
        
        # 遍历张量中的每个图像
        for i in range(image_tensor.shape[0]):
            # 获取单个图像张量 [C, H, W]
            img_tensor = image_tensor[i]
            
            # 将张量从 [C, H, W] 转换为 [H, W, C]
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.permute(1, 2, 0)
            
            # 转换为numpy数组并确保值范围在[0, 255]
            if img_tensor.max() <= 1.0:
                img_array = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            else:
                img_array = img_tensor.cpu().numpy().astype(np.uint8)
            
            # 转换为PIL图像
            pil_image = Image.fromarray(img_array)
            image_pil_list.append(pil_image)
        
        control_torch_list = []
        # 遍历输入的PIL图像列表
        for image_pil in image_pil_list:
            # 获取PIL图像的宽度和高度
            width, height = image_pil.size
            # 对PIL图像进行预处理，生成控制图像
            # 将一个PIL图像转换为一系列帧并放入帧列表，grid_size=[2,2]
            list_of_image_pils = fu.pil_grid_to_frames(image_pil, grid_size=[2,2]) # List[C, W, H] -> len = num_frames
            # 对每个帧进行像素完美处理，处理方式由self.preprocess_name指定
            list_of_pils = [pu.pixel_perfect_process(np.array(frame_pil, dtype='uint8'), control_class) for frame_pil in list_of_image_pils]
            # 将处理后的帧列表转换为numpy数组
            control_images = np.array(list_of_pils)
            # 从numpy数组创建一个新的网格（四合一的大numpy向量），网格大小由self.grid决定
            control_img = ipu.create_grid_from_numpy(control_images, grid_size=[2,2])
            # 将网格转换为灰度图像
            control_pil = PIL.Image.fromarray(control_img).convert("L")
            # 根据控制图像、宽度和高度准备控制图像
            control_image = prepare_control_image(
                image=control_pil,
                width=width,
                height=height,
                device=accelerator.device,
                dtype=controlnet.dtype,
                batch_size=1,
                num_images_per_prompt=1
            )
            # 将控制图像添加到控制图像列表中
            control_torch_list.append(control_image)

        # 将控制图像列表中的torch张量按行拼接，并移动到指定设备上
        control_torch = torch.cat(control_torch_list, dim=0).to(accelerator.device)
        torch.save(control_torch, os.path.join(Control_image_path, 'control.pt'))
        print("control_torch.shape=",control_torch.shape)
    return control_torch

def init_model(pretrained_model_path, method, grid_sizeH, grid_sizeW, model_config,
               test_pipeline_config, video_length, accelerator, data_path, num_inference_steps, device,control_class):
    
    # 动态导入 diffusers 相关模块
    import diffusers
    from diffusers import AutoencoderKL, DDIMScheduler
    from diffusers.utils.import_utils import is_xformers_available

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if method == "FateZero":
        # FateZero的模型初始化
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
            num_inference_steps=num_inference_steps
        )
 
    elif method == "ControlNet":
        # ControlNet的模型初始化
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
    **kwargs
):
    # 使用上下文管理器设置 diffusers 环境
    with diffusers_context(method):
        import diffusers
        #print("________________", diffusers.__file__)
    source_prompt=dataset_config["prompt"]
    target_prompt=editing_config["editing_prompts"]
    args = get_function_args()
    grid_sizeH = editing_config["grid_sizeH"]
    grid_sizeW = editing_config["grid_sizeW"]
    controlnet_conditioning_scale = 1.0
    video_length = dataset_config["n_sample_frame"]
    time_string = get_time_string()
    GPU_num=torch.cuda.device_count()
    if GPU_num>1:
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
        Control_image_path = os.path.join('Control', str(GPU_num) + "GPU",control_class,str(dist.get_rank()), os.path.basename(dataset_config["path"]),str(video_length) )
    else:
        Control_image_path = os.path.join('Control',str(GPU_num) + "GPU",control_class,os.path.basename(dataset_config["path"]),str(video_length)  )
    os.makedirs(Control_image_path, exist_ok = True)
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
        
    # LLM 语义分析与决策已在启动部分完成（llm_decision_routing，见 run()）：
    # model_adaption 的路由结果（是否 tune、加载哪个模型）通过
    # pretrained_model_path / method 覆盖传入，此处不再重复调用 LLM。



    pipeline, unet, vae, text_encoder, tokenizer, weight_dtype, controlnet, prepare_control_image,scheduler_config = init_model(
        pretrained_model_path, method, grid_sizeH, grid_sizeW, model_config,
        test_pipeline_config, video_length, accelerator, dataset_config["path"],
        editing_config['num_inference_steps'], accelerator.device,control_class)

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
   
    # ---- External mask (GroundingDINO + FastSAM) for latent blending ----
    use_external_mask = editing_config.get('use_external_mask', False)
    mask_source_word = editing_config.get('mask_source_word', None)
    mask_model_path = editing_config.get('mask_model_path', gen_masks.DEFAULT_MASK_MODEL_PATH)
    mask_box_threshold = editing_config.get('mask_box_threshold', 0.30)
    mask_dilate_px = editing_config.get('mask_dilate_px', 4)
    # Option C: 外接源物体mask(FastSAM)与cross-attention目标物体mask取并集。仅当 use_external_mask=True 时有意义。
    #   False(默认): 外接mask 直接替换 cross-attention mask(bypass)。
    #   True: 外接mask ∪ cross-attention mask,用于目标物体(如pizza)在源视频里不存在、无法分割的场景。
    fuse_target_attention = editing_config.get('fuse_target_attention', False)
    pipeline.fuse_target_attention = fuse_target_attention
    mask_models = None
    if method == "FateZero" and use_external_mask:
        if not mask_source_word:
            print("[mask] use_external_mask=True but mask_source_word is empty; falling back to cross-attention mask")
            use_external_mask = False
        else:
            print(f"[mask] loading GroundingDINO+FastSAM for source word '{mask_source_word}' on rank {accelerator.process_index}")
            mask_models = gen_masks.load_mask_models(str(accelerator.device), mask_model_path)

    for step, batch in enumerate(train_dataloader):
        assert batch["images"].shape[0] == 1, "Only support, overfiting on a single video"
        print("frames_per_GPU:", batch['images'].shape[2])
        
        vae.eval()
        text_encoder.eval()
        unet.eval()
        
        if method == "ControlNet":
            # ControlNet: 生成深度图作为控制条件
            images=batch["images"]
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
            depth_images=rearrange(depth_images,"b c f h w -> (b f) c h w")
            depth_images = depth_images.to(device=accelerator.device, dtype=weight_dtype)
        elif method == "FateZero":
            # FateZero: 不需要深度图
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
        batch['ddim_init_latents'] = pipeline.prepare_latents_ddim_inverted(
            rearrange(batch["images"].to(dtype=weight_dtype), "b c f h w -> (b f) c h w"),
            batch_size=1,
            num_images_per_prompt=1,
            text_embeddings=text_embeddings,
            prompt=dataset_config.prompt,
            store_attention=use_inversion_attention,
            LOW_RESOURCE=True,
            save_path=logdir if verbose else None,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_image=depth_images,  # 传入深度图作为控制条件
            scheduler_config=scheduler_config
        )
        print(f"[TIMING] rank={accelerator.process_index} use_time_Inversion={time.time()-_t_inv0:.4f}", flush=True)
        
        vae.eval()
        text_encoder.eval()
        unet.eval()
        
        images = batch["images"].to(dtype=weight_dtype)
        images = rearrange(images, "b c f h w -> (b f) c h w")

        # ---- generate per-rank external mask from this batch's frames; attach to pipeline ----
        if use_external_mask and mask_models is not None:
            _imgs = batch["images"][0]  # [3, fpf, 512, 512] in [-1,1]
            _frames_np = ((_imgs / 2 + 0.5).clamp(0, 1).permute(1, 2, 3, 0) * 255).byte().cpu().numpy()  # [fpf,512,512,3] uint8
            _gd_proc, _gd_model, _fsam = mask_models
            _mask512 = gen_masks.generate_word_mask(
                _frames_np, mask_source_word, _gd_proc, _gd_model, _fsam, str(accelerator.device),
                box_threshold=mask_box_threshold, dilate_px=mask_dilate_px,
                verbose=accelerator.is_main_process)
            pipeline.fixed_mask = torch.from_numpy(_mask512.astype(np.float32)).to(accelerator.device)
            print(f"[mask] rank={accelerator.process_index} fixed_mask shape={tuple(pipeline.fixed_mask.shape)} "
                  f"fill={pipeline.fixed_mask.mean().item():.3f}")
            # ---- save the mask used for this edit (raw npz + per-frame overlay PNGs + contact sheet) ----
            try:
                _mask_dir = os.path.join(logdir, "mask")
                os.makedirs(_mask_dir, exist_ok=True)
                _r = accelerator.process_index
                np.savez_compressed(os.path.join(_mask_dir, f"{mask_source_word}_rank{_r}.npz"),
                                    **{mask_source_word: _mask512})
                _sheet = Image.new("RGB", (512 * _mask512.shape[0], 512), (255, 255, 255))
                _tint = np.array([255, 0, 0], np.uint8)
                for _fi in range(_mask512.shape[0]):
                    _arr = _frames_np[_fi].copy()
                    _arr[_mask512[_fi]] = (0.5 * _arr[_mask512[_fi]] + 0.5 * _tint).astype(np.uint8)
                    Image.fromarray(_arr).save(os.path.join(_mask_dir, f"{mask_source_word}_rank{_r}_frame{_fi}.png"))
                    _sheet.paste(Image.fromarray(_arr), (_fi * 512, 0))
                _sheet.save(os.path.join(_mask_dir, f"{mask_source_word}_rank{_r}_sheet.png"))
                print(f"[mask] saved mask artifacts (npz + per-frame PNGs + sheet) -> {_mask_dir}")
            except Exception as _e:
                print(f"[mask] save mask failed: {_e}")
        else:
            pipeline.fixed_mask = None

        if accelerator is not None:
            _t_den0 = time.time()
            validation_sample_logger.log_sample_images(
                image=images,
                pipeline=pipeline,
                device=accelerator.device,
                step=0,
                latents=batch['ddim_init_latents'],
                save_dir=logdir if verbose else None,
                control_image=depth_images,  # 传入控制图像
            )
            print(f"[TIMING] rank={accelerator.process_index} use_time_Denoising={time.time()-_t_den0:.4f}", flush=True)

    if mask_models is not None:
        del mask_models
        torch.cuda.empty_cache()

    if torch.cuda.device_count() > 1:
        dist.barrier()
        rank = dist.get_rank()
        if rank == 0:
            gif1 = Image.open(f"{logdir}/sample/step_0_GPU0.gif")
            gif2 = Image.open(f"{logdir}/sample/step_0_GPU1.gif")
            
            frames1 = []
            frames2 = []
            try:
                while True:
                    frames1.append(gif1.copy())
                    gif1.seek(len(frames1))
            except EOFError:
                pass
            try:
                while True:
                    frames2.append(gif2.copy())
                    gif2.seek(len(frames2))
            except EOFError:
                pass

            combined_frames = frames1 + frames2
            output_path = f"{logdir}/sample/step1.gif"
            combined_frames[0].save(
                output_path,
                save_all=True,
                append_images=combined_frames[1:],
                loop=0,
                duration=gif1.info['duration']
            )
    
    accelerator.end_training()

@click.command()
@click.option("--config", type=str, default="config/sample.yml")
def run(config):
    # 日志重定向到文件时按行刷新，保证 LLM 路由/tune 状态实时可见
    # （否则 stdio 全缓冲会把路由打印憋到进程退出才落盘）
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    # 首先读取配置获取 method
    Omegadict = OmegaConf.load(config)
    method = Omegadict.get('method')

    if not method:
        raise ValueError("Method parameter is required in config")

    # ============================================================
    # 启动部分：pipeline 启动前先跑 LLM 语义分析（论文 Sec 3.3 决策模块）。
    # 流式输出 + model_adaption 前置 -> 不等 Thinking 生成完即可路由：
    #   zero-shot                 直接用预训练模型
    #   one-shot(temporal module) 自主启动 tune，加载微调后的模型文件
    #   one-shot(Motion module)   启动外置 MotionEditor 后直接返回
    # ============================================================
    llm_overrides = {}
    if Omegadict.get('Use_DeepSeek', False):
        llm_overrides = llm_decision_routing(Omegadict, config)
        if llm_overrides is None:  # 已启动 MotionEditor，不进入 ELVE pipeline
            return
        if 'method' in llm_overrides:
            method = llm_overrides['method']  # temporal 路由固定 FateZero 模式

    # 使用上下文管理器运行测试
    with diffusers_context(method):
        # 重新加载配置以确保一致性
        Omegadict = OmegaConf.load(config)
        if llm_overrides:
            # 应用 LLM 决策出的模型配置（覆盖重载回来的 yaml 值）
            Omegadict.merge_with(OmegaConf.create(llm_overrides))

        if method == "MotionEditor":
            # MotionEditor 走独立 pipeline，与 ELVE 的 fatezero/controlnet 不交叉
            run_motioneditor(Omegadict)
            return

        if 'unet' in os.listdir(Omegadict['pretrained_model_path']):
            test(config=config, **Omegadict)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", module="torchvision.io.image")
    run()