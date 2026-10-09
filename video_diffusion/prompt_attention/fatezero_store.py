"""
FateZero 模式（重构版）注意力 / latent 存储与重放数据源。

以原始 FateZero 的 AttentionStore 语义为基础（is_cross 为 bool，键为 {place}_cross / {place}_self），
针对网格机制与显存分级重新设计：

1. 分片粒度为 (inversion_step, grid)：反演期 UNet 逐网格调用，每次调用产生的
   ≤1024 token 的 attention map 归入一个独立分片；编辑期按同样的 (step, grid) 顺序消费，
   天然支持"算当前网格时预取下一个网格"的异步双缓冲。
2. 显存分级：单卡帧数 ≤ 阈值（memory_frames）时特征全部驻留 GPU；超过则即时落盘，
   编辑期由后台线程按消费顺序预取（I/O 与计算 overlap）。
3. 数据裁剪（修正原版 FateZero 存储的浪费）：
   - cross 只存编辑期会消费的步（cross_replace_alpha>0 ∪ self 重放窗口 ∪ latent blend 窗口）
   - self 只存 self_replace 窗口配对的反演步（编辑步 i ↔ 反演步 N-1-i）
   - latent 只存 latent blend 窗口配对的反演步；不用 blend 时完全不存
   - 不保留原版仅用于可视化的 attention_store 累积平均副本；去掉多余 deepcopy
   - temporal attention 完全不存（对齐原版：register 阶段即跳过 attn_temporal，
     且其 to_out 零初始化，zero-shot 下本就是恒等映射）
"""

import json
import os
import threading
from queue import Queue

import torch

ATTN_MAX_TOKENS = 32 ** 2  # 与原版一致：只记录 ≤1024 token 的层（更大层走 xformers，两侧对称跳过）


class AttentionControl:
    """controller 基类：cur_step / cur_att_layer / LOW_RESOURCE 记账（原版语义）。"""

    def step_callback(self, x_t):
        self.cur_att_layer = 0
        self.cur_step += 1
        self.between_steps()
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if self.LOW_RESOURCE:
                # 反演阶段无 CFG，全 batch 记录
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                # 编辑阶段有 CFG，只处理条件分支（后半 batch）
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.LOW_RESOURCE = False
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class InversionStore(AttentionControl):
    """反演期存储：按 (step, grid) 分片，按 need 集合裁剪。

    need: {'cross': set[int] | None, 'self': set[int] | None, 'latents': set[int]}
    其中集合元素是反演步索引；None 表示该类全部步都存。
    """

    MANIFEST = 'manifest.json'

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def __init__(self, store_dir: str = None, need: dict = None, keep_on_gpu: bool = True):
        super().__init__()
        self.LOW_RESOURCE = True  # 反演无 CFG
        self.store_dir = store_dir
        self.need = need or {}
        self.keep_on_gpu = keep_on_gpu
        self.shards = {} if keep_on_gpu else None      # (step, grid) -> store dict
        self.latents = {} if keep_on_gpu else None     # step -> tensor
        self.init_latent = None                        # 反演最终噪声 latent（编辑起点）
        self.step_store = self.get_empty_store()
        self._mark = {}
        self._layers_per_call = {}                     # key -> 每次 UNet 调用的层数（一致性校验）
        self._cur_step = 0

    # ---------------- 反演循环驱动接口（由 pipeline 调用） ----------------

    def begin_step(self, step: int):
        self._cur_step = step

    def begin_grid(self):
        """记录当前 step_store 各 key 的长度，准备截取本网格新增条目。"""
        self._mark = {k: len(v) for k, v in self.step_store.items()}

    def end_grid(self, grid: int):
        """把本次 UNet 调用新增的 attention map 截成一个分片并存储（步号取 begin_step 设定值）。"""
        step = self._cur_step
        """把本次 UNet 调用新增的 attention map 截成一个分片并存储。"""
        shard = {k: v[self._mark[k]:] for k, v in self.step_store.items()}
        for k, v in self.step_store.items():
            del v[self._mark[k]:]
        self._mark = {}

        # 层数一致性校验：need 之外的步该键为空（0 条），跳过；仅对非空分片学习/比对
        for k, v in shard.items():
            if len(v) == 0:
                continue
            known = self._layers_per_call.get(k, 0)
            if known == 0:
                self._layers_per_call[k] = len(v)
            else:
                assert len(v) == known, (
                    f"[InversionStore] {k} 层数不稳定（{len(v)} != {known}），"
                    "分片对齐可能被破坏")

        if all(len(v) == 0 for v in shard.values()):
            return  # 无任何需求的（步, 网格）不落盘（ReplayProvider 也不会读它）
        if self.keep_on_gpu:
            self.shards[(step, grid)] = shard
        else:
            self._save_shard(step, grid, shard)

    def step_callback(self, x_t, step: int):
        """保存本反演步（更新后）的 latent —— 仅 blend 窗口内的步。"""
        if step in self.need.get('latents', set()):
            if self.keep_on_gpu:
                self.latents[step] = x_t.detach()
            else:
                os.makedirs(self.store_dir, exist_ok=True)
                torch.save(x_t.detach().cpu(), os.path.join(self.store_dir, f'lat{step:03d}.pt'))
        return x_t

    def finish(self, init_latent):
        """反演结束：记录最终 latent 并写 manifest（缓存复用凭据）。"""
        self.init_latent = init_latent.detach()
        if not self.keep_on_gpu:
            os.makedirs(self.store_dir, exist_ok=True)
            torch.save(self.init_latent.cpu(), os.path.join(self.store_dir, 'init.pt'))
            self._save_manifest()

    # ---------------- controller 接口：反演期逐层记录 ----------------

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if attn.shape[-2] <= ATTN_MAX_TOKENS:
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            need = self.need.get('cross' if is_cross else 'self')
            if need is None or self._cur_step in need:
                self.step_store[key].append(attn.detach())
        return attn

    # ---------------- 磁盘布局：分片 step{S}_grid{J}.pt / latent lat{S}.pt / init.pt / manifest.json ----------------

    def _save_shard(self, step, grid, shard):
        os.makedirs(self.store_dir, exist_ok=True)
        torch.save({k: [t.cpu() for t in v] for k, v in shard.items()},
                   os.path.join(self.store_dir, f'step{step:03d}_grid{grid:02d}.pt'))

    def _save_manifest(self):
        need = {k: (sorted(v) if v is not None else None) for k, v in self.need.items()}
        manifest = dict(self.meta) if getattr(self, 'meta', None) else {}
        manifest['need'] = need
        with open(os.path.join(self.store_dir, self.MANIFEST), 'w') as f:
            json.dump(manifest, f)

    def set_meta(self, meta: dict):
        self.meta = meta

    # ---------------- 缓存复用 ----------------

    @staticmethod
    def cache_valid(store_dir: str, meta: dict, need: dict) -> bool:
        """缓存有效 = manifest 匹配 + 所需分片/latent 文件齐全 + 新需求被已存需求覆盖。"""
        try:
            with open(os.path.join(store_dir, InversionStore.MANIFEST)) as f:
                manifest = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return False
        for k, v in meta.items():
            if isinstance(v, tuple):
                v = list(v)  # json 序列化把 tuple 写成 list，统一口径再比较（如 latent_replace_steps）
            if manifest.get(k) != v:
                return False
        for kind, new_set in need.items():
            old = manifest.get('need', {}).get(kind)
            if old is None:
                continue  # 旧缓存存了全部步，必然覆盖
            if new_set is None or not set(new_set) <= set(old):
                return False
        # 检查文件齐全性
        for kind, new_set in need.items():
            steps = range(0, manifest['num_steps']) if new_set is None else new_set
            for s in steps:
                if kind == 'latents':
                    if not os.path.exists(os.path.join(store_dir, f'lat{s:03d}.pt')):
                        return False
                else:
                    for g in range(manifest['num_grids']):
                        if not os.path.exists(os.path.join(store_dir, f'step{s:03d}_grid{g:02d}.pt')):
                            return False
        return os.path.exists(os.path.join(store_dir, 'init.pt'))

    @staticmethod
    def load_cached(store_dir: str):
        init = torch.load(os.path.join(store_dir, 'init.pt'), map_location='cpu', weights_only=True)
        return init


class ReplayProvider:
    """编辑期重放数据源。

    消费顺序严格镜像编辑循环：反演步 s = N-1-i 递减，步内 grid 0..G-1 递增；
    每步首个 grid 同时携带该步 latent。后台线程领先消费序列预取（队列容量 4），
    UNet 计算当前网格时磁盘 I/O 加载下一个分片，实现 overlap。
    """

    def __init__(self, num_steps: int, num_grids: int, need: dict,
                 load_shard, load_latent, device, dtype=torch.float16):
        self.num_steps, self.num_grids, self.need = num_steps, num_grids, need
        self.load_shard, self.load_latent = load_shard, load_latent
        self.device, self.dtype = device, dtype

        cross_need = self.need.get('cross')
        self_need = self.need.get('self')
        lat_need = self.need.get('latents', set())
        self.order = []  # ('shard', s, j) / ('latent', s)
        for s in range(num_steps - 1, -1, -1):
            shard_needed = ((cross_need is None or s in cross_need) or
                            (self_need is None or s in self_need))
            if shard_needed:
                for j in range(num_grids):
                    self.order.append(('shard', s, j))
                if s in lat_need:
                    self.order.append(('latent', s))

        self._queue = Queue(maxsize=4)
        self._expected = iter(self.order)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        for item in self.order:
            kind = item[0]
            if kind == 'shard':
                payload = self.load_shard(item[1], item[2])
            else:
                payload = self.load_latent(item[1])
            self._queue.put((item, payload))
        self._queue.put(None)

    def _next(self):
        item = self._queue.get()
        if item is None:
            raise RuntimeError('[ReplayProvider] 预取序列耗尽，消费超出预期')
        expected = next(self._expected, None)
        assert item[0] == expected, f"[ReplayProvider] 消费乱序: got {item[0]}, expect {expected}"
        return item

    def get_shard(self, step: int, grid: int):
        """返回已上 GPU 的 (step, grid) 分片；未命中缓存时同步加载。"""
        item, payload = self._next()
        assert item == ('shard', step, grid), f"[ReplayProvider] 分片错位: got {item}, want {(step, grid)}"
        if isinstance(payload, dict):
            payload = {k: [t.to(self.device, self.dtype) for t in v] for k, v in payload.items()}
        return payload

    def get_latent(self, step: int):
        item, payload = self._next()
        assert item == ('latent', step), f"[ReplayProvider] latent 错位: got {item}, want {('latent', step)}"
        if torch.is_tensor(payload):
            payload = payload.to(self.device, self.dtype)
        return payload

    def has_shard(self, step: int) -> bool:
        cross_need, self_need = self.need.get('cross'), self.need.get('self')
        return (cross_need is None or step in cross_need) or (self_need is None or step in self_need)

    def has_latent(self, step: int) -> bool:
        return step in self.need.get('latents', set())
