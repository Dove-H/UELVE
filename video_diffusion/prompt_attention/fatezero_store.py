
import json
import os
import threading
import time
from queue import Queue

import torch

from .. import elve_timing
from ..elve_timing import timer, add, add_bytes


def _payload_bytes(payload):
    if isinstance(payload, dict):
        return sum(t.numel() * t.element_size() for v in payload.values() for t in v)
    if torch.is_tensor(payload):
        return payload.numel() * payload.element_size()
    return 0

ATTN_MAX_TOKENS = 32 ** 2


class AttentionControl:

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
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
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

    MANIFEST = 'manifest.json'

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def __init__(self, store_dir: str = None, need: dict = None, keep_on_gpu: bool = True):
        super().__init__()
        self.LOW_RESOURCE = True
        self.store_dir = store_dir
        self.need = need or {}
        self.keep_on_gpu = keep_on_gpu
        self.shards = {} if keep_on_gpu else None      # (step, grid) -> store dict
        self.latents = {} if keep_on_gpu else None     # step -> tensor
        self.init_latent = None
        self.step_store = self.get_empty_store()
        self._mark = {}
        self._layers_per_call = {}
        self._cur_step = 0
        self._write_queue = None
        self._write_error = []
        if not keep_on_gpu:
            self._write_queue = Queue(maxsize=4)
            self._write_thread = threading.Thread(target=self._write_worker, daemon=True)
            self._write_thread.start()


    def begin_step(self, step: int):
        self._cur_step = step

    def begin_grid(self):
        self._mark = {k: len(v) for k, v in self.step_store.items()}

    def end_grid(self, grid: int):
        step = self._cur_step
        """Slice the attention maps added by this UNet call into a shard and store it."""
        shard = {k: v[self._mark[k]:] for k, v in self.step_store.items()}
        for k, v in self.step_store.items():
            del v[self._mark[k]:]
        self._mark = {}

        for k, v in shard.items():
            if len(v) == 0:
                continue
            known = self._layers_per_call.get(k, 0)
            if known == 0:
                self._layers_per_call[k] = len(v)
            else:
                assert len(v) == known, (
                    f"[InversionStore] {k} layer count unstable ({len(v)} != {known}), "
                    "shard alignment may be broken")

        if all(len(v) == 0 for v in shard.values()):
            return
        if self.keep_on_gpu:
            self.shards[(step, grid)] = shard
        else:
            self._save_shard(step, grid, shard)

    def step_callback(self, x_t, step: int):
        if step in self.need.get('latents', set()):
            if self.keep_on_gpu:
                self.latents[step] = x_t.detach()
            else:
                os.makedirs(self.store_dir, exist_ok=True)
                path = os.path.join(self.store_dir, f'lat{step:03d}.pt')
                with timer('store_d2h'):
                    lat = x_t.detach().cpu()
                with timer('store_write_latent'):
                    torch.save(lat, path)
                add_bytes('store_write_latent', os.path.getsize(path))
        return x_t

    def finish(self, init_latent):
        self.init_latent = init_latent.detach()
        if not self.keep_on_gpu:
            self._write_queue.put(None)
            self._write_thread.join()
            if self._write_error:
                raise RuntimeError(f'async inversion-shard offload failed: {self._write_error[0]}')
            os.makedirs(self.store_dir, exist_ok=True)
            with timer('store_write_init'):
                torch.save(self.init_latent.cpu(), os.path.join(self.store_dir, 'init.pt'))
            self._save_manifest()


    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if attn.shape[-2] <= ATTN_MAX_TOKENS:
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            need = self.need.get('cross' if is_cross else 'self')
            if need is None or self._cur_step in need:
                self.step_store[key].append(attn.detach())
        return attn


    def _save_shard(self, step, grid, shard):
        path = os.path.join(self.store_dir, f'step{step:03d}_grid{grid:02d}.pt')
        with timer('store_d2h'):
            payload = {k: [t.cpu() for t in v] for k, v in shard.items()}
        with timer('store_write_backpressure'):
            self._write_queue.put((path, payload))

    def _write_worker(self):
        os.makedirs(self.store_dir, exist_ok=True)
        while True:
            item = self._write_queue.get()
            if item is None:
                return
            if self._write_error:
                continue
            path, payload = item
            try:
                with timer('store_write_shard'):
                    torch.save(payload, path)
                add_bytes('store_write_shard', os.path.getsize(path))
            except Exception as e:
                self._write_error.append(e)

    def _save_manifest(self):
        need = {k: (sorted(v) if v is not None else None) for k, v in self.need.items()}
        manifest = dict(self.meta) if getattr(self, 'meta', None) else {}
        manifest['need'] = need
        with open(os.path.join(self.store_dir, self.MANIFEST), 'w') as f:
            json.dump(manifest, f)

    def set_meta(self, meta: dict):
        self.meta = meta


    @staticmethod
    def cache_valid(store_dir: str, meta: dict, need: dict) -> bool:
        try:
            with open(os.path.join(store_dir, InversionStore.MANIFEST)) as f:
                manifest = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return False
        for k, v in meta.items():
            if isinstance(v, tuple):
                v = list(v)
            if manifest.get(k) != v:
                return False
        for kind, new_set in need.items():
            old = manifest.get('need', {}).get(kind)
            if old is None:
                continue
            if new_set is None or not set(new_set) <= set(old):
                return False
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
                with timer('store_read_shard'):
                    payload = self.load_shard(item[1], item[2])
                add_bytes('store_read_shard', _payload_bytes(payload))
            else:
                with timer('store_read_latent'):
                    payload = self.load_latent(item[1])
                add_bytes('store_read_latent', _payload_bytes(payload))
            self._queue.put((item, payload))
        self._queue.put(None)

    def _next(self):
        _t0 = time.perf_counter() if elve_timing.ENABLED else None
        item = self._queue.get()
        if item is None:
            raise RuntimeError('[ReplayProvider] prefetch sequence exhausted; consumed more than expected')
        if _t0 is not None:
            add('store_wait_main', time.perf_counter() - _t0)
        expected = next(self._expected, None)
        assert item[0] == expected, f"[ReplayProvider] out-of-order consumption: got {item[0]}, expect {expected}"
        return item

    def get_shard(self, step: int, grid: int):
        item, payload = self._next()
        assert item == ('shard', step, grid), f"[ReplayProvider] shard mismatch: got {item}, want {(step, grid)}"
        if isinstance(payload, dict):
            with timer('store_h2d'):
                payload = {k: [t.to(self.device, self.dtype) for t in v] for k, v in payload.items()}
        return payload

    def get_latent(self, step: int):
        item, payload = self._next()
        assert item == ('latent', step), f"[ReplayProvider] latent mismatch: got {item}, want {('latent', step)}"
        if torch.is_tensor(payload):
            with timer('store_h2d'):
                payload = payload.to(self.device, self.dtype)
        return payload

    def has_shard(self, step: int) -> bool:
        cross_need, self_need = self.need.get('cross'), self.need.get('self')
        return (cross_need is None or step in cross_need) or (self_need is None or step in self_need)

    def has_latent(self, step: int) -> bool:
        return step in self.need.get('latents', set())
