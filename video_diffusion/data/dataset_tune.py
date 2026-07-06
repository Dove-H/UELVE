import torch
from torch.utils.data import Dataset
import decord
decord.bridge.set_bridge('torch')
from einops import rearrange
import os
from PIL import Image
import numpy as np

class TuneAVideoDataset(Dataset):
    def __init__(
            self,
            video_path: str,
            prompt: str,
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 8,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
            GPU_num: int=1
    ):
        self.video_path = video_path
        self.prompt = prompt
        self.prompt_ids = None
        self.uncond_prompt_ids = None
        self.GPU_num=GPU_num
        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

        if 'mp4' not in self.video_path:
            self.images = []
            _files = [f for f in os.listdir(self.video_path) if f.endswith('jpg')]
            for file in sorted(_files, key=lambda x: int(x[:-4])):
                self.images.append(np.asarray(Image.open(os.path.join(self.video_path, file)).convert('RGB').resize((self.width, self.height))))
        self.images = np.stack(self.images)

    def __len__(self):
        return self.GPU_num  # 返回2，表示我们将数据分成两个批次

    def __getitem__(self, index):
        # load and sample video frames
        if 'mp4' in self.video_path:
            vr = decord.VideoReader(self.video_path, width=self.width, height=self.height)
            sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
            video = vr.get_batch(sample_index)
        else:
            video = self.images[:self.n_sample_frames]
        video = rearrange(video, "f h w c -> f c h w")
        GPU_frmaes=self.n_sample_frames//self.GPU_num
        # Split the 8 frames into 2 batches of 4 frames each
        
        video = video[index*GPU_frmaes:(index+1)*GPU_frmaes]  # First 4 frames
        

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": self.prompt_ids,
        }

        return example