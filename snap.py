import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

# 使用仓库ID，并通过模式匹配只下载特定文件
snapshot_download(
    repo_id="lllyasviel/Annotators",  # 仓库ID
    repo_type="model",
    local_dir="/mnt/fspace/hproject/ELVE/pretrained_models/leres",
    allow_patterns=["*res101.pth"],  # 只下载这个文件
    force_download=False
)