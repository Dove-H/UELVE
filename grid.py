import torch
#frame 2 grid
def combine_frames_in_grid(video_tensor, grid_sizeH,grid_sizeW):
    batch_size, num_channels, video_length, height, width = video_tensor.shape
    assert video_length % (grid_sizeH * grid_sizeW) == 0, "video_length 必须是 g * g 的倍数"

    grid_h = height * grid_sizeH
    grid_w = width * grid_sizeW
    new_video_length = video_length // (grid_sizeH * grid_sizeW)
            
    grid_tensor = torch.zeros((batch_size, num_channels, new_video_length, grid_h, grid_w), dtype=video_tensor.dtype, device=video_tensor.device)

    for b in range(batch_size):
        img_idx = 0
        for n in range(new_video_length):
            for i in range(grid_sizeH):
                for j in range(grid_sizeW):
                    # 复制对应的帧到 grid_tensor 中
                    grid_tensor[b, :, n, i * height:(i + 1) * height, j * width:(j + 1) * width] = video_tensor[b, :, img_idx]
                    img_idx += 1

    return grid_tensor



#grid 2 frame  
def restore_frames_from_grid(grid_tensor, grid_sizeH,grid_sizeW):
    batch_size, num_channels, new_video_length, grid_height, grid_width = grid_tensor.shape
    video_length = new_video_length * (grid_sizeH * grid_sizeW)
    height = grid_height // grid_sizeH
    width = grid_width // grid_sizeW

    # 初始化一个全零的张量用于存放还原后的帧
        
    restored_tensor = torch.zeros((batch_size, num_channels, video_length, height, width), device=grid_tensor.device, dtype=grid_tensor.dtype)
    # 遍历每个批次
    for b in range(batch_size):
        img_idx = 0  # 初始化图像索引
        for n in range(new_video_length):
            for i in range(grid_sizeH):
                for j in range(grid_sizeW):
                    restored_tensor[b, :, img_idx] = grid_tensor[b, :, n, i * height:(i + 1) * height, j * width:(j + 1) * width]
                    img_idx += 1  # 更新帧索引
                    

    return restored_tensor



