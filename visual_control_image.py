import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

def visualize_control_tensor(control_pt_path, output_dir=None, save_images=True, show_images=True):
    """
    将control.pt文件可视化为图片
    
    参数:
    - control_pt_path: control.pt文件的路径
    - output_dir: 保存图片的目录，如果为None则不保存
    - save_images: 是否保存图片
    - show_images: 是否显示图片
    """
    # 加载control.pt文件
    control_tensor = torch.load(control_pt_path)
    
    print(f"control_tensor形状: {control_tensor.shape}")
    print(f"control_tensor数据类型: {control_tensor.dtype}")
    print(f"control_tensor值范围: [{control_tensor.min():.3f}, {control_tensor.max():.3f}]")
    
    # 确保输出目录存在
    if save_images and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # 将张量转换为numpy数组并处理
    if control_tensor.dim() == 4:  # [batch_size, channels, height, width]
        batch_size = control_tensor.shape[0]
        
        for i in range(batch_size):
            # 获取单个控制图像
            img_tensor = control_tensor[i]
            
            # 如果通道数大于1，取第一个通道（假设是灰度图）
            if img_tensor.shape[0] > 1:
                img_tensor = img_tensor[0:1]  # 取第一个通道
            
            # 转换为numpy数组
            img_array = img_tensor.squeeze().cpu().numpy()
            
            # 归一化到[0, 255]
            if img_array.max() > 1.0 or img_array.min() < 0.0:
                # 如果是浮点数且范围不在[0,1]，则归一化
                img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
            else:
                img_array = img_array * 255
            
            img_array = img_array.astype(np.uint8)
            
            # 创建PIL图像
            pil_image = Image.fromarray(img_array)
            
            # 显示图像
            if show_images:
                plt.figure(figsize=(8, 8))
                plt.imshow(pil_image, cmap='gray')
                plt.title(f'Control Image {i+1}/{batch_size}')
                plt.axis('off')
                plt.show()
            
            # 保存图像
            if save_images and output_dir is not None:
                output_path = os.path.join(output_dir, f'control_image_{i}.png')
                pil_image.save(output_path)
                print(f"保存控制图像到: {output_path}")
    
    return control_tensor

# 使用方法
if __name__ == "__main__":
    control_pt_path = "/mnt/fspace/hproject/ELVE/Control/1GPU/canny/wolf/8/control.pt"
    output_dir = "control_image/canny"
    
    # 可视化control.pt
    control_tensor = visualize_control_tensor(
        control_pt_path, 
        output_dir=output_dir,
        save_images=True,
        show_images=True
    )