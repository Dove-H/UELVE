from PIL import Image
import os
import imageio

def merge_gifs(gif1_path, gif2_path, output_path, duration=None):
    """
    合并两个GIF图片
    
    Args:
        gif1_path: 第一个GIF路径
        gif2_path: 第二个GIF路径
        output_path: 输出GIF路径
        duration: 每帧持续时间（毫秒），None表示使用原GIF的时长
    """
    try:
        # 读取两个GIF
        gif1 = Image.open(gif1_path)
        gif2 = Image.open(gif2_path)
        
        frames = []
        
        # 提取第一个GIF的所有帧
        try:
            while True:
                frame = gif1.copy()
                frames.append(frame)
                gif1.seek(gif1.tell() + 1)
        except EOFError:
            pass
        
        # 提取第二个GIF的所有帧
        try:
            while True:
                frame = gif2.copy()
                frames.append(frame)
                gif2.seek(gif2.tell() + 1)
        except EOFError:
            pass
        
        # 保存合并后的GIF
        if duration is None:
            # 尝试获取第一个GIF的duration
            gif1_duration = gif1.info.get('duration', 100)
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=gif1_duration,
                loop=0
            )
        else:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0
            )
        
        print(f"GIF合并完成: {output_path}")
        print(f"总帧数: {len(frames)}")
        
    except Exception as e:
        print(f"GIF合并失败: {e}")

# 使用imageio的替代方案
def merge_gifs_imageio(gif1_path, gif2_path, output_path, fps=10):
    """
    使用imageio合并GIF（另一种方法）
    
    Args:
        gif1_path: 第一个GIF路径
        gif2_path: 第二个GIF路径
        output_path: 输出GIF路径
        fps: 帧率
    """
    try:
        # 读取GIF帧
        gif1_frames = imageio.mimread(gif1_path)
        gif2_frames = imageio.mimread(gif2_path)
        
        # 合并帧
        merged_frames = gif1_frames + gif2_frames
        
        # 保存合并后的GIF
        imageio.mimsave(output_path, merged_frames, fps=fps)
        
        print(f"GIF合并完成: {output_path}")
        print(f"总帧数: {len(merged_frames)}")
        
    except Exception as e:
        print(f"GIF合并失败: {e}")

# 使用示例
if __name__ == "__main__":
    gif1 = "result/edit/car-tree-tractor_251001-1033/sample/step_0_0_0_GPU0.gif"
    gif2 = "result/edit/car-tree-tractor_251001-1033/sample/step_0_0_0_GPU1.gif"
    output = "result/edit/car-tree-tractor_251001-1033/sample/step2.gif"
    
    # 方法1：使用PIL
    merge_gifs(gif1, gif2, output)
    
    
    # 方法2：使用imageio
    # merge_gifs_imageio(gif1, gif2, output, fps=10)