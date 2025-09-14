from controlnet_aux import CannyDetector
from PIL import Image
import numpy as np

# 初始化检测器
canny_detector = CannyDetector()

# 加载图像
image = Image.open("data/wolf/1.jpg").convert("RGB")

# 处理图像（自动进行像素完美处理）
canny_image = canny_detector(
    image, 
    detect_resolution=512,  # 检测分辨率
    image_resolution=512,   # 输出分辨率
    low_threshold=100,      # Canny低阈值
    high_threshold=200      # Canny高阈值
)

# 保存结果
canny_image.save("canny_output.jpg")