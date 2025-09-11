from controlnet_aux import ZoeDetector
from PIL import Image

# 使用 from_pretrained 方法
zoe_detector = ZoeDetector.from_pretrained("pretrained_models/zoedepth")

# 加载图像
image = Image.open("data/wolf/1.jpg").convert("RGB")

# 处理图像
zoe_depth_map = zoe_detector(
    image,
    detect_resolution=512,
    image_resolution=512,
)

# 保存结果
zoe_depth_map.save("zoe_depth_output.jpg")