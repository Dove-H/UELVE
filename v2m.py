import cv2  
import os  
  
def extract_frames(video_path, output_folder, num_frames=27, resolution=(512, 512)):  #改12帧3网格
    # 检查输出文件夹是否存在，不存在则创建  
    if not os.path.exists(output_folder):  
        os.makedirs(output_folder)  
  
    # 打开视频文件  
    cap = cv2.VideoCapture(video_path)  
  
    # 获取视频的总帧数  
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
  
    # 计算间隔  
    interval = total_frames // num_frames  
  
    # 提取帧  
    frame_indices = [i * interval for i in range(num_frames)]  
  
    count = 1  
    for idx in frame_indices:  
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  
        ret, frame = cap.read()  
  
        if ret:  
            # 调整帧的分辨率  
            resized_frame = cv2.resize(frame, resolution)  
  
            # 将帧保存为JPG文件  
            output_path = os.path.join(output_folder, f"{count}.jpg")  
            cv2.imwrite(output_path, resized_frame)  
            print(f"保存帧 {count} 到 {output_path}")  
            count += 1  
        else:  
            print(f"无法读取第 {idx} 帧")  
  
    # 释放视频对象  
    cap.release()  
  
# 调用  
video_path = 'data/deer.mp4'  # 输入视频路径  
output_folder = 'data/deer27f'  # 输出文件夹  
extract_frames(video_path, output_folder)