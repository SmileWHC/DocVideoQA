# import os
# import sys

# video_path = "/apdcephfs_cq10/share_2992827/wrwwang/dataset/video"
# mp3_path = "/apdcephfs_cq10/share_2992827/wrwwang/dataset/audio"
# for root, dirs, files in os.walk(video_path):
#     for file in files:
#         if file.endswith(".mp4"):
#             video_file = os.path.join(root, file)
#             mp3_file = os.path.join(mp3_path, file.replace(".mp4", ".mp3"))
#             os.system(f"ffmpeg -i {video_file} -vn -c:a libmp3lame -y {mp3_file}")
#             sys.stdout.flush()
# # ffmpeg -i input.mp4 -vn -acodec copy output.mp3

import os
import subprocess

video_path = "/apdcephfs_cq10/share_2992827/wrwwang/dataset/video"
wav_path = "/apdcephfs_cq10/share_2992827/wrwwang/dataset/audio_wav"  # 注意这里更改为.wav的输出目录

# 确保WAV存储路径存在
if not os.path.exists(wav_path):
    os.makedirs(wav_path)

for root, dirs, files in os.walk(video_path):
    for file in files:
        if file.endswith(".mp4"):
            video_file = os.path.join(root, file)
            wav_file = os.path.join(wav_path, file.replace(".mp4", ".wav"))
            # 使用subprocess调用ffmpeg命令进行转换
            command = ["ffmpeg", "-i", video_file, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", "-y", wav_file]
            try:
                subprocess.run(command, check=True)
                print(f"转换成功：{video_file} -> {wav_file}")
            except subprocess.CalledProcessError:
                print(f"转换失败：{video_file}")