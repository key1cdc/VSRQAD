% 使用FFmpeg裁剪视频
system('ffmpeg -i K:\video_jitter\video007\x8\video007_29fps_04_x8_01.mp4 -filter:v "crop=300:200:400:300" K:\video_jitter\video007\x8\video007_x8_output.mp4')
%% 宽 高 左上顶点

% system('ffmpeg -i K:\video_jitter\video008\x2\video008_29fps_09_x2_03.mp4 -filter:v "crop=800:500:1200:5" K:\video_jitter\video008\x2\video008_x2_output.mp4')

% system('ffmpeg -i K:\video_jitter\video222\x2\video222_29fps_05_x2_03.mp4 -filter:v "crop=200:200:500:600" K:\video_jitter\video222\x2\video222_x2_output.mp4')
