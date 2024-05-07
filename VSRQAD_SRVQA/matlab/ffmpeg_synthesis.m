clear all;
% 定义要转换的文件夹路径和输出视频的文件名
folder_path = 'J:\x';
output_filename = 'G:\';

% 获取文件夹中的所有子文件夹
subfolders = dir(folder_path);
subfolders = subfolders([subfolders.isdir]);
subfolders = subfolders(3:end);

% 遍历子文件夹，将其中的图片转换为视频
for i = 1:numel(subfolders)
    subfolder = fullfile(folder_path, subfolders(i).name);
    [Filepath,Name,Ext] = fileparts(subfolders(i).name);
    num=Name(6:end);
    image_files = dir(fullfile(subfolder, '*.bmp')); % 读取文件夹中的所有图片
    [filepath,name,ext] = fileparts(image_files(1).name);
    num2_fps=name(10:11);
    frame_rate=str2double(num2_fps); % 将字符串转换为数字
    num1_he=name(16:17);
    num3_sr=name(22:23);
    num4_x=name(19:20);
    out_file=sprintf('video%s_%sfps_%s_%s_%s.mp4',num,num2_fps,num1_he,num4_x,num3_sr);
    output_file = fullfile(output_filename, out_file); % 输出视频的文件名
  
    % 使用FFmpeg命令将图片转换为视频
%     ffmpeg_cmd = sprintf('ffmpeg -y -r %d -f image2 -pattern_type glob -i %s -c:v ffv1 -pix_fmt yuv420p %s', frame_rate, fullfile(subfolder, '*.bmp'), output_file);
%     system(ffmpeg_cmd);
    % 获取文件夹中的所有bmp文件
    image_files = dir(fullfile(subfolder, '*.bmp'));
    image_files = {image_files.name};
    
    % 将文件名保存到一个文本文件中
    fid = fopen('file_list.txt', 'wt');
    for j = 1:length(image_files)
        fprintf(fid, 'file ''%s''\n', fullfile(subfolder, image_files{j}));
    end
    fclose(fid);

    % 使用文本文件作为输入参数传递给ffmpeg
    ffmpeg_cmd = sprintf('ffmpeg -r %d -f concat -safe 0 -i file_list.txt -c:v libx265  -pix_fmt yuv420p -preset slow -x265-params lossless=1 "%s"', frame_rate, output_file);
    system(ffmpeg_cmd);

end

