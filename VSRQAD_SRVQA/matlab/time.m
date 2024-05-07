% 指定视频文件所在的文件夹路径
parentFolder = 'K:\VSR\sr';

% 定义要读取的文件夹名称
subfolders = {'x2','x4','x8'};

% 循环处理每个子文件夹
for k = 1:numel(subfolders)
    subfolder = subfolders{k};
    folder = fullfile(parentFolder, '10', subfolder);
    
    % 获取文件夹中所有的视频文件
    fileList = dir(fullfile(folder, '*.mp4')); % 根据需要修改视频文件的扩展名
    
    % 循环处理每个视频文件
    for i = 1:numel(fileList)
        % 构造视频文件的完整路径
        filePath = fullfile(folder, fileList(i).name);
        
        % 创建VideoReader对象读取视频文件
        videoObj = VideoReader(filePath);
        
        % 获取视频时长（以秒为单位）
        videoDuration = videoObj.Duration;
        
        % 检查视频时长是否超过6秒
        if videoDuration < 6
            fprintf('视频文件名：%s，时长：%.2f秒\n', fileList(i).name, videoDuration);
        end
    end
end
