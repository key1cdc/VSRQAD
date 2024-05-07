% clear all
% % 指定父文件夹路径
% parentFolder = 'F:\AIVQA\Val\ValFrame';
% 
% % 获取父文件夹中的子文件夹列表
% subfolders = dir(parentFolder);
% subfolders = subfolders([subfolders.isdir]); % 仅保留子文件夹
% 
% % 创建保存调整大小后图像的文件夹
% outputFolder = 'F:\AIVQA\Val\ValFrame_384×224';
% if ~exist(outputFolder, 'dir')
%     mkdir(outputFolder);
% end
% 
% % 遍历每个子文件夹
% for i = 1:numel(subfolders)
%     if strcmp(subfolders(i).name, '.') || strcmp(subfolders(i).name, '..')
%         continue; % 跳过当前文件夹和上一级文件夹
%     end
%     
%     % 获取子文件夹路径
%     subfolderPath = fullfile(parentFolder, subfolders(i).name);
%     
%     % 读取子文件夹中的所有图片
%     imageFiles = dir(fullfile(subfolderPath, '*.png')); % 根据实际图片格式更改后缀
%     
%     % 遍历子文件夹中的每张图片进行调整大小
%     for j = 1:numel(imageFiles)
%         % 读取当前图片
%         imagePath = fullfile(subfolderPath, imageFiles(j).name);
%         img = imread(imagePath);
%         
%         % 调整图片大小为384x224
%         resizedImg = imresize(img, [224 384]);
%         
%         % 构造保存路径和文件名
%         [~, name, ext] = fileparts(imageFiles(j).name);
%         outputName = sprintf('%s%s', name, ext);
%         outputPath = fullfile(outputFolder, subfolders(i).name, outputName);
%         
%         % 创建子文件夹
%         suboutputFolder = fullfile(outputFolder, subfolders(i).name);
%         if ~exist(suboutputFolder, 'dir')
%             mkdir(suboutputFolder);
%         end
%         
%         % 保存调整大小后的图像
%         imwrite(resizedImg, outputPath);
%     end
% end


clear all
% 指定父文件夹路径
parentFolder = 'G:\VSR_QAD_Frame\lr\x8';

% 获取父文件夹中的子文件夹列表
subfolders = dir(parentFolder);
subfolders = subfolders([subfolders.isdir]); % 仅保留子文件夹

% 创建保存调整大小后图像的文件夹
outputFolder = 'G:\VSR_QAD_Frame\lr_up\x8';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% 遍历每个子文件夹
for i = 1:numel(subfolders)
    if strcmp(subfolders(i).name, '.') || strcmp(subfolders(i).name, '..')
        continue; % 跳过当前文件夹和上一级文件夹
    end
    
    % 获取子文件夹路径
    subfolderPath = fullfile(parentFolder, subfolders(i).name);
    
    % 读取子文件夹中的所有图片
    imageFiles = dir(fullfile(subfolderPath, '*.png')); % 根据实际图片格式更改后缀
    
    % 遍历子文件夹中的每张图片进行调整大小
    for j = 1:numel(imageFiles)
        % 读取当前图片
        imagePath = fullfile(subfolderPath, imageFiles(j).name);
        img = imread(imagePath);
        
        % 调整图片大小为384x224
        resizedImg = imresize(img, [1080 1920]);
        
        % 构造保存路径和文件名
        [~, name, ext] = fileparts(imageFiles(j).name);
        outputName = sprintf('%s%s', name, ext);
        outputPath = fullfile(outputFolder, subfolders(i).name, outputName);
        
        % 创建子文件夹
        suboutputFolder = fullfile(outputFolder, subfolders(i).name);
        if ~exist(suboutputFolder, 'dir')
            mkdir(suboutputFolder);
        end
        
        % 保存调整大小后的图像
        imwrite(resizedImg, outputPath);
    end
end