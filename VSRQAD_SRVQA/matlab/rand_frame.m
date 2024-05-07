clear all
sourceFolder = 'G:\VSR_QAD_Frame\hr';
targetFolder = 'H:\database\VSR_613\new_data\hr';
seed = 1;
rng(seed);
% 创建目标文件夹（如果不存在）
if ~exist(targetFolder, 'dir')
    mkdir(targetFolder);
end


subFolders = dir(sourceFolder);
subFolders = subFolders([subFolders.isdir]); % 只选择文件夹

for i = 1:numel(subFolders)
    imageNames = '007'; % 需要复制的图片名称的最后三位
%     imageNames = sprintf('%03d', imageNames);
    subFolderName = subFolders(i).name;

    if strcmp(subFolderName, '.') || strcmp(subFolderName, '..')
        continue;
    end

    subFolderPath = fullfile(sourceFolder, subFolderName);
    imageFiles = dir(fullfile(subFolderPath, '*.png')); % 假设图片格式为jpg，可根据实际情况修改

    for j = 1:numel(imageFiles)
        imageFileName = imageFiles(j).name;
        [~, imageName, imageExt] = fileparts(imageFileName);
        lastThreeChars = imageName(end-2:end);

%         if ismember(lastThreeChars, imageNames) 包含
        if lastThreeChars == imageNames
            sourceFile = fullfile(subFolderPath, imageFileName);
            targetFile = fullfile(targetFolder,imageFileName);

            % 复制文件
            copyfile(sourceFile, targetFile);
        end
    end
end
