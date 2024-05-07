clear all;
close all;

% 文件夹路径
folder_path0 = 'J:\SR_VQA\MOS_VSR_Frame\train\sr';
save_path = 'G:\VSR_QAD\VSR_613\ST_30_9';
subfolders0 = dir(folder_path0);
subfolders0 = subfolders0([subfolders0.isdir]);
subfolders0 = subfolders0(3:end);
for r = 1:numel(subfolders0)
    folder_path1 = fullfile(folder_path0, subfolders0(r).name);
    subfolders1 = dir(folder_path1);
    subfolders1 = subfolders1([subfolders1.isdir]);
    subfolders1 = subfolders1(3:end);
    for j = 1:numel(subfolders1)
        folder_path2 = fullfile(folder_path1, subfolders1(j).name);
        subfolders2 = dir(folder_path2);
        subfolders2 = subfolders2([subfolders2.isdir]);
        subfolders2 = subfolders2(3:end);
        for i = 1:numel(subfolders2)
            tic;
            subfolder = fullfile(folder_path2, subfolders2(i).name);

            % 获取文件夹中的所有图片文件
            image_files = dir(fullfile(subfolder, '*.bmp'));
            [filepath,name,ext] = fileparts(image_files(1).name);
            name_num = name(1:8);
            num2_fps=name(10:11);
            frame_rate=str2double(num2_fps); % 将字符串转换为数字
            num1_he=name(16:17);
            num3_sr=name(22:23);
            num4_x=name(19:20);
            out_file=sprintf('%s_%sfps_%s_%s_%s.bmp',name_num,num2_fps,num1_he,num4_x,num3_sr);
            output_path = fullfile(save_path, out_file); % 输出视频的文件名

            T = min(30, numel(image_files));  % 选择文件夹中的前 T 张图片
            W = 1920;
            H = 1080;
            K = 9;

            % 设置随机种子
            rng(42);

            % 生成 9 个 1080 以内的随机数，并排序
            random_numbers = sort(randi(1080, 1, K));
            % random_numbers  = [108,216,324,432,540,648,756,864,972];

            % 初始化结果图像矩阵
            result_image = zeros(T * K, W, 3, 'uint8');

            for k = 1:K
                % 循环读取每张图片并进行处理
                for s = 1:T
                    % 读取图片
                    current_image = imread(fullfile(subfolder, image_files(s).name));

                    cut_line = current_image(random_numbers(k), :, :);

                    % 将结果存储到结果矩阵中
                    result_image((k-1)*T + s, :, :) = cut_line;
                end
            end

            % 显示结果图像
        %     imshow(result_image);

            % 保存结果图像
            imwrite(result_image, fullfile(output_path));
            elapsed_time = toc;
            disp(['完成一个视频切片耗时为：', num2str(elapsed_time), ' 秒']);
        end
    end
end