function [positions, fps] = dsst(params)

% [positions, fps] = dsst(params)

% parameters
padding = params.padding;                         	%extra area surrounding the target
output_sigma_factor = params.output_sigma_factor;	%spatial bandwidth (proportional to target)
lambda = params.lambda;
learning_rate = params.learning_rate;
nScales = params.number_of_scales;
scale_step = params.scale_step;
scale_sigma_factor = params.scale_sigma_factor;
scale_model_max_area = params.scale_model_max_area;

video_path = params.video_path;
img_files = params.img_files;
pos = floor(params.init_pos);
target_sz = floor(params.wsize);

visualization = params.visualization;

num_frames = numel(img_files);     % 元素个数

init_target_sz = target_sz;

% target size att scale = 1
base_target_sz = target_sz;

% window size, taking padding into account  
% 追踪区域
sz = floor(base_target_sz * (1 + padding));

% 平移滤波器  %产生高斯理想模板
% desired translation filter output (gaussian shaped), bandwidth
% proportional to target size
output_sigma = sqrt(prod(base_target_sz)) * output_sigma_factor;     % Gauss参数  sigma 
[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));     %  x, y; rs、cs对应位置的点构成的组合，能够取边所有可能的组合
y = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));     % 二维高斯分布函数
yf = single(fft2(y));    %将矩阵变为单精度浮点型

% 比例滤波器
% desired scale filter output (gaussian shaped), bandwidth proportional to
% number of scales
scale_sigma = nScales/sqrt(33) * scale_sigma_factor;
ss = (1:nScales) - ceil(nScales/2);    
ys = exp(-0.5 * (ss.^2) / scale_sigma^2);     % 一维高斯分布
ysf = single(fft(ys));

% store pre-computed translation filter cosine window   平移滤波器 余弦窗口    hann(3)=[0 1 0]'
cos_window = single(hann(sz(1)) * hann(sz(2))');       % 降低FFT变换时图像边缘对变换结果的影响

% store pre-computed scale filter cosine window    比例滤波器 加窗
if mod(nScales,2) == 0
    scale_window = single(hann(nScales+1));
    scale_window = scale_window(2:end);
else
    scale_window = single(hann(nScales));    % hann(3)=[0 0.5 1 0.5 0]'
end;

% scale factors   比例因子  尺度金字塔每层比例系数   
ss = 1:nScales;
scaleFactors = scale_step.^(ceil(nScales/2) - ss);     % 以 nScales 种不同的尺度获取候选框，最终提取fHOG特征
   
% compute the resize dimensions used for feature extraction in the scale estimation
scale_model_factor = 1;
if prod(init_target_sz) > scale_model_max_area    % 如果初始目标大小（所包含的像素个数）大于最大值512，则
    scale_model_factor = sqrt(scale_model_max_area/prod(init_target_sz));    % scale模型因子 = sqrt(512/初始大小)
end
scale_model_sz = floor(init_target_sz * scale_model_factor);     % 调整scale模型的大小

currentScaleFactor = 1;   

% to calculate precision
positions = zeros(numel(img_files), 4);

% to calculate FPS
time = 0;

% find maximum and minimum scales   计算最大和最小尺度因子（尺度）
im = imread([video_path  'img/'  img_files{1}]);
min_scale_factor = scale_step ^ ceil(log(max(5 ./ sz)) / log(scale_step));
max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));

for frame = 1:num_frames
    %load image
    im = imread([video_path 'img/'  img_files{frame}]);

    tic;
    
    if frame > 1
        %% （预测目标位置）
        
        %-------针对translation filter--------提取fHOG特征
        % extract the test sample feature map for the translation filter
        % 用
        xt = get_translation_sample(im, pos, sz, currentScaleFactor, cos_window);
        
        % calculate the correlation response of the translation filter   
        % 其中sum( , , 3)   指按照特征通道维，各通道自行进行hf_num .* xtf运算 （第三维的大小为此时的通道数）
        % Y = Ht.* Z ，论文中z表示从目标区域中提取的 Patch z,  也就是这里的 xt；  H = {A/B} = hf_num/hf_den
        % 于是就用到了上一帧中得到的 H ，也就是 H = hf_num/hf_den
        xtf = fft2(xt);
        response = real(ifft2(sum(hf_num .* xtf, 3) ./ (hf_den + lambda)));   % 计算相关分数   公式4
        
        % find the maximum translation response
        [row, col] = find(response == max(response(:)), 1);   % 找出response最大值的索引
        
        % update the position     修正目标位置
        pos = pos + round((-sz/2 + [row, col]) * currentScaleFactor);     % round--四舍五入
        
        %-------针对scale filter--------   
        % extract the test sample feature map for the scale filter     
        % 根据 translation filter 预测的pos 提取 fHOG 尺度特征 5 个通道，然后得出相应最大值，即为当前尺度
        xs = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);
        
        % calculate the correlation response of the scale filter    
        % 此处尺度特征为二维矩阵，sum( , , 1)   指将所有列元素相加，最终得出一个列向量，
        % 按各自通道进行hf_num .* xtf运算         同 translation response
        xsf = fft(xs,[],2);
        scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + lambda)));    % 公式4
        
        % find the maximum scale response     找出response最大值的索引
        recovered_scale = find(scale_response == max(scale_response(:)), 1);
        
        % update the scale    更新currentScaleFactor因子，用于更新 scale
        currentScaleFactor = currentScaleFactor * scaleFactors(recovered_scale);
        if currentScaleFactor < min_scale_factor
            currentScaleFactor = min_scale_factor;
        elseif currentScaleFactor > max_scale_factor
            currentScaleFactor = max_scale_factor;
        end
    end
    
    %%  通过预测的信息，更新追踪模型
    
    % extract the training sample feature map for the translation filter   为 translation filter 提取特征
    % 得到的是样本的fHOG特征图，并且用hann窗口减少图像边缘频率对FFT变换的影响
    xl = get_translation_sample(im, pos, sz, currentScaleFactor, cos_window);
    
    % calculate the translation filter update    公式3
    xlf = fft2(xl);
    new_hf_num = bsxfun(@times, yf, conj(xlf));    % 矩阵对应元素相乘 点乘操作      yf 对应公式中的G期望输出
    new_hf_den = sum(xlf .* conj(xlf), 3);            % 其中sum( , , 3)   指按照各特征通道运算

    
    % extract the training sample feature map for the scale filter   
    % 通过scale filter提取特征    nScales 种不同尺度的候选框得到的fHOG特征  xs 大小为 n * nScales
    xs = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);
    
    % calculate the scale filter update    公式3
    xsf = fft(xs,[],2);
    new_sf_num = bsxfun(@times, ysf, conj(xsf));     % 互相关
    new_sf_den = sum(xsf .* conj(xsf), 1);      % 自相关，按列相加
    
    % 更新分子、分母
    if frame == 1
        % first frame, train with a single image
        hf_den = new_hf_den;
        hf_num = new_hf_num;
        
        sf_den = new_sf_den;
        sf_num = new_sf_num;
    else
        % subsequent frames, update the model   更新滤波器    公式7
        hf_den = (1 - learning_rate) * hf_den + learning_rate * new_hf_den;
        hf_num = (1 - learning_rate) * hf_num + learning_rate * new_hf_num;
        sf_den = (1 - learning_rate) * sf_den + learning_rate * new_sf_den;
        sf_num = (1 - learning_rate) * sf_num + learning_rate * new_sf_num;
    end
    
    % calculate the new target size
    target_sz = floor(base_target_sz * currentScaleFactor);
    
    %save position
    positions(frame,:) = [pos target_sz];
    
    time = time + toc;
    
    
    %visualization
    if visualization == 1
        rect_position = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        if frame == 1  %first frame, create GUI
            figure('Name',['Tracker - ' video_path]);
            im_handle = imshow(uint8(im), 'Border','tight', 'InitialMag', 100 + 100 * (length(im) < 500));
            rect_handle = rectangle('Position',rect_position, 'EdgeColor','g');
            text_handle = text(10, 10, int2str(frame));
            set(text_handle, 'color', [0 1 1]);
        else
            try  %subsequent frames, update GUI
                set(im_handle, 'CData', im)
                set(rect_handle, 'Position', rect_position)
                set(text_handle, 'string', int2str(frame));
            catch
                return
            end
        end
        drawnow
%         pause
    end
end

fps = num_frames/time;
% disp(['fps: ' num2str(fps)])