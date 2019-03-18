function out = get_scale_sample(im, pos, base_target_sz, scaleFactors, scale_window, scale_model_sz)

% out = get_scale_sample(im, pos, base_target_sz, scaleFactors, scale_window, scale_model_sz)
% 
% Extracts a sample for the scale filter at the current location and scale.
% 为 scale filter 提取样本

nScales = length(scaleFactors);    % 尺度个数，也就是最终的尺度通道个数
 
for s = 1:nScales
    patch_sz = floor(base_target_sz * scaleFactors(s));      % 根据尺度因子，对目标大小进行缩放
    
    % 与 translation filter 相同，记录目标中心周围的像素索引
    xs = floor(pos(2)) + (1:patch_sz(2)) - floor(patch_sz(2)/2);
    ys = floor(pos(1)) + (1:patch_sz(1)) - floor(patch_sz(1)/2);
    
    % check for out-of-bounds coordinates, and set them to the values at
    % the borders
    xs(xs < 1) = 1;
    ys(ys < 1) = 1;
    xs(xs > size(im,2)) = size(im,2);
    ys(ys > size(im,1)) = size(im,1);
    
    % extract image     按像素索引，读取像素值
    im_patch = im(ys, xs, :);
    
    % resize image to model size      改变 patch 大小，使其与filter大小相同
    im_patch_resized = mexResize(im_patch, scale_model_sz, 'auto');
    
    % extract scale features   
    % fhog( I, [binSize], [nOrients] )  binSize默认为8，nOrients默认为9
    % 最终维度为[h/binSize w/binSize nOrients*3+5]      
    temp_hog = fhog(single(im_patch_resized), 4);       % 32为空
    temp = temp_hog(:,:,1:31);
    
    if s == 1
        out = zeros(numel(temp), nScales, 'single');    % numel()  矩阵元素数
    end
    
    % window
    out(:,s) = temp(:) * scale_window(s);     % temp(:) 转化为列向量(按列读取) * 当前尺度（通道数）
end