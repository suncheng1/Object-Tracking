function out = get_translation_sample(im, pos, model_sz, currentScaleFactor, cos_window)

% out = get_subwindow(im, pos, model_sz, currentScaleFactor, cos_window)
% 
% Extracts the a sample for the translation filter at the current location and scale.
% 从当前位置和尺度，提取平移filter样本

if isscalar(model_sz)  %square sub-window   判断输入是否是标量，是标量返回1，反之返回0,
    model_sz = [model_sz, model_sz];
end

patch_sz = floor(model_sz * currentScaleFactor);

%make sure the size is not to small
if patch_sz(1) < 1
    patch_sz(1) = 2;
end;
if patch_sz(2) < 1
    patch_sz(2) = 2;
end;

% 这里的pos为目标区域的中心，因此取其周围像素点索引
xs = floor(pos(2)) + (1:patch_sz(2)) - floor(patch_sz(2)/2);
ys = floor(pos(1)) + (1:patch_sz(1)) - floor(patch_sz(1)/2);

% check for out-of-bounds coordinates, and set them to the values at the borders
% 判断是否越界（小于最小值，超出最大值），若越界，则将越界值设为边界值
xs(xs < 1) = 1;
ys(ys < 1) = 1;
xs(xs > size(im,2)) = size(im,2);
ys(ys > size(im,1)) = size(im,1);

% extract image   根据索引，提取像素值
im_patch = im(ys, xs, :);      %矩阵的行是图像的高，列是图像的宽

% resize image to model size      改变 patch 大小，使其与filter大小相同
% mexResize：原始图像，输出图像，参数”auto“,当新的图像宽大于原始图像的高，使用双线性插值；反之使用INTER_AREA。
im_patch = mexResize(im_patch, model_sz, 'auto');     % mexResize：用OpenCV中resize进行优化，matlab和c++混合编程

% compute feature map     最终特征为：a*b*1的标准化灰度值[-0.5,0.5]，a*b*[2:28]的fHOG特征
out = get_feature_map(im_patch);      
  
% apply cosine window    @times，矩阵点乘
out = bsxfun(@times, cos_window, out);
end

