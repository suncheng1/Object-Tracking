function [out_npca, out_pca] = get_subwindow(im, pos, model_sz, currentScaleFactor)

if isscalar(model_sz)
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

%check for out-of-bounds coordinates, and set them to the values at
%the borders
xs(xs < 1) = 1;
ys(ys < 1) = 1;
xs(xs > size(im,2)) = size(im,2);
ys(ys > size(im,1)) = size(im,1);

%extract image   根据索引，提取像素值
im_patch = im(ys, xs, :);

%resize image to model size
% im_patch = imresize(im_patch, model_sz, 'bilinear');   改变 patch 大小，使其与filter大小相同
im_patch = mexResize(im_patch, model_sz, 'auto');

% compute non-pca feature map
out_npca = [];

% compute pca feature map     下述结果均标准化至 [-0.5,0.5]
% fhog( I, [binSize], [nOrients] )  binSize默认为8，nOrients默认为9
% 最终维度为[h/binSize w/binSize nOrients*3+5]       
temp_pca = fhog(single(im_patch),4);
temp_pca(:,:,32) = cell_grayscale(im_patch,4);     % 计算4*4像素块为一个cell的平均灰度值

% 将temp_pca的前两维合并
out_pca = reshape(temp_pca, [size(temp_pca, 1)*size(temp_pca, 2), size(temp_pca, 3)]);
end

