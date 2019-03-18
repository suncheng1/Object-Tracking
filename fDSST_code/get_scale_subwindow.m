function [out_pca,out_npca] = get_scale_subwindow(im, pos, base_target_sz, scaleFactors, scale_model_sz)

nScales = length(scaleFactors);

for s = 1:nScales
    patch_sz = floor(base_target_sz * scaleFactors(s));
    
    xs = floor(pos(2)) + (1:patch_sz(2)) - floor(patch_sz(2)/2);
    ys = floor(pos(1)) + (1:patch_sz(1)) - floor(patch_sz(1)/2);
    
    %check for out-of-bounds coordinates, and set them to the values at
    %the borders
    xs(xs < 1) = 1;
    ys(ys < 1) = 1;
    xs(xs > size(im,2)) = size(im,2);
    ys(ys > size(im,1)) = size(im,1);
    
    %extract image
    im_patch = im(ys, xs, :);
    
    % resize image to model size
%     im_patch_resized = imresize(im_patch, scale_model_sz, 'bilinear');
    im_patch_resized = mexResize(im_patch, scale_model_sz, 'auto');
    
    % extract scale features
    temp_hog = fhog(single(im_patch_resized), 4);
    
    % 将尺度特征按尺度因子（尺度大小）分为 nScales 个通道，将每个通道上的hog特征由矩阵变为行向量
    % 即为 temp_hog的三个维度相乘，又因为temp_hog[][][32]为空，所以取前31个矩阵，即为下式
    if s == 1
        dim_scale = size(temp_hog,1)*size(temp_hog,2)*31;     % 32为空
        out_pca = zeros(dim_scale, nScales, 'single');     % 输出为dim_scale*nScales（通道数）
    end 
    
    out_pca(:,s) = reshape(temp_hog(:,:,1:31), dim_scale, 1);
end

out_npca = [];