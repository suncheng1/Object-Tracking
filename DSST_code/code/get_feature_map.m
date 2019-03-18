function out = get_feature_map(im_patch)

% allocate space
out = zeros(size(im_patch, 1), size(im_patch, 2), 28, 'single');

% if grayscale image  
if size(im_patch, 3) == 1    % 灰度图像
    out(:,:,1) = single(im_patch)/255 - 0.5;     % 标准化至[-0.5,0.5]
    % fhog( I, [binSize], [nOrients] )  binSize默认为8，nOrients默认为9
    % 最终维度为[h/binSize w/binSize nOrients*3+5]  
    temp = fhog(single(im_patch), 1);    
    out(:,:,2:28) = temp(:,:,1:27);    % 最终特征为：a*b*1的标准化灰度值，a*b*[2:28]的fHOG特征 
else   % 彩色图像
    out(:,:,1) = single(rgb2gray(im_patch))/255 - 0.5;    % 先转化为灰度图像，在标准化至[-0.5,0.5]
    temp = fhog(single(im_patch), 1);     % temp: 第三维度为：3*9+5 = 32
    out(:,:,2:28) = temp(:,:,1:27);    % 这里只取前1:27个通道的值
end
