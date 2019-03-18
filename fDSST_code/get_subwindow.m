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

% �����posΪĿ����������ģ����ȡ����Χ���ص�����
xs = floor(pos(2)) + (1:patch_sz(2)) - floor(patch_sz(2)/2);
ys = floor(pos(1)) + (1:patch_sz(1)) - floor(patch_sz(1)/2);

%check for out-of-bounds coordinates, and set them to the values at
%the borders
xs(xs < 1) = 1;
ys(ys < 1) = 1;
xs(xs > size(im,2)) = size(im,2);
ys(ys > size(im,1)) = size(im,1);

%extract image   ������������ȡ����ֵ
im_patch = im(ys, xs, :);

%resize image to model size
% im_patch = imresize(im_patch, model_sz, 'bilinear');   �ı� patch ��С��ʹ����filter��С��ͬ
im_patch = mexResize(im_patch, model_sz, 'auto');

% compute non-pca feature map
out_npca = [];

% compute pca feature map     �����������׼���� [-0.5,0.5]
% fhog( I, [binSize], [nOrients] )  binSizeĬ��Ϊ8��nOrientsĬ��Ϊ9
% ����ά��Ϊ[h/binSize w/binSize nOrients*3+5]       
temp_pca = fhog(single(im_patch),4);
temp_pca(:,:,32) = cell_grayscale(im_patch,4);     % ����4*4���ؿ�Ϊһ��cell��ƽ���Ҷ�ֵ

% ��temp_pca��ǰ��ά�ϲ�
out_pca = reshape(temp_pca, [size(temp_pca, 1)*size(temp_pca, 2), size(temp_pca, 3)]);
end

