function out = get_feature_map(im_patch)

% allocate space
out = zeros(size(im_patch, 1), size(im_patch, 2), 28, 'single');

% if grayscale image  
if size(im_patch, 3) == 1    % �Ҷ�ͼ��
    out(:,:,1) = single(im_patch)/255 - 0.5;     % ��׼����[-0.5,0.5]
    % fhog( I, [binSize], [nOrients] )  binSizeĬ��Ϊ8��nOrientsĬ��Ϊ9
    % ����ά��Ϊ[h/binSize w/binSize nOrients*3+5]  
    temp = fhog(single(im_patch), 1);    
    out(:,:,2:28) = temp(:,:,1:27);    % ��������Ϊ��a*b*1�ı�׼���Ҷ�ֵ��a*b*[2:28]��fHOG���� 
else   % ��ɫͼ��
    out(:,:,1) = single(rgb2gray(im_patch))/255 - 0.5;    % ��ת��Ϊ�Ҷ�ͼ���ڱ�׼����[-0.5,0.5]
    temp = fhog(single(im_patch), 1);     % temp: ����ά��Ϊ��3*9+5 = 32
    out(:,:,2:28) = temp(:,:,1:27);    % ����ֻȡǰ1:27��ͨ����ֵ
end
