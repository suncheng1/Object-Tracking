function out = get_translation_sample(im, pos, model_sz, currentScaleFactor, cos_window)

% out = get_subwindow(im, pos, model_sz, currentScaleFactor, cos_window)
% 
% Extracts the a sample for the translation filter at the current location and scale.
% �ӵ�ǰλ�úͳ߶ȣ���ȡƽ��filter����

if isscalar(model_sz)  %square sub-window   �ж������Ƿ��Ǳ������Ǳ�������1����֮����0,
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

% check for out-of-bounds coordinates, and set them to the values at the borders
% �ж��Ƿ�Խ�磨С����Сֵ���������ֵ������Խ�磬��Խ��ֵ��Ϊ�߽�ֵ
xs(xs < 1) = 1;
ys(ys < 1) = 1;
xs(xs > size(im,2)) = size(im,2);
ys(ys > size(im,1)) = size(im,1);

% extract image   ������������ȡ����ֵ
im_patch = im(ys, xs, :);      %���������ͼ��ĸߣ�����ͼ��Ŀ�

% resize image to model size      �ı� patch ��С��ʹ����filter��С��ͬ
% mexResize��ԭʼͼ�����ͼ�񣬲�����auto��,���µ�ͼ������ԭʼͼ��ĸߣ�ʹ��˫���Բ�ֵ����֮ʹ��INTER_AREA��
im_patch = mexResize(im_patch, model_sz, 'auto');     % mexResize����OpenCV��resize�����Ż���matlab��c++��ϱ��

% compute feature map     ��������Ϊ��a*b*1�ı�׼���Ҷ�ֵ[-0.5,0.5]��a*b*[2:28]��fHOG����
out = get_feature_map(im_patch);      
  
% apply cosine window    @times��������
out = bsxfun(@times, cos_window, out);
end

