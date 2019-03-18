function [ cell_gray ] = cell_grayscale( img, w )
%CELL_GRAYSCALE Average the intensity over a single hog-cell
%   Should use the same way to make cells as the fhog function  
% 计算a single hog-cell的平均灰度值

if size(img,3) == 3
   %convert to grayscale
   gray_image = rgb2gray(img);
else
   gray_image = img;
end

gray_image = single(gray_image);


%compute the integral image
iImage = integralImage(gray_image);

i1 = (w:w:size(gray_image,1)) + 1;    % 以w+1开始，步长为w，到gray_image的长+1截止
i2 = (w:w:size(gray_image,2)) + 1;    % 以w+1开始，步长为w，到gray_image的宽+1截止
cell_sum = iImage(i1,i2) - iImage(i1,i2-w) - iImage(i1-w,i2) + iImage(i1-w,i2-w);
cell_gray = cell_sum / (w*w * 255) - 0.5;

end

