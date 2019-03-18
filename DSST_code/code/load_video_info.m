function [img_files, pos, target_sz, ground_truth, video_path] = load_video_info(video_path)

% [img_files, pos, target_sz, ground_truth, video_path] = load_video_info(video_path)

text_files = dir([video_path 'groundtruth_rect.txt']);
assert(~isempty(text_files), 'No initial position and ground truth (groundtruth_rect.txt) to load.')

f = fopen([video_path text_files(1).name]);
ground_truth = textscan(f, '%f,%f,%f,%f');  %[x, y, width, height]
ground_truth = cat(2, ground_truth{:});  %按行连接 [g1,g2]
fclose(f);

text_files = dir([video_path '*_frames.txt']);
if ~isempty(text_files)
    f = fopen([video_path text_files(1).name]);
	frames = textscan(f, '%f,%f');
	fclose(f);

	%see if they are in the 'imgs' subfolder or not
	if exist([video_path 'img/' num2str(frames{1}, '%04i.png')], 'file')
         %list the files
         img_files = num2str((frames{1} : frames{2})', '%04i.png');
    elseif exist([video_path 'img/' num2str(frames{1}, '%04i.jpg')], 'file')
        %list the files
         img_files = num2str((frames{1} : frames{2})', '%04i.jpg');
    else
        error('No image files to load.')
    end

	%list the files
	img_files = num2str((frames{1} : frames{2})', '%04i.jpg');
	img_files = cellstr(img_files);
else
	%no text file, just list all images
	img_files = dir([video_path 'img/' '*.png']);
	if isempty(img_files)
		img_files = dir([video_path 'img/' '*.jpg']);
		assert(~isempty(img_files), 'No image files to load.')
	end
	img_files = sort({img_files.name});
end

%set initial position and size     这里pos指目标所在区域的左上角位置
target_sz = [ground_truth(1,4), ground_truth(1,3)];
pos = [ground_truth(1,2), ground_truth(1,1)];

ground_truth = [ground_truth(:,[2,1]) + (ground_truth(:,[4,3]) - 1) / 2 , ground_truth(:,[4,3])];

end

