function results = fDSST(params)

padding = params.padding;
output_sigma_factor = params.output_sigma_factor;
lambda = params.lambda;
interp_factor = params.interp_factor;
refinement_iterations = params.refinement_iterations;
translation_model_max_area = params.translation_model_max_area;
nScales = params.number_of_scales;
nScalesInterp = params.number_of_interp_scales;    
scale_step = params.scale_step;
scale_sigma_factor = params.scale_sigma_factor;
scale_model_factor = params.scale_model_factor;
scale_model_max_area = params.scale_model_max_area;
interpolate_response = params.interpolate_response;
num_compressed_dim = params.num_compressed_dim;

video_path = params.video_path;

s_frames = params.s_frames;
pos = floor(params.init_pos);
target_sz = floor(params.wsize * params.resize_factor);    % 当前尺度下的目标大小

visualization = params.visualization;

num_frames = numel(s_frames);
init_target_sz = target_sz;

% 计算当前目标最大的尺度因子   
if prod(init_target_sz) > translation_model_max_area
    currentScaleFactor = sqrt(prod(init_target_sz) / translation_model_max_area);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale   根据当前的尺度因子设置目标大小
base_target_sz = target_sz / currentScaleFactor;

%window size, taking padding into account   目标追踪区域
sz = floor( base_target_sz * (1 + padding ));

featureRatio = 4;

% 平移滤波器
% desired translation filter output (gaussian shaped), bandwidth
% proportional to target size
output_sigma = sqrt(prod(floor(base_target_sz/featureRatio))) * output_sigma_factor;
use_sz = floor(sz/featureRatio);
rg = circshift(-floor((use_sz(1)-1)/2):ceil((use_sz(1)-1)/2), [0 -floor((use_sz(1)-1)/2)]);
cg = circshift(-floor((use_sz(2)-1)/2):ceil((use_sz(2)-1)/2), [0 -floor((use_sz(2)-1)/2)]);

[rs, cs] = ndgrid( rg,cg);
y = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
yf = single(fft2(y));     %将矩阵变为单精度浮点型

interp_sz = size(y) * featureRatio;
cos_window = single(hann(floor(sz(1)/featureRatio))*hann(floor(sz(2)/featureRatio))' );

% 尺度滤波器   scale filter
if nScales > 0
    scale_sigma = nScalesInterp * scale_sigma_factor;
    
    scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2)) * nScalesInterp/nScales;
    scale_exp_shift = circshift(scale_exp, [0 -floor((nScales-1)/2)]);
    
    interp_scale_exp = -floor((nScalesInterp-1)/2):ceil((nScalesInterp-1)/2);
    interp_scale_exp_shift = circshift(interp_scale_exp, [0 -floor((nScalesInterp-1)/2)]);
    
    scaleSizeFactors = scale_step .^ scale_exp;
    interpScaleFactors = scale_step .^ interp_scale_exp_shift;
    
    % desired scale filter output (gaussian shaped), bandwidth proportional to
    % number of scales
    ys = exp(-0.5 * (scale_exp_shift.^2) /scale_sigma^2);
    ysf = single(fft(ys));
    scale_window = single(hann(size(ysf,2)))';
    
    %make sure the scale model is not to large, to save computation time
    if scale_model_factor^2 * prod(init_target_sz) > scale_model_max_area
        scale_model_factor = sqrt(scale_model_max_area/prod(init_target_sz));
    end
    
    %set the scale model size
    scale_model_sz = floor(init_target_sz * scale_model_factor);
    
    im = imread([video_path  'img/'  s_frames{1}]);
    
    %force reasonable scale changes    计算最大和最小尺度因子（尺度）
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
    
    max_scale_dim = strcmp(params.s_num_compressed_dim,'MAX');
    if max_scale_dim     % strcmp两参数相同返回1，反之0
        s_num_compressed_dim = length(scaleSizeFactors);
    else
        s_num_compressed_dim = params.s_num_compressed_dim;
    end
end

% initialize the projection matrix
projection_matrix = [];
rect_position = zeros(num_frames, 4);

time = 0;

for frame = 1:num_frames
    %load image
    im = imread([video_path  'img/'  s_frames{frame}]);
    
    tic();
    
    % do not estimate translation and scaling on the first frame, since we 
    % just want to initialize the tracker there
    % 不要尝试估计第一个帧上的目标信息 
    if frame > 1
        %% （预测目标位置和尺度）
        old_pos = inf(size(pos));
        iter = 1;
        
        %% translation search  位置估计
        while iter <= refinement_iterations && any(old_pos ~= pos)
            [xt_npca, xt_pca] = get_subwindow(im, pos, sz, currentScaleFactor);
            
            xt = feature_projection(xt_npca, xt_pca, projection_matrix, cos_window);
            xtf = fft2(xt);
            
            responsef = sum(hf_num .* xtf, 3) ./ (hf_den + lambda);    % 相关性分数，公式8
            
            % if we undersampled features, we want to interpolate the
            % response so it has the same size as the image patch   
            % 如果我们对特征进行欠采样，想要插值响应，须使其具有与图像块相同的大小
            if interpolate_response > 0
                if interpolate_response == 2
                    % use dynamic interp size  
                    interp_sz = floor(size(y) * featureRatio * currentScaleFactor);
                end               
                responsef = resizeDFT2(responsef, interp_sz);
            end
            
            response = ifft2(responsef, 'symmetric');
            % disp_row，disp_col 为目标与当前目标中心的平移参数
            [row, col] = find(response == max(response(:)), 1);
            disp_row = mod(row - 1 + floor((interp_sz(1)-1)/2), interp_sz(1)) - floor((interp_sz(1)-1)/2);
            disp_col = mod(col - 1 + floor((interp_sz(2)-1)/2), interp_sz(2)) - floor((interp_sz(2)-1)/2);
            
            switch interpolate_response
                case 0
                    translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor);
                case 1
                    translation_vec = round([disp_row, disp_col] * currentScaleFactor);
                case 2
                    translation_vec = [disp_row, disp_col];
            end
            
            old_pos = pos;
            pos = pos + translation_vec;     % 目标中心修正
            
            iter = iter + 1;
        end
        
        %% scale search   尺度估计
        if nScales > 0         
            % create a new feature projection matrix    
            % 获得 n * nScales 的特征矩阵，其中 nScales为通道数
            [xs_pca, xs_npca] = get_scale_subwindow(im,pos,base_target_sz,currentScaleFactor*scaleSizeFactors,scale_model_sz);
            % 再将获得每个通道的特征降维至 nScales（将 n降维至nScales），最终得到的特征矩阵为nScales*nScales
            xs = feature_projection_scale(xs_npca,xs_pca,scale_basis,scale_window);
            xsf = fft(xs,[],2);
            
            scale_responsef = sum(sf_num .* xsf, 1) ./ (sf_den + lambda);
            
            interp_scale_response = ifft( resizeDFT(scale_responsef, nScalesInterp), 'symmetric');
            
            recovered_scale_index = find(interp_scale_response == max(interp_scale_response(:)), 1);
        
            % set the scale  设置当前尺度因子（尺度金字塔每层之间比例系数）
            currentScaleFactor = currentScaleFactor * interpScaleFactors(recovered_scale_index);            
            %adjust to make sure we are not to large or to small
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end
        end
    end
    
    %%  通过预测的信息，更新追踪模型
    %this is the training code used to update/initialize the tracker
    
    %Compute coefficients for the tranlsation filter    计算 tranlsation filter 的参数
    [xl_npca, xl_pca] = get_subwindow(im, pos, sz, currentScaleFactor);
    
    if frame == 1
        h_num_pca = xl_pca;
        h_num_npca = xl_npca;
        
        % set number of compressed dimensions to maximum if too many
        % 将压缩后的维度设为 两者之间的较小值
        num_compressed_dim = min(num_compressed_dim, size(xl_pca, 2));
    else
        h_num_pca = (1 - interp_factor) * h_num_pca + interp_factor * xl_pca;
        h_num_npca = (1 - interp_factor) * h_num_npca + interp_factor * xl_npca;
    end;
    
    data_matrix = h_num_pca;
    
    % 这里 data_matrix 为 n*通道数,相当于一行为一个样本，则协方差矩阵为 data_matrix'*data_matrix
    [pca_basis, ~, ~] = svd(data_matrix' * data_matrix);    
    projection_matrix = pca_basis(:, 1:num_compressed_dim);    % 取前num_compressed_dim项为投影矩阵
    
    hf_proj = fft2(feature_projection(h_num_npca, h_num_pca, projection_matrix, cos_window));
    hf_num = bsxfun(@times, yf, conj(hf_proj));     % At，公式7a中
    
    xlf = fft2(feature_projection(xl_npca, xl_pca, projection_matrix, cos_window));
    new_hf_den = sum(xlf .* conj(xlf), 3);     % Bt，公式7b中
    
    if frame == 1
        hf_den = new_hf_den;
    else
        %更新分母，公式7b      因为作者所用更新方法中，分子无须更新，因此只跟新分母
        hf_den = (1 - interp_factor) * hf_den + interp_factor * new_hf_den;    
    end
    
    %Compute coefficents for the scale filter
    if nScales > 0
        
        %create a new feature projection matrix
        [xs_pca, xs_npca] = get_scale_subwindow(im, pos, base_target_sz, currentScaleFactor*scaleSizeFactors, scale_model_sz);
        %更新 s_num (用上一帧的s_num和当前帧的xs_pca）
        if frame == 1
            s_num = xs_pca;
        else     
            s_num = (1 - interp_factor) * s_num + interp_factor * xs_pca;
        end;
        
        bigY = s_num;
        bigY_den = xs_pca;
        
        if max_scale_dim       % qr分解：将矩阵分解为一个正交矩阵和一个上三角矩阵的乘积
            [scale_basis, ~] = qr(bigY, 0);     % 如果bigY(m*n), m > n，只计算Q的前n列和R的前n行
            [scale_basis_den, ~] = qr(bigY_den, 0);
        else
            [U,~,~] = svd(bigY,'econ');
            scale_basis = U(:,1:s_num_compressed_dim);
        end
        scale_basis = scale_basis';
        
        % create the filter update coefficients
        % 将s_num的每个通道中的特征信息降维至 nScales 个维度
        sf_proj = fft(feature_projection_scale([],s_num,scale_basis,scale_window),[],2);
        sf_num = bsxfun(@times,ysf,conj(sf_proj));     %分子，公式7a中
        
        xs = feature_projection_scale(xs_npca,xs_pca,scale_basis_den',scale_window);
        xsf = fft(xs,[],2);
        new_sf_den = sum(xsf .* conj(xsf),1);     %分母，公式7b中
        
        if frame == 1
            sf_den = new_sf_den;
        else       %更新分母，公式7b      因为作者所用更新方法中，分子无须更新，因此只跟新分母
            sf_den = (1 - interp_factor) * sf_den + interp_factor * new_sf_den;
        end;
    end
    
    target_sz = floor(base_target_sz * currentScaleFactor); 
    %save position and calculate FPS
    rect_position(frame,:) = [pos([2,1]) - floor(target_sz([2,1])/2), target_sz([2,1])];
    
    time = time + toc();
    
    %% visualization
    if visualization == 1
        rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        if frame == 1
            figure;
            im_handle = imshow(im, 'Border','tight', 'InitialMag', 100 + 100 * (length(im) < 500));
            rect_handle = rectangle('Position',rect_position_vis, 'EdgeColor','g');
            text_handle = text(10, 10, int2str(frame));
            set(text_handle, 'color', [0 1 1]);
        else
            try
                set(im_handle, 'CData', im)
                set(rect_handle, 'Position', rect_position_vis)
                set(text_handle, 'string', int2str(frame));
                
            catch
                return
            end
        end
        
        drawnow
        %pause
    end
end

fps = numel(s_frames) / time;

results.type = 'rect';
results.res = rect_position;
results.fps = fps;
