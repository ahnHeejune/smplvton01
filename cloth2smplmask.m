% 
clear all;  % all variable cleared
close all;  % all figures closed

%{
This is not for Lips classification

labels = {"background", #     0
            "hat", #            1
            "hair", #           2 
            "sunglass", #       3
            "upper-clothes", #  4
            "skirt",  #          5
            "pants",  #          6
            "dress", #          7
            "belt", #           8
            "left-shoe", #      9
            "right-shoe", #     10
            "face",  #           11
            "left-leg", #       12
            "right-leg", #      13
            "left-arm",#       14
            "right-arm", #      15   
            "bag", #            16
            "scarf" #          17    
        ]  
%}

addpath('shape_context')

smpl_model = true;

%%%%%%% VITON DATASET %%%%%%%%%%%%%%%%%%%%
%%%  train -- cloth      : cloth images [hxw =256x192]  jpg
%%%           cloth-mask : FG mask of cloth images [fg: white]  %%% Some are not clean, JPG ^^
%%%           image      : model image [256x192x3] jpg 
%%%           image-pare : segmentation label image PNG
%%%           pose       : joint info JSON 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

backward = true; 
scale_down = false;


%DATA_TOP ='D:\3.Project\9.Fashion\3.Dataset\VITON_TPS\viton_resize\train';
DATA_TOP ='D:\3.Project\9.Fashion\3.Dataset\VITON_TPS\viton_resize\test';

RESULT_DIR = './scmm_results/';

%DATA_ROOT='cloth/woman_top';               % in-shop cloth 
DATA_ROOT= [DATA_TOP,'/image-parse/'];       
MODEL_ROOT=[DATA_TOP, '/image/'];       

%MASK_DIR='results/stage1/tps/00015000_';   % mask for cloth area in model using NN model
MASK_DIR  = [DATA_TOP,'/cloth-mask/'];
CLOTH_DIR = [DATA_TOP,'/cloth/'];

% Check if using MATLAB or Octave
isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
if(isOctave)
  % Load image package for resizing images
  pkg load image;
  % Turn off warning
  warning('off', 'Octave:possible-matlab-short-circuit-operator');
end

%pairs_file = 'data/viton_test_pairs_classified_same.txt';  % 'data/viton_test_pairs.txt'
%pairs_file = 'data/viton_test_pairs_classified_diff.txt';  % 'data/viton_test_pairs.txt'
pairs_file = 'data/viton_smplmatching.txt';  % 'data/viton_test_pairs.txt'


[image1, image2, comment] = textread(pairs_file, '%s %s %s');

%
% @TODO: improve target area with paramgers of input cloth 
%
if smpl_model
    
    smpl_mask_original = imread('templatemask.png');  %'smplmaskref.png');
    smpl_mask_long = smpl_mask_original;  % it is deep copy in matlab
    smpl_mask_long(200:end, :) = 0; % hide the hands and legs for maskt 
    smpl_mask_long(1:65, :) = 0; % hide neck and head 
    
    smpl_mask_short = smpl_mask_long;  % it is deep copy in matlab
    % get the body bounddary
    body_line = zeros(size(smpl_mask_short,1),2);
    mid_x = round(size(smpl_mask_short,2)/2);
    for y = 1: size(smpl_mask_short,1) 
        for x= mid_x: size(smpl_mask_short,2)
                if smpl_mask_short(y, x) <= 0  % out of body 
                    break;
                end
              
        end
        body_line(y,1) = mid_x - (x - mid_x); % start of body 
        body_line(y, 2) = x; % end of body 
    end
    
     for y = 1: size(smpl_mask_short,1) 
        for x= 1: size(smpl_mask_short,2)
            % delete out side of body 
            if x < body_line(y,1) || x > body_line(y,2)
                smpl_mask_short(y,x) = 0;     
            end
        end 
     end
     
     
     %imshow(smpl_mask_short);
     %axis('image');
end


     

% using a smaller height and width for the shape context matching
% can save time without hurting the perform too much.

% original image size
h_o = 256;
w_o = 192;

if scale_down
    h = h_o/4;
    w = w_o/4;
else
    h = h_o;
    w = w_o;    
end

% we use 10x10 control_points
n_control = 100;
for i = 1:length(image1) % only run over 1 image (for now)
    
    if i == 1
        smpl_mask = smpl_mask_long;
    elseif  i == 2 || i == 3
        smpl_mask = smpl_mask_short;
    end  
        
    
    
    image_name1 = image1{i};
    image_name2 = image2{i};
    
    if exist([MASK_DIR, image_name1, '_', image_name2, '_tps.mat'])
        disp('already done');
        %continue;
    end
    
    
    TOP_LABEL = 5; 
    % MASK in model image
    V1 = imread([DATA_ROOT, image_name2]);
    model_original_mask = imread([DATA_ROOT, image_name2]);  % for later use 
    [h0, w0, ~] = size(V1);
    if ~backward
        grayImage = imresize(im2double(V1), [h,w]);
        orig_im_mask = cat(3, grayImage*255, grayImage*255, grayImage*255);
    end
    % extract fashion item masks
    if false 
        V1 = V1(:,:,1) ~= 255 & V1(:,:,2) ~= 255 & V1(:,:,3) ~= 255;
        V1 = imresize(double(V1), [h,w], 'nearest');
    else
        if ~smpl_model
            V1 = (V1 == TOP_LABEL);
        else
            V1  = (smpl_mask > 200);
        end
        V1 = imresize(double(V1), [h,w], 'nearest');
      
        % model 
        model_img_name = strrep(image_name2,'png','jpg');
        model = imread([MODEL_ROOT, model_img_name]);
        model = imresize(im2double(model), [h,w]);
        
    end
    
    if ~smpl_model
        V1 = imfill(V1);
        V1 = medfilt2(V1);
    else
        se = strel('square',3);
        V1 = imdilate(V1, se);
    end
    % Load product mask of image.
    % AHN: needs to generate using the 'parsed' image  
    if false
        V2 = load([MASK_DIR, image_name1, '_', image_name2, '_mask.mat']); % stored in mat format where?
        V2 = imresize(double(V2.mask), [h,w]);
    else
        V2 = imread([MASK_DIR, image_name1]);  %% stupid JPEG
        V2 = (V2 > 128);
        V2 = imresize(double(V2), [h,w],'nearest');
        
        cloth = imread([CLOTH_DIR, image_name1]);
        cloth = imresize(cloth, [h,w]);
    end    
   
    if backward
        grayImage = imresize(im2double(V2), [h,w]);
        orig_im_mask = cat(3, grayImage*255, grayImage*255, grayImage*255);
    end
    
    
    % CHECK the input to Shape Context 
    fig = figure;
    if backward
         subplot(2,4,3);
    else
         subplot(2,4,2);
    end
    imshow(uint8(V1*255.0));  % instead of imagesc(V1)
    axis('image');
    title('mask in model image');  
    
    
    if backward
        subplot(2,4,2);
    else
        subplot(2,4,3);
    end    
    imshow(uint8(V2*255));
    axis('image');
    title('cloth mask'); 
    
    
    subplot(2,4,1);
    imshow(cloth);
    axis('image');
    title(['cloth', image1{i}]); 
    
    subplot(2,4,4);
    if ~smpl_model
        %imshow(model);
        imagesc(model); 
        axis('image');
    else
        imagesc(smpl_mask); 
        axis('image'); 
    end
    title(['model:', comment{i}]); 

    
    % TPS transformation
    % Paramter estimation (in fact, grid/control points at in shop cloth (src)
    % V1 (orig_im)   => TPS => V2 (warped_im)
    %    keypoints1                kypoints2
    %   
    try
        if backward
              tic;[keypoints1, keypoints2, warp_points0, warped_cloth, warped_mask] = tps_main(V2, V1, n_control, im2double(cloth), orig_im_mask, 0);toc;
              warped_mask(isnan(warped_mask)) = 0.0; 
              
              %tic;[keypoints1, keypoints2, warp_points0, warped_cloth] = tps_main(V2, V1, n_control, im2double(cloth), 0);toc;
              warped_cloth(isnan(warped_cloth)) = 255.0; 
        
        else
              tic;[keypoints1, keypoints2, warp_points0, warp_im] = tps_main(V1, V2, n_control, orig_im, 0);toc;
        end
        
    catch ME
        % when there is not enough keypoints for estimating the TPS
        % transformation
        disp('not enough keypoints')
        disp(ME)
        continue
    end
    
    % CHECK the input to Shape Context 
    figure(fig);
    subplot(2,4,5);
    imagesc(orig_im_mask); %imshow(uint8(orig_im*255.0));  % instead of imagesc(V1)
    axis('image');
    title('orig im');
    
    subplot(2,4,6);
    imagesc(warped_mask); %imshow(uint8(warp_im*255.0));
    axis('image');
    title('warp mask');
    
    subplot(2,4,7);
    imagesc(warped_cloth); %imshow(uint8(warp_im*255.0));
    axis('image');
    title('warp cloth');
    
    subplot(2,4,8);
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Blending
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    alpha_ch = warped_mask;
    alpha_ch = alpha_ch/255.0;  % binary alpha 
    
    % clear the cloth area 
    % simply masking
    %    model_wo_cloth = model .* ~V1; 
   
    if smpl_model
        model_wo_cloth =  cat(3, smpl_mask, smpl_mask, smpl_mask);
    else
        % inpainting
        model_wo_cloth =  cat(3, regionfill(model(:,:,1),V1),regionfill(model(:,:,2),V1),regionfill(model(:,:,2),V1));
    end
    
    overlayed = im2double(model_wo_cloth) .* ( 1 - alpha_ch) + warped_cloth .* alpha_ch;    
    
    % restore the hair (face), hair, hands, (pants) etc except the target cloth 
    if ~smpl_model
        face_hairs_arms_mask =  model_original_mask == 11 | model_original_mask ==  11 | model_original_mask == 14  |  model_original_mask == 15;
        overlayed = overlayed .* ( 1 - face_hairs_arms_mask) + model .* face_hairs_arms_mask;    
    end  
       
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Evaluate the result 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    has_gt = false;
    if has_gt
        % evalaute the GMM by IoU
        % 1. convert to logical type 
        cloth_mask_gt =   V1 > 0;  %uint8(V1);
        cloth_mask_est =  warped_mask(:,:, 1) > 0.0; % 3 channels are same 
        %iouval = evaluateSemanticSegmentation(cloth_mask_gt, cloth_mask_est);  
        uinon_area = cloth_mask_gt | cloth_mask_est;
        intersect_area = cloth_mask_gt & cloth_mask_est;
        xor_area = xor( cloth_mask_gt, cloth_mask_est);
        iouval = sum(intersect_area(:))/sum(uinon_area(:));

        % evaluate the TON by SSIM 
        [ssimval,ssimmap] = ssim(overlayed, model); 

        msg = sprintf('IOU=%f, SSIM=%f', iouval, ssimval);
        disp(msg);
    end
    
    imagesc(overlayed); %imshow(uint8(warp_im*255.0))
    axis('image');
    if exist('msg')
        title(['overlayed(', msg, ')']);
    else
        title(['overlayed']);
    end
        drawnow;
    
    % SAVING
    %filename = [RESULT_DIR, image_name2, '_', image_name1, '_result.jpg'];
    %saveas(fig, filename)
    filename = [RESULT_DIR, image_name2, '_', image_name1, '_overlayed.png'];
    imwrite(overlayed, filename);
    filename = [RESULT_DIR, image_name2, '_', image_name1, '_2dwarped.png'];
    imwrite(warped_cloth, filename);
    filename = [RESULT_DIR, image_name2, '_', image_name1, '_2dwarpedmask.png'];
    imwrite(warped_mask, filename);
   
end
