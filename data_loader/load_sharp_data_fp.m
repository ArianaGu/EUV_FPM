%% Loading FP data series from the SHARP EUV microscope
% awojdyla@lbl.gov, June 2023

% Fourier ptychography Microscopy on a phase defect located on IMO113715

% locate the data
folder_defect_imo = '../real_data/SHARP_2016-04-20_LBNL/';


%read the data and the metadata
[img_defect_imo, meta_defect_imo] = Sharp.read(folder_defect_imo,'IMO288835',16);

% wavelength
lambda_m = 13.5e-9;

% numerical aperture
fc_lens = (asin(0.33/4)/lambda_m);

% effective pixel size
dx_m = 15e-9;

% fetch the region of interest
roi_size_px = 332*5;
[~,x_roi,y_roi] = Sharp.ROI(img_defect_imo{1},roi_size_px,16,0);

% Effective image size
Dx_m = length(x_roi)*dx_m;

x_m = x_roi*dx_m;
y_m = y_roi*dx_m;

% frequency scaling
freq_cpm = Sharp.fs(x_m);
%%
% display
% imagesc(img_defect_imo{65}(:,:))
% axis image off

%%

% some data processing: 
load drift_defectimo2.mat;

for i=1:61
    % one image out of two has centered illumination, to track drift
    idx = 2*(i-1)+2;
    % remove drift, background, rotate by 1.2 deg
    img_temp = Sharp.circshift2(...
                Sharp.removeBG(...
                    img_defect_imo{idx}),x_d_defectimo(i),y_d_defectimo(i));
    img_temp2 = Sharp.rotate(img_temp,1.2);
    % crop to ROI
    img{i} = img_temp2(y_roi,x_roi);
    % get illumination angle from data (in normalized pupil coordindates)
    fx_c{i} = meta_defect_imo{idx}.ma_arg0.*cos(meta_defect_imo{idx}.ma_arg1*pi/180);
    fy_c{i} = meta_defect_imo{idx}.ma_arg0.*sin(meta_defect_imo{idx}.ma_arg1*pi/180);
    fprintf('%d ',i)
end

%% save readout to png files with name coding
% specify path and name
folder_name = '../real_data/full_tile/';
data_name = 'tile';
num_images = length(img);

% decide scale factor
global_max = max(cellfun(@(c) max(c(:)), img));

for i = 1:num_images
    fx_str = sprintf('%+0.2f', fx_c{i}); % format fx_c{i} as a string with sign and 2 decimal places
    fy_str = sprintf('%+0.2f', fy_c{i}); % similarly for fy_c{i}
    file_name = sprintf('%s%s_sx%s_sy%s.png', folder_name, data_name, fx_str, fy_str); % create the file name
%     file_name = sprintf('%s%s_%s.png', folder_name, data_name, num2str(i)); % file name for calib data
    scaled_img = img{i} / global_max;
    scaled_img = uint8(scaled_img * 255);
    imwrite(scaled_img, file_name); % write the image to a PNG file
end
