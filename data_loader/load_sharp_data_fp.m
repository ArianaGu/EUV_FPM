%% Loading FP data series from the SHARP EUV microscope
% awojdyla@lbl.gov, June 2023

% Fourier ptychography Microscopy on a phase defect located on IMO113715

% locate the data
folder_defect_imo = './SHARP_2014-10-09_LBNL/';


%read the data and the metadata
[img_defect_imo, meta_defect_imo] = Sharp.read(folder_defect_imo,'IMO113715',26);

% wavelength
lambda_m = 13.5e-9;

% numerical aperture
fc_lens = (asin(0.33/4)/lambda_m);

% effective pixel size
dx_m = 15e-9;

% fetch the region of interest
roi_size_px = 332;
[~,x_roi,y_roi] = Sharp.ROI(img_defect_imo{1},roi_size_px,16,0);

% Effective image size
Dx_m = length(x_roi)*dx_m;

x_m = x_roi*dx_m;
y_m = y_roi*dx_m;

% frequency scaling
freq_cpm = Sharp.fs(x_m);
%%
% display
imagesc(img_defect_imo{65}(:,:))
axis image off

%%

% some data processing: 
load drift_defectimo.mat;
for i=1:81
    % one image out of two has centered illumination, to track drift
    idx = 2*(i-1)+4;
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