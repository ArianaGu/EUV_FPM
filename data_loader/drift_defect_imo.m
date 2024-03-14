%% Multi-pass drift correction 
% We will perform a multipass drift correction, progressively refining the
% drift guess, and finally using a subpixel correction on the data

%% reading the data
clear all; clf

%all Matlab functions required in this script are available on git.
%if there is no access, one might want to rewrite this part of the code
%to read the data.
%Filename format is SHARP's default.
img_folder = '../real_data/SHARP_2016-04-20_LBNL/';
N_img = 62;

img_ref = cell(1,N_img);
for i=1:N_img
    [img_ref{i}, meta_ref{i}] = Sharp.read(img_folder,'IMO288835',16,2*(i-1)+1);
    imagesc(img_ref{i})
    axis('equal')
    axis('off')
    colormap("gray")
    drawnow
end
[~,x_roi, y_roi] = Sharp.ROI(img_ref{1}, 256);

% x_roi = x_roi - 512;
% y_roi = y_roi - 512;

gif_show(img_ref, [700,860,930,1090], './center.gif')
gif_show(img_ref, [1040,1200,600,760], './bottom_left.gif')


%% First pass - removing the linear trend
% We first perform a linear drift correction, where the drift is handpicked
% from the first and last image
imagesc(img_ref{1}+img_ref{end})
colormap("gray")
axis image 

%%

%handpicking
% delta_lin = -([178,152] - [136,130]);
delta_lin = [0,0];
%first and last image
idx1 = 1;
idx2 = length(img_ref);
%linear drift, consecutive iamges
dlin = round(delta_lin*(idx2-idx1)/length(img_ref));
clf
imshow(img_ref{idx1}+Sharp.circshift2(img_ref{idx2},-dlin(1),dlin(2)))
colormap gray

% first-pass corrections arrays
x_d1 = round(delta_lin(1)*(1:length(img_ref))/length(img_ref));
y_d1 = round(delta_lin(2)*(1:length(img_ref))/length(img_ref));
img_last = Sharp.circshift2(img_ref{idx2},x_d1(idx2),y_d1(idx2));
imshow(img_ref{idx1}(y_roi,x_roi)+img_last(y_roi,x_roi))
colormap gray

%% Second pass - coarse adjustement with crosscorrelation
% Since the linear drift has now been removed, the crosscorrelation
% centering is much better. We can use this to find the maximum over a
% rather limited support.

clf
hold off
x_d2 = zeros(1,N_img);
y_d2 = zeros(1,N_img);
size_autoco =  32;

ft = @(signal) fftshift( ifft2( ifftshift( signal ) ) );
ift = @(SIGNAL) fftshift( fft2( ifftshift( SIGNAL ) ) );

for i = 1:N_img
    %img{127} is the middle images, with centered illumination; it will be
    %used as a reference for cross-correlation
    XCO = abs(ift(ft(img_ref{i}(y_roi-y_d1(i),x_roi-x_d1(i))).*conj(ft(img_ref{1}(y_roi-y_d1(1),x_roi-x_d1(1))))));
    if mod(i,10)==1
    plot(1:(size_autoco+1),...
        sum(XCO(...
        length(y_roi)/2,((length(x_roi)-size_autoco)/2):(length(x_roi)+size_autoco)/2),1));
    title(sprintf('cross-correlation, y-collapse, barycenter position stdev = %1.3f px', std(y_d2)))
    xlabel('pixel position')
    ylabel('cross-correlation (%first image)')
    end
    % estimate of the maximum using [arg]max
    [~,x_d2(i)] =  max(XCO(length(y_roi)/2,((length(x_roi)-size_autoco)/2):(length(x_roi)+size_autoco)/2));
    [~,y_d2(i)] =  max(XCO(((length(x_roi)-size_autoco)/2):((length(x_roi)+size_autoco)/2),length(x_roi)/2));
    % second pass drift correction arrays (corrected for centering)
    x_d2(i) = x_d2(i)-size_autoco/2-1;
    y_d2(i) = y_d2(i)-size_autoco/2-1;
    axis tight
    drawnow
    hold on
end

%%
x_d_defectimo = x_d1-x_d2;
y_d_defectimo = y_d1-y_d2;
x_d_defectimo = x_d_defectimo - x_d_defectimo(1);
y_d_defectimo = y_d_defectimo - y_d_defectimo(1);


save("drift_defectimo2.mat", "x_d_defectimo", "y_d_defectimo")


%%
idx1 = 1;
idx2 = 62;

img_last = Sharp.circshift2(img_ref{idx2},x_d_defectimo(idx2),y_d_defectimo(idx2));
imagesc(img_ref{idx1}(y_roi,x_roi)-img_last(y_roi,x_roi))
axis image off

%%




%%
for i = 1:N_img
    %img{127} is the middle images, with centered illumination; it will be
    %used as a reference for cross-correlation
    XCO = abs(ift(ft(img_ref{i}(y_roi-y_d1(i)+y_d2(i),x_roi-x_d1(i)+x_d2(i))).*...
             conj(ft(img_ref{1}(y_roi-y_d1(1)+y_d2(1),x_roi-x_d1(1)+x_d2(1))))));
    if mod(i,10)==1
    plot(1:(size_autoco+1),...
        sum(XCO(...
        length(y_roi)/2,((length(x_roi)-size_autoco)/2):(length(x_roi)+size_autoco)/2),1));
    title(sprintf('cross-correlation, y-collapse, barycenter position stdev = %1.3f px', std(y_d2)))
    xlabel('pixel position')
    ylabel('cross-correlation (%first image)')
    end
    drawnow
    hold on
end

%% Third pass - refining with polynomial fit

% if polymax function is not defined, here's a snippet :

% function [ pmax, pidx] = polymax( varargin )
% %POLYMAX Second order maximum estimate
% %    [ pmax, pidx] = polymax( y ) computes the location of the maximum
% %           of array y
% %    [ pmax, pidx] = polymax( x, y ) computes the location of the maximum
% %           of array y, scaled by y
% 
% 
% if nargin == 1
%     y = varargin{1};
%     x = (1:length(y));
% elseif nargin == 2
%     x = varargin{1};
%     y = varargin{2};
% else
%     error('not enough input argument')
% end
% 
% if size(y,2)<size(y,1)
%     y = y';
% end
% 
% p = polyfit(x, y, 2);
% 
% pidx = -p(2)/(2*p(1));
% pmax = polyval(p, pidx);
% end

clf
hold off
x_d3 = zeros(1,N_img);
y_d3 = zeros(1,N_img);
size_autoco =  32;
for i = 1:N_img
    XCO = abs(ift(ft(img_ref{i}(y_roi+y_d1(i)+y_d2(i),x_roi+x_d1(i)+x_d2(i)))...
        .*conj(ft(img_ref{1}(y_roi+y_d1(1)+y_d2(1),x_roi+x_d1(1)+x_d2(1))))));
    if mod(i,10)==1
    plot(1:(size_autoco+1),...
        sum(XCO(...
        length(y_roi)/2,((length(x_roi)-size_autoco)/2):(length(x_roi)+size_autoco)/2),1));
    title(sprintf('cross-correlation, y-collapse, barycenter position stdev = %1.3f px', std(y_d2)))
    xlabel('pixel position')
    ylabel('cross-correlation (%first image)')
    end
    %centroid(-16:16,sum(AUTO(496:528,496:528),1).^1);
    
    %polynomial fitting to get the exact maximum 
    [~,x_d3(i)] =  polymax(1:(size_autoco+1),XCO(length(y_roi)/2,((length(x_roi)-size_autoco)/2):(length(x_roi)+size_autoco)/2));
    [~,y_d3(i)] =  polymax(1:(size_autoco+1),XCO(((length(x_roi)-size_autoco)/2):((length(x_roi)+size_autoco)/2),length(x_roi)/2));
    % third-pass drift corraction arrays
    x_d3(i) = x_d3(i)-size_autoco/2-1;
    y_d3(i) = y_d3(i)-size_autoco/2-1;
    axis tight
    drawnow
    hold on
end

%%
%displaying the drift
clf
imsum = zeros(length(y_roi),length(x_roi));
for i=1:length(img_ref)
    idx = i;
    imsum = imsum + img_ref{idx}(y_roi+y_d1(idx)+y_d2(idx)+round(y_d3(idx)),...
        x_roi+x_d1(idx)+x_d2(idx)+round(x_d3(idx)));
end
imshow(imsum)
colormap gray


%% Final sub-pixel registration
clf

%sum of all drift corrections (linear+argmax+polymax)
dx = x_d1 + x_d2 + x_d3;
dy = y_d1 + y_d2 + y_d3;
%normalized frequency meshgrid
[FX,FY] = meshgrid((-(length(x_roi)/2):((length(x_roi)/2)-1))/length(x_roi));
img_corr = cell(1,N_img);
for i = 1:N_img
    %correcting the drift using a linear phase in the Fourier domain.
    %Requires centering etc... fftshifts are correctly interwinded !
    % (there is a correct zero-centering at stake here)
    img_corr{i} = abs(fftshift(ifft2(ifftshift( ...
                    fftshift(fft2(ifftshift( img_ref{i}(y_roi,x_roi))))...
                    .* exp(1i*2*pi*(FX*dx(i)+FY*dy(i))))...
                  )));
end
%%
%%dislay
imsum = zeros(length(y_roi),length(x_roi));
for i=1:length(img_ref)
    idx = i;
    imsum = imsum + img_corr{idx};
end
imshow(imsum)
colormap gray

%% sum of all images
imsum = zeros(length(y_roi),length(x_roi));
for i=1:N_img
    idx = i;
    imsum = imsum + img_corr{idx};
end
imshow(imsum)
colormap gray


%%
idx = round(rand()*20);
imagesc(img_ref{idx}(y_roi,x_roi));
imagesc(img_ref{idx}(y_roi+y_d1(idx),x_roi+x_d1(idx)));
imagesc(img_ref{idx}(y_roi+y_d1(idx)+y_d2(idx),x_roi+x_d1(idx)+x_d2(idx)));
imagesc(img_ref{idx}(y_roi+y_d1(idx)+y_d2(idx)+round(y_d3(idx)),x_roi+x_d1(idx)+x_d2(idx)+round(x_d3(idx))));
    axis image off

%%
function gif_show(img_ref, bound, filename)
    % save regional comparison to gif
    for i = 1:length(img_ref)
        imagesc(img_ref{i}(bound(1):bound(2), bound(3):bound(4)));
        colormap('gray');
        axis('equal')
        axis('off')
        drawnow
        frame = getframe(gcf); 
        im = frame2im(frame); 
        [imind, cm] = rgb2ind(im, 256); 
    
        % Write to the GIF File 
        if i == 1 
            imwrite(imind, cm, filename, 'gif', 'Loopcount', inf, 'DelayTime',0.05); 
        else 
            imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime',0.05); 
        end
    end
end

function [ pmax, pidx] = polymax( varargin )
%POLYMAX Second order maximum estimate
%    [ pmax, pidx] = polymax( y ) computes the location of the maximum
%           of array y
%    [ pmax, pidx] = polymax( x, y ) computes the location of the maximum
%           of array y, scaled by y

    if nargin == 1
        y = varargin{1};
        x = (1:length(y));
    elseif nargin == 2
        x = varargin{1};
        y = varargin{2};
    else
        error('not enough input argument')
    end
    
    if size(y,2)<size(y,1)
        y = y';
    end
    
    p = polyfit(x, y, 2);
    
    pidx = -p(2)/(2*p(1));
    pmax = polyval(p, pidx);
end