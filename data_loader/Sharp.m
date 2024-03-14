classdef Sharp < handle
    %SHARP Image processing suite for the SHARP EUV Microscope data
    %   
    % SHARP can be used as a static class to perform many kind of
    %   file-handling and data processing e.g. :
    %     folder = '/data/SHARP_2014-08-29/';
    %     img_gf = Sharp.removeBG(Sharp.read(folder,'GREMLIN3_AGC',6));
    %
    % The software is provided "as is", 
    % without warranty of any kind, express or implied

    % A Wojdyla, CXRO/LBNL
    % awojdyla@lbl.gov
    % September 2014 - August 2015
    
    % Last update : 08/15/2015


    properties (Constant)
        % Constants from the SHARP EUV Microscope
        pixel_size_m = 13.5e-6;
        lambda_m     = 13.5e-9;
        na4x  =     [0.25, 0.33, 0.35, 0.42, 0.50, 0.625]
        mag =       [ 900,  900,  900, 1125, 1250,  1634]
        px_nm =     [  15,   15,   15,   12, 10.8, 8.262]
        cra_deg =   [   6,    6,    6,    8,    8,    10]
        dark_pixels   = [794,931; 872,710; 1013,1032; 1191,1179; 1288,1303]
        rot_angle_deg = 1.2; % camera rotoation relative to the mask
        fhwm_to_sigma = 1/2*sqrt(2*log(2));
        
        vis2euv = [72.581;-69.236;-5.3595]; 
        
        
        px = [ 6.420e+05  3.280e+04 -7.651e+03 -3.982e+02  3.153e+01  1.693e+00 -4.944e-02 -3.872e-04 -4.217e-03 -3.175e-07 ];
        py = [-9.038e+05 -2.237e+04  1.096e+04  2.977e+02 -4.386e+01 -1.389e+00  6.430e-02  1.586e-03  4.143e-03  2.601e-08 ];

        
        data_folder = '/Volumes/OSX/Users/bl1132user/SHARP_images/';
        data_folder_win = 'C:\data\';
        data_folder_mac= '/Users/awojdyla/Documents/MATLAB/SHARP/data/';
        data_folder2= '/Users/awojdyla/Documents/MATLAB/SHARP/data/';
        save_folder = '/Users/awojdyla/Documents/MATLAB/SHARP/data/';
        save_folder2= '/Users/awojdyla/Documents/printouts/';
        save_folder_mac= '/Users/awojdyla/Documents/printouts/';
        save_folder_win = 'C:\data\printouts\'
    end
    
    methods(Static)

        function [ img, metaData ] = read( folder, mask_name, series, img_number )
            %SHARP.READ    Read a SHARP image or series and its metadate
            %   [ img, metaData ] = Sharp.read(folder, mask_name, series, img_number)
            %   reads the specified image image; output is a double-2048x2048 array
            %   [ img, metaData ] = Sharp.read(folder, mask_name, series)
            %   reads all the images in the series;
            %   output is a cell array of double-2048x2048 array
            %   
            %   Example : 
            %       folder ='/Users/awojdyla/Documents/MATLAB/SHARP/data/SHARP_2014-08-29/';
            %       [img, metaData] = Sharp.read(folder,'GREMLIN3_AGC',6,1);
            %       img(1:2048,1:2048);
            %       metaData.pixis_exptime;
            %
            %       [img, metaData] = Sharp.read(folder,'GREMLIN3_AGC',6)
            %       img{1}(1:2048,1:2048);
            %       metaData{1}.pixis_exptime;
            %
           
            % This function will load a sharp image and the image data if requested.
            % Inputs
            %      folder -- the absolute path of the data folder on the users machine,
            %      this folder should contain all of the images along with the logfile
            %      mask_name -- string, the name of the mask imaged, generally this is
            %      the begenning of all the image file names
            %      series -- int, series number to be loaded
            %      img_number -- image number in the series, if this input is not
            %      included the function will load all images and metaData will be loaded
            %      for image files with the matching mask name and series number
            % Outputs
            %      img -- cell array containing the image loaded as a matrix, if only one
            %      image is selected the cell is 1x1, if multiple images are selected
            %      then the cell is numberOfImagesx1.
            %      metaData -- the metaData read from the .csv logfile if a log file is
            %      not present then metaData is set to false. If a single image is
            %      selected to load then metaData is a 1x1 cell containing a structure,
            %      if all series images are selected then a NumberOfImagesx1 cell is
            %      returned with each cell containing a structure.
            
            % Authors: A Wojdyla, A Donoghue, CXRO/LBNL
            %          <awojdyla@lbl.gov>
            %
            
            %parsing data
            if nargin~=1
                
                % back compatibility
                if exist(folder,'dir')~=7
                    if ispc
                        folder = strcat(Sharp.data_folder_win,strrep(folder(45:end),'/','\'));
                    elseif ismac
                        folder = strcat(Sharp.data_folder_mac,strrep(folder(45:end),'\','/'));
                    end
                end
                
                if folder(end)~=filesep
                    folder(end+1) = filesep;
                end
                
                if exist(folder,'dir')~=7
                    error('Folder\n"%s"\ndoes not exist.',folder)
                else
                    
                    %FIXME : use a "_" instead !
                    try
                        folder_date = datenum(folder((end-10):(end-1)),'yyyy-mm-dd');
                    catch err
                        switch folder((end-4):(end-1))
                            case 'LBNL'
                                folder_date = datenum(folder((end-15):(end-6)),'yyyy-mm-dd');
                            case 'TSMC'
                                folder_date = datenum(folder((end-15):(end-6)),'yyyy-mm-dd');
                            case 'NTEL'
                                folder_date = datenum(folder((end-16):(end-7)),'yyyy-mm-dd');
                            case 'LOFO'
                                folder_date = datenum(folder((end-16):(end-7)),'yyyy-mm-dd');
                            case 'SUNG'
                                folder_date = datenum(folder((end-18):(end-9)),'yyyy-mm-dd');
                            case 'TECH'
                                folder_date = datenum(folder((end-19):(end-10)),'yyyy-mm-dd');
                            case '_IBM'
                                folder_date = datenum(folder((end-14):(end-5)),'yyyy-mm-dd');
                            otherwise
                                warning('Sharp.read : folder error\n%s\n',folder)
                                rethrow(err);
                        end
                    end
                end
            end
            
            %FIXME add isempties
            switch nargin
                case 1  % if function is called with image number the only look at load one image
                   try
                       if ~isa(folder,'cell') || length(folder)== 1
                           img = double(imread(folder)); %create img to return
                           metaData = Sharp.metadata(img);
                       else
                           img = cell(1,length(folder));
                           metaData = cell(1,length(folder));
                           for i=1:length(folder)
                               img{i} = Sharp.read(folder{i});
                               metaData{i} = Sharp.metadata(img{i});
                           end
                       end
                   catch err
                        display(err.message)
                        display('Image file could not be found')
                        img = false;
                        metaData = false;
                        return
                    end
                case 4  % if function is called with image number the only look at load one image
                    filename = sprintf('%s%s-%s-%04d-%04d.png',...
                        folder, mask_name, datestr(folder_date,'yymmdd'), series, img_number);
                    try
                        img = double(imread(filename)); %create img to return
                        metaData = Sharp.metadata(img);
                    catch err
                        display(err.message)
                        display('Image file could not be found')
                        img = false;
                        metaData = false;
                        return
                    end
                case 3  %if img_number is not give load all the images in the series
                    filename = sprintf('%s%s-%s-%04d-*.png',...  %note wildcard *
                        folder, strtrim(mask_name), datestr(folder_date,'yymmdd'), series);
                    image_files = dir(filename);  %look for all files with correct mask name and series number
                    num_images = size(image_files, 1);
                    if num_images == 0
                    fprintf('Sharp.read :: no data was loaded\nCheck series number\n')
                    end
                    image_file = cell(num_images,1);
                    metaData = cell(num_images,1);
                    img = cell(num_images, 1);
                    
                    for c = 1:num_images
                        image_file{c} = image_files(c).name; %plop filename in cell
                        img{c} = double(imread(strcat(folder, image_file{c})));  %load images in to cell to return.
                        %
                        metaData{c} = Sharp.metadata(img{c});
                        %
                    end
            end
            
        end
        
        function [ metadata ] = metadata( input )
            %SHARP.METADATA	Read the metadata included in a SHARP image
            %   metadata  = Sharp.metadata(image)
            %   reads the metadata included in a SHARP image data
            %   metadata  = readSHARPmetadata('filename')
            %   reads the metadata included in a SHARP image file name
            %
            %   the data is returned as a structure
            %
            %   See also SHARP.READ
            
            %   Feb 2014
            %   A Wojdyla <awojdyla@lbl.gov>
            %   SHARP team, Center for X-Ray Optics, LBNL
            
            % reading the image
            
            if ischar(input)
                img = uint16(imread(input));
            else
                img = input;
            end
            
            % looking for end delimiter ('99999')
            last_idx = find(img(end-1,1:end) == 9, 1);
            % ASCII conversion, with leading delimiter ('11111') removal
            cell_data = textscan(char([img(end,7:end), img(end-1,1:last_idx)]),'%s','delimiter',',');
            % converting the resulting cell into an array of strings
            str_data = char(cell_data{1}(:));
            
            % populating the metadata structure
            metadata = struct;
            
            try
                for i = 1:length(str_data)
                    % separating field names and values
                    tmp = textscan(str_data(i,:),'%s','delimiter','=');
                    tmp_field = (char(tmp{1}(1)));
                    if size(tmp{1},1)==2 % protection against empty field
                        tmp_value = char(tmp{1}(2));
                    else
                        tmp_value = '';
                    end
                    

                    test = strtrim(tmp_field);
                    % populating fields with numerals or strings
                    if isempty(regexpi(tmp_value,'[a-z;]','once'))
                        try
                            metadata.(tmp_field) = str2double(tmp_value);
                        catch error
                            if strcmp(error.identifier,'MATLAB:AddField:InvalidFieldName')
                                fprintf('metadata retrieval error; make sure image is not processed\n')
                                rethrow(error)
                            else
                                rethrow(error)
                            end
                        end
                    else

                        metadata.(test) = tmp_value;
                    end
                    
                end
            catch err
                fprintf('Impossible read meta data.\n ... Make sure you use a recent image and that you have performed no processing on it.\n')
                if isempty(metadata)
                    rethrow(err)
                end
            end
        end
        
        function str_cell= find(varargin)
            % SHARP.FIND Find all the data containt the specified string
            %   str_cell= Sharp.find('str1','str2')
            % 
            % See also SHARP.GLANCE, SHARP.LESS, SHARP.CAT

            dta_folder = Sharp.get_datafolder();
            
            a = dir(dta_folder);
            f = filesep;
            idx = 1;
            str_cell = cell(0,1);
            for i=1:size(a)
                if a(i).isdir
                    %fprintf('%s ',a(i).name)
                    b = dir(strcat(dta_folder,f,a(i).name));
                    for j = 1:size(b)
                        str_name = b(j).name;
                        is_valid = true;
                        for k=1:length(varargin)
                            is_valid = is_valid.*(~isempty(strfind(str_name,varargin{k})));
                        end
                        if is_valid && ~isempty(strfind(str_name,'.png')) ...
                                    && isempty(strfind(str_name,'detail')) ...
                                    && isempty(strfind(str_name,'-b2')) ...
                                    && isempty(strfind(str_name,'-b4'))
                            str_cell{idx} = strcat(dta_folder,f,a(i).name,f,str_name);
                            idx = idx+1;
                            %fprintf('%s\n ',str_name)
                        end
                    end
                end
            end
            if size(str_cell,1)==0
                fprintf('no corresponding image found');
            end
        end
        
        function cat(cell_str)
            % SHARP.CAT Display cell strings in the console
            %   Sharp.cat(cell_str)
            %
            % See also SHARP.FIND, SHARP.LESS, SHARP.GLANCE
            for i=1:length(cell_str)
                fprintf('%s\n',cell_str{i})
            end
        end
        
        function cell_less = less(cell_str)
            % SHARP.LESS Reduce cell strings to one only image per series
            %   cell_less = Sharp.less(cell_str)
            %
            % See also SHARP.FIND, SHARP.CAT, SHARP.GLANCE
            
            mask = false(1,length(cell_str));
            for i=1:length(cell_str)
                if strcmp(cell_str{i}((end-7):end),'0001.png')
                    mask(i) = true;
                end
            end
            cell_less = cell_str(mask);
        end
        
        
        function glance(filename)
            [img, meta] = Sharp.read(filename);
            imagesc(Sharp.ROI(Sharp.remove_bg(img)))
            title(sprintf('NA = %1.2f, illum : %1.0f/%1.0f/%1.0f',meta.zp_na, meta.ma_arg0,meta.ma_arg1,meta.ma_arg2))
            axis image off
            colormap gray

            
        end
        
        function dta_folder = get_datafolder()
            P = mfilename('fullpath');
            isSettings =  exist(strcat(P(1:end-5),'Sharp_settings'),'file');
            if isdir(Sharp.data_folder)
                dta_folder = Sharp.data_folder;
            elseif isdir(Sharp.data_folder2)
                dta_folder = Sharp.data_folder2;
            elseif isSettings
                fid = fopen(strcat(P(1:end-5),'Sharp_settings'),'r');
                dta_folder = char(fread(fid))';
                fclose(fid);
                if ~Sharp.set_datafolder()
                    error('check the existence of the data folder')
                end
            end
        end
        
        function isSet = set_datafolder()
            answer = questdlg('No data folder set. Would you like to set one','Set Dialog');
            isSet = false;
            if strcmp(answer,'Yes')
                file_dir = uigetdir();
                if file_dir~=0
                    fid = fopen(strcat(mfilename('fullpath'),'_settings'),'w');
                    fwrite(fid,file_dir)
                    fclose(fid)
                    isSet = true;
                end
            end
        end


        function [ img_wo_bg , bg ] = remove_bg( img )
            % SHARP.REMOVE_BG     Removes the background of a full-size SHARP image
            %   [img_wo_bg, bg] = Sharp.remove_bg(img)
            %    where img is a full-size  SHARP image or a cell made of SHARP images
            %
            % See also SHARP.REMOVE_DC
            
            if isa(img,'cell')
                
                if size(img{1},1)~=2048 && size(img{1},2)~=2048
                    error('The images size must be 2048x2048')
                end
                
                img_wo_bg = cell(size(img));
                
                for i = 1:max(size(img))
                    bg = sum(sum(img{i}(1537:1792,1:256)))/(256.^2);
                    img_wo_bg{i} = img{i};
                    img_wo_bg{i}(1:1878,:) = abs(img_wo_bg{i}(1:1878,:)-bg);
                end
                
            else
                
                if size(img,1)==2048 && size(img,2)==2048
                    bg = sum(sum(img(1537:1792,1:256)))/(256.^2);
                    img_wo_bg = img;
                    img_wo_bg(1:1878,:) = abs(img_wo_bg(1:1878,:)-bg);
                else
                    error('The provided image do not have the right size.')
                end
                
            end
            
             if nargout==0
                 error('Sharp.remove_bg :: disabled functionnality')
             end
%                 [X,Y] = meshgrid(1:2048);
%                 mask = X>Y;
%                 
%                 if isa(img,'cell')
%                     img_hl = img{1};
%                     img_bg = img_wo_bg{1};
%                 end
%                 
%                 img_hl(~mask) = img_bg(~mask);
%                 image(img_hl./max(img_hl(:))*64);
%                 title('current background removal')
%             end
            
        end


        

        function [ img_roi, x_roi, y_roi ] = ROI( img, roi_size_px, x_d, y_d)
            % SHARP.ROI     Extract the region of interest from a SHARP image
            %   [img_roi, x_roi, y_roi] = Sharp.ROI(img)
            %       uses a (512px)^2 region centered around the ROI
            %   [img_roi, x_roi, y_roi] = Sharp.ROI( img, roi_size_px)
            %       lets you define the size of the ROI
            %   [img_roi, x_roi, y_roi] = Sharp.ROI( img, roi_size_px, x_d, y_d)
            %       lets you define an offset in the x- and y-direction 
            %
            % See also SHARP.GETSWEETSPOT
            if nargin ==1
                roi_size_px = 512;
            end
            if nargin <= 2
                x_d = 0;
                y_d = 0;
            elseif nargin <=3
                error('not enough arguments')
            end
            
            if isa(img,'cell')
                img_roi = cell(size(img));
                img_temp = img{1};
                for i = 1:max(size(img))
                    [x_roi, y_roi] = Sharp.getSweetSpot(img{i}, roi_size_px);
                        img_roi{i} = img{i}(y_roi+y_d(min(i,length(y_d))),x_roi+x_d(min(i,length(x_d))));
                end
            else
                if size(img,1)==2048 && size(img,2)==2048
                    img_temp = img;
                    [x_roi, y_roi] = Sharp.getSweetSpot(img, roi_size_px);
                    if round(x_d) == x_d && round(y_d) == y_d
                        img_roi = img(y_roi+(y_d),x_roi+x_d);
                    else
                        error('offset indices must be integer')
                    end
                        
                else
                    error('The provided image do not have the right size.')
                end
            end
            
            if nargout == 0
                img_hl = img_temp;
                img_hl(y_roi+y_d,x_roi+x_d) = 2*img_temp(y_roi+y_d,x_roi+x_d);
                imagesc(img_hl)
                axis image
                title('current ROI')
            end
            
            x_roi = x_roi+x_d(1);
            y_roi = y_roi+y_d(1);
        end



        function img_rot = rotate(img, angle_deg, opt_arg)
            %SHARP.ROTATE Rotate an image using interpolation
            %   img_rotate = Sharp.rotate(img,angle_deg)
            %   rotates the image by the specified angle, ccw.
            %   the input can be a single image or a cell stack
            %
            %   img_rotate = Sharp.rotate(img,angle_deg,'crop')
            %   does the same but crops the image to remove nan zones
            %
            % See also SHARP.RESIZE, SHARP.CROP2
            
            %batch mode, if the input is an image stack
            if isa(img,'cell')
                img_rot = Sharp.batch(img,sprintf('Sharp.rotate(x,%c)',angle_deg));
            else
                
                if isreal(img)
                    [X1,Y1] = meshgrid(linspace(-0.5,0.5,size(img,1)),...
                        linspace(-0.5,0.5,size(img,2)));
                    rot = @(theta) [cos(theta), sin(theta); -sin(theta) cos(theta)];
                    
                    mat_rot = rot(-angle_deg*pi/180);
                    
                    X2 = reshape(X1(:)*mat_rot(1,1) + Y1(:)*mat_rot(1,2),size(img,1),size(img,2));
                    Y2 = reshape(X1(:)*mat_rot(2,1) + Y1(:)*mat_rot(2,2),size(img,1),size(img,2));
                    
                    img_rot = interp2(X1,Y1,img,X2,Y2,'linear');
                else % recursive calls for amplitude and phase;
                    img_rot = abs(Sharp.rotate(abs(img),angle_deg)).*...
                        exp(1i.*Sharp.rotate(angle(img),angle_deg));
                end
                
                if nargin==3
                    if strcmp(opt_arg,'crop')
                        [~,m1] = min(isnan(img_rot(:,1)));
                        [~,m2] = min(isnan(img_rot(1,:)));
                        img_rot = Sharp.crop2(img_rot,...
                            size(img,1)-m1,size(img,2)-m2);
                    else
                        error(sprintf('Sharp.rotate > unknown argument :"%s"',opt_arg))
                    end
                end
            end
        end

        
        function img_resized = resize(img,x_size_px,y_size_px)
            % RESIZE  Resize an image using interpolation
            %   img_resized = Sharp.resize(img, size_px)
            %   img_resized = Sharp.resize(img, x_size_px, y_size_px)
            if nargin==2 
                if size(img,1)==size(img,2)
                    y_size_px = x_size_px;
                else
                    error('the image should be square if you only use two arguments.')
                end
            end
            
            [X1,Y1] = meshgrid(linspace(-0.5,0.5,size(img,1)),linspace(-0.5,0.5,size(img,2)));
            [X2,Y2] = meshgrid(linspace(-0.5,0.5,  x_size_px),linspace(-0.5,0.5,  y_size_px));
            img_resized = interp2(X1,Y1,img',X2,Y2)';
        end
        
        function img_binned = bin(img,p)
            %SHARP.BIN   binning of an image
            %   img_binned = Sharp.bin2(img, N_bin)
            %
            % Example :
            %   img_binned = bin(img, 2);
            
            % Matt, from MathCentral
            % http://www.mathworks.com/matlabcentral/newsreader/view_thread/248189
            if size(img,1)~=1 && size(img,1)~=1
                error('wrong size, consider bin2 instead')
            end
            
            img_binned=sum( reshape(img,p,[]) ,1 );
        end

         function img_binned = bin2(img,p,q)
            %SHARP.BIN2  2-dimensionnal binning of an image
            %   img_binned = Sharp.bin2(img, x_bin, y_bin)
            %
            % Example : 
            %   img_binned = bin2(img, 2, 3);
            
            % Matt, from MathCentral
            % http://www.mathworks.com/matlabcentral/newsreader/view_thread/248189
            [m,n]=size(img); %M is the original matrix
            
            img=sum( reshape(img,p,[]) ,1 );
            img=reshape(img,m/p,[]).'; %Note transpose
            
            img=sum( reshape(img,q,[]) ,1);
            img_binned=reshape(img,n/q,[]).'; %Note transpose
         end

        
        function out = pad2(x, numrows, numcols)
            %SHARP.PAD2  Pad a matrix with zeros, symmetrically in bothdirections
            %   out = Sharp.pad2(in, newlength)
            %   assumes that the final size in both directions is the same
            %   out = Sharp.pad2(in, numrows, numcols)
            %   pad with different size in both directions
            %
            %   See also SHARP.CROP2
            
            % R Miyakawa
            
            if nargin == 2
                numcols = numrows;
            end
            [sr, sc] = size(x);
            
            if numrows < sr && numcols < sc
                out = crop2(x,numrows, numcols);
                return;
            end
            
            out = zeros(numrows, numcols);
            startr = ceil((numrows - sr)/2);
            startc = ceil((numcols - sc)/2);
            
            out(startr + 1: startr + sr, startc + 1: startc + sc) = x;
        end
        
        
        
        function out = crop2(in, lenr, lenc)
            %SHARP.CROP2 Crop a matrix, symetrically in both directions
            %   out = Sharp.crop2(in, length)
            %   out = Sharp.crop2(in, length, width)
            %
            %   See also SHARP.PAD2
            
            % R Miyakawa
            
            if isa(in,'cell')
                out = Sharp.batch(in,sprintf('Sharp.crop2(x,%d)',lenr));
            else
            
            
            [sr,sc] = size(in);
            
            if nargin<3
                lenc = lenr;
            end
            
            if lenr > sr && lenc > sc
                out = pad2(in, lenr, lenc);
                return;
            end
            
            if nargin == 2
                lenc = lenr;
            end
            
            wr = lenr/2;
            wc = lenc/2;
            out = zeros(lenr, lenc);
            out(1:lenr, 1:lenc) = ...
                in( floor(sr/2-wr+1):floor(sr/2+wr), floor(sc/2 - wc+1):floor(sc/2+wc));
            end
        end


        function image_shifted = circshift2(input, x_shift_px, y_shift_px)
           % SHARP.CIRCSHIFT2 Image shift using circular permutation
            %   img_shifted = Sharp.circshift2(img, x_shift_px, y_shift_px)
            %   shifts the image in x- and y-direction, to the nearest pixel
            % 
            % See also SHARP.SPSHIFT2
            image_shifted = circshift(circshift(input',round(x_shift_px))',round(y_shift_px));
        end
        

        function img_shifted = spshift2( img,x_shift_px, y_shift_px )
            %SHARP.SPSHIFT Sub-pixel image shift using spectral phase
            %   img_shifted = Sharp.spshift2( img,x_shift_px, y_shift_px )
            %       shifts the image in x- and y-direction with fractional pixel
            % 
            % See also SHARP.CIRCSHIFT2
            
            if size(x_shift_px,1) ~= 1 || size(x_shift_px,2) ~= 1 ...
            || size(y_shift_px,1) ~= 1 || size(y_shift_px,2) ~= 1 
                error('Sparp.spshift :: the shift must be a scalar')
            elseif isempty(x_shift_px) || isempty(x_shift_px)
                error('Sparp.spshift :: empty shift')
            end
            
            [FX,FY] = meshgrid(...
                (ceil(-size(img,2)/2):ceil(-1+size(img,2)/2))/size(img,2),...
                (ceil(-size(img,1)/2):ceil(-1+size(img,1)/2))/size(img,1));
            
            if nargin==2
                y_shift_px = 0;
            end
            
            
            %x_shift_px = round(x_shift_px);
            %y_shift_px = round(y_shift_px);
            
            
            img_shifted_re = real((fftshift(ifft2(ifftshift( ...
                fftshift(fft2(ifftshift(real(img)))).*...
                exp(-1i*2*pi*(FX*x_shift_px+FY*y_shift_px)))...
                ))));
            
            img_shifted_im = real((fftshift(ifft2(ifftshift( ...
                fftshift(fft2(ifftshift(imag(img)))).*...
                exp(-1i*2*pi*(FX*x_shift_px+FY*y_shift_px)))...
                ))));
            
            if isreal(img)
                img_shifted = img_shifted_re;
            elseif isreal(1i*img)
                img_shifted = 1i*img_shifted_im;
            else
                img_shifted = img_shifted_re+1i*img_shifted_im;

            end
        end        
        
        
        function [ img_lowpass ] = lowpass( img, radius )
            %LOWPASS Applies a gaussian lowpass filter to an image
            %   img_lowpass = lowpass2( img, radius )
            %       performs a Gaussian lowpassing with specified radius
            
            % A Wojdyla (awojdyla@lbl.gov) April 2014
            
            if nargin == 1
                radius = 10;
            end
            
            [XX,YY] = meshgrid(...
                (1:size(img,1))-floor(size(img,1)/2),...
                (1:size(img,2))-floor(size(img,2)/2));
            RR = sqrt(XX.^2+YY.^2);
            GAUSSIAN_KERNEL = fftshift(fft2(ifftshift( exp(-(RR/(2*radius)).^2))));
            
            if isa(img,'cell')
                img_lowpass = cell(size(img));
                for i=1:length(img_lowpass)
                    IMG =             fftshift(fft2(ifftshift( img{i} ) ) );
                    img_temp = real(ifftshift(ifft2(fftshift( IMG.*GAUSSIAN_KERNEL))));
                    img_lowpass{i} = img_temp*norm(img{i}(:))/norm(img_temp);
                end

            else
                IMG =             fftshift(fft2(ifftshift( img ) ) );
                img_temp = real(ifftshift(ifft2(fftshift( IMG.*GAUSSIAN_KERNEL))));
                img_lowpass = img_temp*norm(img(:))/norm(img_temp);
            end

        end

        function img_out = filter(img_in)
            % FILTER Remove the noise that lies outside the angular
            % spectrum
            % img_out = Sharp.filter(img_in) works for .33 4xNA data
            %
            % See also SHARP.HOMOGENIZE
            [FX,FY] = meshgrid(linspace(-2.703,2.703,size(img_in,2)),...
                               linspace(-2.703,2.703,size(img_in,1)));
            FILTER  = double(FX.^2+FY.^2<1);
            img_out = real(Sharp.ift(Sharp.ft(img_in).*FILTER));
        end
        
        function [ img_out ] = detrend2( img_in )
            % SHARP.DETREND2 Remove a 2nd order 2D polynomial trend in an image
            %   [ img_out ] = Sharp.detrend2(img_in)
            
            
            [X,Y]=meshgrid(1:size(img_in,2),1:size(img_in,1));
            mask = img_in>0.5*max(img_in(:));
            
            x = X(mask);
            y = Y(mask);
            z = img_in(mask) ;
            
            C = ones(length(x),1) ;
            V = [x.*y y.^2 y C x x.^2] ;
            
            a = V \ z ;
            poly_xy = @(X,Y) a(1)*X.*Y+a(2)*Y.^2+a(3)*Y+a(4)+a(5).*X+a(6)*X.^2;
            
            %img_out = img_in./poly_xy(X,Y).*double((poly_xy(X,Y)>100));
            img_out = img_in./poly_xy(X,Y).*(1+sign(poly_xy(X,Y)-500))*0.5.*max(img_in(:));
            warning('detrending still under testing')
        end


        
        
        
%         %FIXME
%         function img_uniform = homogenize2(img_in,radius)
%             % HOMOGENIZE    Homogenize an image (removing low variations)
%             %   img_hom = Sharp.homogenize(img, radius)
%             %   homogenizes the image using the specufied radius
%             %
%             % See also SHARP.DENOISE
%             
%             img_lowpass = lowpass2(img_in,radius); %FIXME arbitrary
%             img_lowpass = img_lowpass - min(img_lowpass(:));
%             %img_lowpass = img_lowpass/max(img_lowpass(:));
%             
%             img_uniform = img_in-img_lowpass;
%             clf
%             imagesc(img_lowpass);
%             imagesc(img_uniform);
%             axis image
%         end


        
        %TODO : automate on series
        function [ img_appended ] = tile(varargin)
            % TILE  Appends images horizontally
            %   img_tiled = Sharp.tile(img1,img2,img3,...)
            %   creates a tiled image of all the input images
            
            if nargin == 1 % do nothing
                img_appended = varargin{1};
                
            elseif nargin == 2 % main part
                img1 = varargin{1};
                img2 = varargin{2};
                % resisizing the initial image if needed
                if size(img1,1)<size(img2,1)
                    img_temp = img1;
                    img1 = zeros(size(img2,1), size(img1,2));
                    img1(1:size(img_temp,1),1:size(img_temp,2)) = img_temp;
                end
                img_appended = zeros(size(img1,1), size(img1,2)+size(img2,2));
                img_appended(1:size(img1,1), 1:size(img1,2)) = img1;
                img_appended(1:size(img2,1), (size(img1,2)+1):end) = img2;
            else % recursive
                img_appended = append2(varargin{1},append2(varargin{2:end}));
            end
        end
        
        function tile_q = tile_cell(img_cells)
        % SHARP.TILE_CELL     Transform a cell array into an image
        %   tile_q = Sharp.tile_cell(img_cells)
        %
        % See also : SHARP.CELL_STITCH ,SHARP.TILE
            tile_q = img_cells{1};
            for q=2:length(img_cells)
                tile_q = Sharp.tile(tile_q,img_cells{q});
            end
        end

        
        function stack_q = stack_cell(img_cells)
            % SHARP.STACK_CELL 
            %   tile_q = Sharp.stack_cell(img_cells)
            %
            % imagesc(Sharp.stack_cell(Sharp.ROI(Sharp.read(Sharp.less((Sharp.find('IMO14')))),300)));
            %
            % See also : SHARP.CELL_STITCH ,SHARP.TILE_CELL
            N_r = floor(sqrt(length(img_cells)));
            img_stack = cell(N_r,N_r+1);
            for i= 1:N_r
                for j=1:(N_r+1)
                    if (i-1)*N_r+j<length(img_cells)
                        img_stack{i,j} = img_cells{(i-1)*N_r+j};
                    end
                end
                stack_q = Sharp.cell_stitch(img_stack);
            end
        end
        
        function cell_out = batch(cell_in, instruction)
            %SHARP.BATCH Batch processing for cells of image
            %   cell_out = batch(cell_in, instruction)
            %   applies the 'instruction' to the cells considered
            %   the instruction is a function handle
            %                   or a string to be evaluated
            %                   (e.g. 'abs(x).^2')
            %
            % See also Sharp.mat2cell, Sharp.tile_cell
            
            if min(size(cell_in))~=1
                error('Sharp.batch : cell arrays must be 1xN. See Sharp.batch2')
            end
            cell_out = cell(size(cell_in));
            if isa(instruction,'function_handle')
                for i=1:length(cell_in)
                    cell_out{i} = instruction(cell_in{i});
                end
            else
                for i=1:length(cell_in)
                    x = cell_in{i};
                    cell_out{i} = eval(instruction);
                end
            end
        end
        
        function cell_out = batch2(cell_in, instruction)
            %FIXME
            %SHARP.BATCH2 Batch processing for cells of image
            %   cell_out = batch(cell_in, instruction)
            %   applies the 'instruction' to the cells considered
            %   the instruction is a function handle
            %                   or a string to be evaluated
            %                   (e.g. 'abs(x).^2')
            %
            % See also Sharp.mat2cell, Sharp.tile_cell
            
            cell_out = cell(size(cell_in));
            if isa(instruction,'function_handle')
                for i=1:size(cell_in,1)
                    for j=1:size(cell_in,2)
                        cell_out{i,j} = instruction(cell_in{i,j});
                    end
                end
            else
                for i=1:size(cell_in,1)
                    for j=1:size(cell_in,2)
                        x = cell_in{i,j};
                        cell_out{i,j} = eval(instruction);
                    end
                end
            end
        end
        

        %FIXME : fail grace typechecking
        function img_as_mat = cell2mat(img_as_cell)
            % CELL2MAT Transforms a cell-based stack of images into a 3D matrix
            %   img_as_mat = Sharp.cell2mat(img_as_cell)
            %
            % See also SHARP.MAT2CELL
            
            N_img = length(img_as_cell);
            img_as_mat = zeros(size(img_as_cell{1},1),...
                               size(img_as_cell{1},2),...
                               N_img);
           for i=1:N_img
               img_as_mat(:,:,i) = img_as_cell{i};
           end
        end
        

        %FIXME : fail grace typechecking
        function img_as_cell = mat2cell(img_as_mat)
            % MAT2CELL Transforms a 3D matric into a cell-based stack of images
            %   img_as_cell = Sharp.mat2cell(img_as_mat)
            %
            % See also SHARP.CELL2MAT
            
            N_img = size(img_as_mat,3);
            img_as_cell = cell(N_img,1);
            for i=1:N_img
                img_as_cell{i} = img_as_mat(:,:,i);
            end
        end
        

        %FIXME non square images, inline cells
        function img_full = cell_stitch(img_sub)
            % CELL_STITCH Stitches many 2D cells to form a complete 2D image
            %   img_full = Sharp.cell_stitch(img_sub)
            
            %warning('function probably broken')
            
            if ~isempty(img_sub{1,1})
                N_r = size(img_sub,1);
                N_c = size(img_sub,2);
                roi_size_px = size(img_sub{1,1},1);
            else
                roi_size_px = 0;
                for i=1:size(img_sub,1)*size(img_sub,1)
                    roi_size_px = max(roi_size_px, size(img_sub{i},1));
                end
            end
            
            
            if ~(size(img_sub{1,1},1)==size(img_sub{1,1},2))
                error('Sharp.cell_stitch :: sub-images must be square')
            end
            
            img_full = zeros(N_r*roi_size_px,N_c*roi_size_px,size(img_sub{1,1},3));

            for r = 1:N_r
                for c = 1:N_c
                    if ~isempty(img_sub{r,c})
                    img_full(((r-1)*roi_size_px+1):r*roi_size_px,...
                             ((c-1)*roi_size_px+1):c*roi_size_px,:) ...
                             = img_sub{r,c};
                    end
                end
            end
            
        end
        
        %% Image stats

        function img_sum = sum_img(img)
            N_img = length(img);
            img_sum = zeros(size(img{1}));
            for i = 1:N_img
                img_sum = img_sum +img{i}/N_img;
            end
        end

        function rms = rms_diff(img_ref,img_comp)
            rms = sum(abs(img_ref(:)-img_comp(:)).^2)...
                 /sum(abs(img_ref(:)).^2);
        end
        
        function rms = nrms_diff(img_ref,img_comp)
            rms = sqrt(sum((img_ref(:)-img_comp(:)).^2)) ...
                ./sqrt(sum( img_ref(:)             .^2));
        end

        function g = gaussian(x_px,mean_px,fwhm_px)
        % GAUSSIAN Generates a gaussian function
        %   g = gaussian(x_px) generates a gaussian function with zero mean
        %       and 1 unit full-width at half-maximum
        %   g = gaussian(x_px,mean_px,fwhm_px)
        
            if nargin == 1
                mean_px = 0;
                fwhm_px = 1;
            end

            sigma_x = fwhm_px/2*sqrt(2*log(2));
            g = exp(-((x_px-mean_px)/(sqrt(2)*sigma_x)).^2);
        end

        %% visualize parameters


        function mask = mc_mask()
            % MC_MASK Mask corresponding to the MC shadow
            %   mask = Sharp.mc_mask()
            [X,Y] = meshgrid(1:2048);
            x_c = 941;
            y_c = 850;
            y_t = 1157;
            r_c = 883;
            
            mask = (Y-y_t)<0 | (X-x_c).^2+(Y-y_c).^2<r_c.^2;
        end


        %FIXME : for cells
        %TODO refactor a function that operates on metadata
        function [sx, sy] = monopole_sigmaxy(data)
            % MONOPOLE_SIGMAXY Get the position of the monopole in
            %   [sx, sy] = monopole_sigmaxy(img);
            %   [sx, sy] = monopole_sigmaxy(meta);
            %   sives the position of the monople in lens units
            %   with cartesian coordinates
            if ~isa(data,'struct')
                data = Sharp.metadata(data);
            end
            sx = data.ma_arg0.*cos(data.ma_arg1*pi/180);
            sy = data.ma_arg0.*sin(data.ma_arg1*pi/180);
        end


        
        %FIXME : check parameters are ok
        %TODO  : mimick from input image
        function [ sx, sy ] = pupil_fill( type, arg0, arg1, arg2, N_pts )
            %SHARP.PUPIL_FILL Pupil fill following SHARP conventions
            % [sx,sy] = Sharp.pupil_fill( type, arg0, arg1, arg2, N_pts )
            %   where sx and sy are a 1D cells containing the fill pos in NA units
            %   type 1,2,4 : monopole/dipole/quadrupole
            %   type 5,6 :   quasars
            %   type 12-15 : crosspoles
            %   arg0 : outer radius
            %   arg1 : inner radius (if any)
            %   N_pts : number of points for the output (approx)
            %
            % See also : Sharp.view_fill

            R1 = arg0;
            R2 = arg1;
            
            switch type
                %monopole, dipole, quadrupole
                case {1,2,4}
                    theta_deg = arg2;
                    sx1 = round(R1.*cos(theta_deg*pi/180)*100)/100;
                    sy1 = round(R1.*sin(theta_deg*pi/180)*100)/100;
                    sx = cell(1,type);
                    sy = cell(1,type);
                    sx{1} = sx1;
                    sy{1} = sy1;
                    if type >= 2
                        sx{2} = -sx1;
                        sy{2} = -sy1;
                        if type == 4
                            sx{3} = sy1;
                            sy{3} =-sx1;
                            sx{4} =-sy1;
                            sy{4} = sx1;
                        end
                    end
                    
                    % denser fills
                case {3,5,6,12,13,14,15}
                    dR = (R1-R2);
                    N_r = round(dR*10);
                    dr = dR/N_r;
                    
                    % total area
                    A = pi*(R1.^2-R2.^2);
                    dtheta = 0;
                    switch type
                        % disk/annular
                        case 3
                            phi = 2*pi;
                            % quasar/crosspole
                        case {5,6,12,13,14,15}
                            phi = arg2*pi/180;
                    end
                    
                    switch type
                        case 3
                            subdiv = 1;
                        case {5,6}
                            subdiv = 4;
                        case {12,13,14,15}
                            subdiv = 2;
                    end
                    
                    switch type
                        case 3
                            phi_0 = 0;
                        case {5,12}
                            phi_0 = 0;
                        case {6,13}
                            phi_0 = pi/4;
                        case {14}
                            phi_0 = pi/2;
                        case {15}
                            phi_0 = 3*pi/4;
                    end
                    
                    
                    clear sx sy
                    w = zeros(1,N_r);
                    np = zeros(1,N_r);
                    idx = 1;
                    %computing weights
                    for i = 1:(N_r);
                        w(i) = pi*((R2+i*dr).^2-(R2+(i-1)*dr).^2)/A;
                        np(i) = ceil(w(i)*N_pts/subdiv);
                        thetas = linspace(-(phi)/2+phi_0,(phi)/2+phi_0,np(i));
                        for j = 1:np(i)
                            for k=1:subdiv
                                sx{idx} = (R2+(i-1)*dr).*cos(thetas(j)+2*pi*(k-1)/subdiv+dtheta);
                                sy{idx} = (R2+(i-1)*dr).*sin(thetas(j)+2*pi*(k-1)/subdiv+dtheta);
                                idx = idx+1;
                            end
                        end
                        if type == 3
                            dtheta = dtheta + pi/np(i);
                        end
                    end
                case default
                    error('Sharp.fill : wrong type')
            end
        end
        

        
        %FIXME : better cell/array handling
        function img = view_fill(sx, sy )
            %SHARP.VIEW_FILL Visualize pupil fills (paris of 1D cells)
            %   img = view_fill( sx,sy );
            
            [XX,YY] = meshgrid(1:512);
            xc = 256;
            yc = 256;
            R = 256;
            
            if ~isa(sx,'cell')
                sx_temp = sx;
                sy_temp = sy;
                sx = cell(size(sx_temp));
                sy = cell(size(sy_temp));
                for i=1:length(sx_temp)
                    sx{i} = sx_temp(i);
                    sy{i} = sy_temp(i);
                end
                
            end
            
            mini = zeros(512,512);
            mask_circle = (XX-xc).^2 + (YY-yc).^2 < R.^2;
            mini(mask_circle) = 0.5;
            for i=1:length(sx)
                if ~isempty(sx{i}) && ~isempty(sy{i}) && abs(sx{i})<=1 && abs(sy{i})<=1
                    X0 = (sx{i}*256);
                    Y0 = (-sy{i}*256);
                    
                    mask_pupil = (XX-xc-X0).^2 + (YY-yc-Y0).^2 < 5.^2;
                    mini(mask_pupil) = 1;
                else
                    warning('Sharp.viewfill : there are points outside the pupil !')
                end
            end
            
            img = mini;
            imagesc(mini);
            axis image off;
        end


        % FIXME : phase out for view fill
        function img = pupil_show(sx,sy)
        %SHARP.PUPIL_SHOW Colordisplay for MA sigma
        %
        %
        % See also SHARP.VIEWFILL

            [XX,YY] = meshgrid(1:512);
            xc = 256;
            yc = 256;
            R = 256;
            X0 = (sx*256);
            Y0 = (sy*256);
            mini = zeros(512,512);
            mini2 = zeros(512,512);
            mini3 = zeros(512,512,3);
            mask = (XX-xc).^2 + (YY-yc).^2 < R.^2;
            mask2 = (XX-xc-X0).^2 + (YY-yc-Y0).^2 < 15.^2;
            mini(mask) = 0.5;
            mini2(mask2) = 1;
            mini3(:,:,1) = mini;
            mini3(:,:,2) = mini;
            mini3(:,:,3) = mini;
    
            mini3(:,:,1)= mini3(:,:,1)+mini2.*(1-mini);
            
            if nargout == 0
                image(mini3);
                axis off image
            else
                img = mini3;
            end
        end
        
        function [axis, ord] = csx()
            % SHARP.CSS Perform a cross-section on the current image
            %   Sharp.csx()
            %   [axis, ord] = csx()
            %
            % See also Sharp.cross_section
            
            % A Wojdyla, awojdyla@lbl.gov
            % August 2015
            
            fprintf('please select two points in the image to perform a cross-section\n')
            data = get(gca,'Children');
            a = data.CData;
            
            [x,y] = ginput(2);
            [axis, ord] = Sharp.cross_section(a,x(1),y(1),x(2),y(2));
            if nargout==0
                h = figure;
                fcn = @(a,b) close(h);
                set(h,'WindowButtonUpFcn',fcn);
                plot(axis, ord);
            end
        end
        

        %% SHARP data specific formatting


        function v_offsets = offaxis_correction(var1, mag, dz_m, cra_deg)
            %SHARP.OFFAXIS_CORRECTION pixel correction for through focus series
            %   v_offsets = Sharp.offaxis_correction(N_img, mag, dz, cra)
            %   v_offsets = Sharp.offaxis_correction(imgs_cell)
            %   uses the meta-information in the image to perform
            %   computation
            %
            % See also SHARP.FORMAT_TF
            
            % off-axis experimental shift for .33NA 4x Lens @dz=500nm
            
            p_tfd = 0.103846; %empirical value, from internal study
            if nargin ==1
                images_RAW = var1;
                meta = cell(size(images_RAW));
                for i=1:length(images_RAW)
                    meta{i} = Sharp.metadata(images_RAW{i});
                end
                cx = 900/meta{1}.image_mag;
                cz = meta{1}.series_dz/0.5e-3;
                cr = meta{1}.ma_cra/6;
                N_img = meta{1}.series_nsteps;
            elseif nargin == 4
                cx = 900/mag;
                cz = dz_m/500e-9;
                cr = cra_deg/6;
                N_img = var1;
            else
                error('offaxis')
            end
            
            v_offsets = (1:N_img)*500/(15)*p_tfd*cz*cr/cx;
            %centering the offset
            v_offsets = v_offsets-v_offsets((N_img+mod(N_img,2))/2);
        end

        function [img_formatted, dz_m]  = format_tf(img_in,roi_size_px,x_decenter,y_decenter,opt_arg)
            %FORMAT_TF Format SHARP through-focus data (bg removal, rotation, axis-correction)
            %   img_cell = Sharp.format_tf(img_cell, roi_size)
            %       format a through focus series with :
            %       - background subration
            %       - correction of camera clocking
            %       - compensation of focus vertical shift
            %       centered on the region of interest of size roi_size_px
            %   img_cell = Sharp.format_tf(img_cell, roi_size, x_decenter, y_decenter)
            %       does the same, but allows for some adjustment in the centering
            %   [img_cell, dz_m] = Sharp.format_tf(img_cell, roi_size, x_decenter, y_decenter,'fine')
            %       uses image registration to refine the focal step
            %
            % See also Sharp.offaxis_correction
            if nargin==1
                x_decenter = 0;
                y_decenter = 0;
                roi_size_px = 512;
            elseif nargin==2
                x_decenter = 0;
                y_decenter = 0;
            end
            
            if ~isa(img_in,'cell')
                is_array = true;
                img_in = {img_in};
            else
                is_array = false;
            end
            
            xc = x_decenter;
            yc = y_decenter;
            [~,x_roi,y_roi]   = Sharp.ROI(img_in{1},roi_size_px,xc,yc);
            [~,x_roi2,y_roi2] = Sharp.ROI(img_in{1},roi_size_px+30,xc,yc);
            offsets = Sharp.offaxis_correction(img_in);
            if nargin == 5 
                if  strcmp(opt_arg,'fine') && length(img_in)>1
                    dy_px = zeros(1,length(img_in));
                    for i= 2:length(img_in)
                        [~,dy_px(i)] = Sharp.register(img_in{i-1}(y_roi,x_roi),img_in{i}(y_roi,x_roi));
                    end
                    dz_m= 10*(cumsum(dy_px)*15e-9);
                    offsets = offsets+cumsum(dy_px);
                    warning('Sharp.format_tf > "fine" feature is experimental')
                else
                    error('Sharp.format_tf > invalid argument : "%s"',opt_arg)
                end
                
            end
            
            
            img_formatted = cell(size(img_in));
            for idx = 1:length(img_in)
                [~, bg] = Sharp.remove_bg(img_in{idx});
                img_formatted{idx} =  Sharp.crop2(...
                    Sharp.rotate(img_in{idx}(round(y_roi2+offsets(idx)),x_roi2),Sharp.rot_angle_deg),...
                                                  length(x_roi),length(y_roi));
                img_formatted{idx} = abs(img_formatted{idx}-bg);
                % imagesc(img{idx})
                % axis image off
                % drawnow
            end
            
            if is_array
                img_formatted = img_formatted{1};
            end
        end
        
        function [img_formatted, dz_m]  = format_fp(img_in,roi_size_px,x_decenter,y_decenter,opt_arg)

            if nargin==1
                x_decenter = 0;
                y_decenter = 0;
                roi_size_px = 512;
            elseif nargin==2
                x_decenter = 0;
                y_decenter = 0;
            end
            
            if ~isa(img_in,'cell')
                is_array = true;
                img_in = {img_in};
            else
                is_array = false;
            end
            
            xc = x_decenter;
            yc = y_decenter;
            [~,x_roi,y_roi]   = Sharp.ROI(img_in{1},roi_size_px,xc,yc);
            [~,x_roi2,y_roi2] = Sharp.ROI(img_in{1},roi_size_px+30,xc,yc);
            
            img_formatted = cell(size(img_in));
            for idx = 1:length(img_in)
                [~, bg] = Sharp.remove_bg(img_in{idx});
                img_formatted{idx} =  Sharp.crop2(...
                    Sharp.rotate(img_in{idx}(round(y_roi2),x_roi2),Sharp.rot_angle_deg),...
                                                  length(x_roi),length(y_roi));
                img_formatted{idx} = abs(img_formatted{idx}-bg);
                % imagesc(img{idx})
                % axis image off
                % drawnow
            end
            
            if is_array
                img_formatted = img_formatted{1};
            end
        end
        
        function img_all = FPM_showall(img_cell,roi_size_px)
            if nargin<2
                roi_size_px = 512;
            end
            
            for idx = 1:length(img_cell)/2
                [sx,sy] = Sharp.monopole_sigmaxy(img_cell{idx*2});
                j = round((sx+1)*10)/2;
                i = round((sy+1)*10)/2;
                real_all{i,j} = (Sharp.crop2((((Sharp.ROI(img_cell{idx*2})))),roi_size_px));
                %spec_all{i,j} = (Sharp.crop2(abs(Sharp.remove_dc(Sharp.ft(Sharp.ROI(img{idx*2})))).^0.1,200));
                axis image off
                drawnow
            end
            img_all = Sharp.cell_stitch(real_all);
        end

        function img_fills = FPM_fulldelta(img, sx, sy, object_guess, lens_guess,Dx,fc_lens )
            % FPM_FULLDELTA Discrepancy analysis for Fourier Ptychography
            % See also Sharper.FPM_fulldelta
            img_fills = Sharper.FPM_fulldelta(img, sx, sy, object_guess, lens_guess,Dx,fc_lens);
        end
        
        function img_DPC = DPC_from_FP(img,sx)
            % DPC_from_FP DPC image from FP data
            %   See also Sharper.DPC_from_FP
            img_DPC = DPC_from_FP(img,sx);
        end

        
        %FIXME : for smaller series
        function [ imgs_out ] = focusStitch( imgs_in )
            % FOCUSSTITCH Stitch Through-focus images to get constant focus
            %   See also Sharper.FocusStitch
            [ imgs_out ] = Sharper.focusStitch( imgs_in );
        end
        
        %FIXME : check
        function [ array_out ] = circsum( img_in )
            % CIRCSUM Circular sum
            %  array_out = circsum(img_in)
            %
            % See also Sharp.FRC
            
            if size(img_in,1) ~= size(img_in,2)
                error('the input matrix must be square')
            end
            
            N_px = size(img_in,1);
            
            [X,Y] = meshgrid(1:N_px);
            xc = floor(N_px/2)+1;
            yc = floor(N_px/2)+1;
            
            dcirc = zeros(1,floor(N_px/2));
            for i=1:floor(N_px/2)
                domain = ((X-xc).^2+(Y-yc).^2)>=(i-1)^2 & ((X-xc).^2+(Y-yc).^2)<(i)^2;
                dcirc(i) = sum(sum(domain.*(img_in)));
%                 fprintf('%d ',i)
%                 if mod(i, 30)==0
%                     fprintf('\n')
%                 end
            end
            
            %array_out = cumsum(dcirc);
            array_out = dcirc;
            
        end
        

        function [ freq_axis ] = fs(real_axis)
            %SHARP.FS 1D or 2D Frequency scale, zero-centered in inverse spatial units
            %   f = fs(t) creates a frequency scale that matches
            %     the zero centered Fourier transform of a signal
            %   freq = Sharp.fs(time)
            %
            % See also SHARP.FT, SHARP.IFT
            
            fs = 1/(real_axis(2)-real_axis(1));
            Nfft = length(real_axis);
            
            df = fs/Nfft;
            freq_axis = (0:df:(fs-df)) - (fs-mod(Nfft,2)*df)/2;
        end

        
        function SIGNAL = ft(signal)
            %SHARP.FT 1D or 2D zero-centered Fourier Transform
            %   SHARP.FT Performs a Fourier transform using optics convention
            %   SIGNAL = Sharp.ft(signal)
            %
            %   See also SHARP.IFT, SHARP.FS
            
            if size(signal,1) == 1 || size(signal,2) == 1
                SIGNAL = fftshift( ifft( ifftshift( signal ) ) );
            else %perform a 2D fourier Transform
                SIGNAL = fftshift( ifft2( ifftshift( signal ) ) );
                if size(signal,1) ~= size(signal,2)
%                    warning('ft:A 2D FT has been performed by default')
                end
            end
        end

        
        function signal = ift(SIGNAL)
            %SHARP.IFT 1D or 2D zero-centered Inverse Fourier Transform
            %   SHARP.IFT Performs a Inv Fourier transform using optics convention
            %   signal = ift(SIGNAL)
            %   See also SHARP.FT, SHARP.FS
            
            % A Wojdyla, Jan 2014
            
            if size(SIGNAL,1) == 1 || size(SIGNAL,2) == 1
                signal = fftshift( fft( ifftshift( SIGNAL ) ) );
            else %perform a 2D fourier Transform
                signal = fftshift( fft2( ifftshift( SIGNAL ) ) );
                if size(SIGNAL,1) ~= size(SIGNAL,2)
                    warning('ift:A 2D IFT has been performed by default')
                end
            end
        end
        
        function [ img_out ] = remove_dc( img_in )
            % SHARP.REMOVE_DC Removes the DC componenta of an image (for better display)
            %   [ img_out ] = remove_dc(img_in)
            %
            % See also SHARP.REMOVE_BG
            img_size = size(img_in,1);
            [X,Y] = meshgrid(1:img_size);
            mask = (X == round(1+img_size/2)) | (Y == round(1+img_size/2));
            img_out = img_in;
            img_out(mask) = 0;mean(img_in(:));
        end
        
        function [ img_out ] = remove_zero( img_in )
            % SHARP.REMOVE_DC Removes the DC componenta of an image (for better display)
            %   [ img_out ] = remove_dc(img_in)
            %
            % See also SHARP.REMOVE_BG
            img_size = size(img_in,1);
            [X,Y] = meshgrid(1:img_size);
            mask = (X == round(1+img_size/2)) & (Y == round(1+img_size/2));
            img_out = img_in;
            img_out(mask) = 0;mean(img_in(:));
        end
        

        function s_out = sgolay(s_in)
            % SHARP.SGOLAY Savitzky-Golay filtering (5pt window, 3rd order)
            %   s_out = Sharp.sgolay(s_in)
            %   (implementation for missing Signal Processing toolbox)
            %
            % See also 
            s_out = s_in;
            s_out(3:end-2) = 1/35*(...
                 -3*(s_in(1:end-4)+s_in(5:end))...
                +12*(s_in(2:end-3)+s_in(4:end-1))...
                +17* s_out(3:end-2));
        end
        
        function s_out = sdiff(s_in)
            %SHARP.SDIFF Savitzky-Golay differential (5pt window, 3rd order)
            %   s_out = Sharp.sdiff(s_in)
            %   (implementation for missing Signal Processing toolbox)
            s_out = s_in;
            s_out(3:end-2) = 1/12*(...
                 1*(s_in(1:end-4)+s_in(5:end))...
                -8*(s_in(2:end-3)+s_in(4:end-1))...
                +0* s_out(3:end-2));
        end
        
        function s_out = sdiff2(s_in)
            %SHARP.SGOLAY2 Savitzky-Golay differential (5pt window, 3rd order)
            s_out = s_in;
            s_out(3:end-2) = 1/7*(...
                 2*(s_in(1:end-4)+s_in(5:end))...
                -1*(s_in(2:end-3)+s_in(4:end-1))...
                -2* s_out(3:end-2));
        end

        % TODO : better tests
        function pitch_m = count_pitch(img)
            %SHARP.COUNT_PITCH determination of the pitch based on an image
            % Only works for vertical lines
            dx = 15e-9;
            
            if isa(img,'cell')
                img = img{1};
            end
            if size(img,1) == 2048
                meta = Sharp.metadata(img);
                dx   =  meta.image_umperpx*1e-6;
                img  = Sharp.crop2(Sharp.rotate(Sharp.ROI(img,600),Sharp.rot_angle_deg),512);
            end
            
            IMG = fft2(img);
            [~, imax] = max(IMG(1,2:end/2));
            pitch_m = dx*size(img,2)/(imax + 1);
        end


        function [frc_array, halfbit_threshold]  = frc(img1, img2)
            % FRC Fourier Ring Coefficients between two images
            %   frc_coeffs = Sharp.frc(img1, img2)
            %       computes the FRC coefficients between two images
            %   [frc_array, halfbit_threshold] = Sharp.frc(img1,img2)
            %
            % See Nieuwenhuizen et al. Nature Methods 10, 6 (2013)
            %  http://doi.org/10.1038/nmeth.2448
            frc_array = Sharp.circsum(Sharp.ft(img1).*conj(Sharp.ft(img2)))./...
                ((sqrt(Sharp.circsum(abs(Sharp.ft(img1)).^2)).*sqrt(Sharp.circsum(abs(Sharp.ft(img2)).^2))));
            
            n_vox = Sharp.circsum(ones(size(img1)));
            halfbit_threshold = (0.2071+1.9102./sqrt(n_vox))./(1.2071+0.9102./sqrt(n_vox));
        end
        


        %TODO implement
        function [argout1, argout2] = cross_section(img_data,x1,y1,x2,y2)
            % CROSS_SECTION Computes the cross-section between two points
            %   [data] = Sharp.cross_section(img,x1,y1,x2,y2)
            %   [px_scale,data] = Sharp.cross_section(img,x1,y1,x2,y2)
            
            if x1==x2 && y1==y2
                error('the two datapoints should be distinct')
            end
            
            data = img_data;
            % angle of the cross-cut
            theta_rad = atan((y2-y1)./(x2-x1));
            N_pts = sqrt((x2-x1).^2 + (y2-y1).^2);
            pts_px = linspace(0,1,floor(N_pts+1));
            
            % intersects
            x_i_0 = pts_px*cos(theta_rad)*(x2-x1);
            y_i_0 = pts_px*sin(theta_rad)*(y2-y1);
            r_i = sqrt((x_i_0).^2+(y_i_0).^2);
            x_i = x_i_0+x1;
            y_i = y_i_0+y1;
            
            
            % Extracting oblique strids for linear interpolation
            ff = data(mod(floor(y_i)+size(data,1)*floor(x_i-1)-1,size(data,1)*size(data,2))+1);
            cf = data(mod(floor(y_i)+size(data,1)*floor(x_i)-1,size(data,1)*size(data,2))+1);
            fc = data(mod(floor(y_i+1)+size(data,1)*floor(x_i-1)-1,size(data,1)*size(data,2))+1);
            cc = data(mod(floor(y_i+1)+size(data,1)*floor(x_i)-1,size(data,1)*size(data,2))+1);
            
            
            data = (1-(y_i-floor(y_i))).*...
                ((1-(x_i-floor(x_i))).*ff...
                +(x_i-floor(x_i)).*cf)...
                +(y_i-floor(y_i)).*...
                ((1-(x_i-floor(x_i))).*fc...
                +(x_i-floor(x_i)).*cc);
            
            if nargout == 1
                argout1 = data;
            elseif nargout == 2
                argout1 = r_i;
                argout2 = data;
            end
        end
        
        function half_pitch_4x_nm = measure_halfpitch(img,nm_px,opt_arg)
        % MEASURE_HALFPITCH Estimate of the Half-pitch, wafer units (4x)
        %   half_pitch_nm_4x = measure_pitch(img,nm_px)
        %
        %   half_pitch_nm_4x = measure_pitch(img)
        %       assumes .33 4x NA lens data
        %
        % See also SHARP.CROSS_SECTION, SHARP.EXTRACT_LER
        
            % effective pixel size (default as .33 4xNA measurements)
            if nargin <2
                nm_px = 15;
            end
            % lateral size of the image (for the cross-cut)
            roi_size_px = size(img,2);
            % frequency scale
            f_cpnm = Sharp.fs((0:(roi_size_px-1))*nm_px);
            % Fourier Transform of the image
            IMG = abs(Sharp.ft(img));
            
            % take a lateral cross-section to locate the first order
            s = IMG(floor(end/2+1),:);
            df_cpnm = f_cpnm(2)-f_cpnm(1);
            
            % Getting coarse estimate of the 1st order frequency
            [~,imax] = max(s.*(f_cpnm>=df_cpnm));
            f0 = f_cpnm(imax);
            
            % Refining the frequency using a 2nd order fit
            pp = polyfit(f_cpnm((imax-1):(imax+1)),s((imax-1):(imax+1)),2);
            f0_fine = -0.5*pp(2)/pp(1);
            % if the coarse estimate does not agree with the fine estimate,
            % something is wrong
            if abs(f0_fine-f0)>df_cpnm
                warning('pitch detection might be inaccurate')
            end
            
            % half-pitch in wafer units
            half_pitch_4x_nm = 1/f0_fine/2;
            
            % consistency test; if the result is not consistent, maybe the
            % region of interest hasn't been properly chosen
            if nargin>2 && strcmp(opt_arg,'ConsistencyTest')
                if size(img,2)>10
                    % iteratively redo with smaller rois
                    fp = 0.5./Sharp.measure_halfpitch(img(:,1:end-2),nm_px);
                    fm = 0.5./Sharp.measure_halfpitch(img(:,1:end-5),nm_px);
                    
                    if abs(f0_fine-fp)>df_cpnm || abs(f0_fine-fm)>df_cpnm
                        disp('Sharp.pitch_measure :: Consistency test not passed')
                    end
                end
            end
        end
        
        %TODO IMPLEMENT
        function img_lines = extract_lines(img, offset_px)
            % EXTRACT_LINES Extract individual lines from an image for later analysis
            %   img_lines = Sharp.extract_lines(img)
            %
            % See also SHARP.EXTRACT_LER, SHARP.MEASURE_HALFPITCH
            
            if nargin<2
                offset_px = 0
            end
            
            % effective pixel size
            pxnm = 15;
            % determine the pitch
            halfpitch_4xnm = Sharp.measure_halfpitch(img,pxnm);
            % set the pitch size in px
            pitch_px = halfpitch_4xnm*2/pxnm;
            % number of lines contained in the picture
            N_lines = round(size(img,2)/pitch_px);
            % find the maximum oof the center line to center the other lines
            [~,i_offset] = max(sum(img(:,floor((N_lines-1)/2*pitch_px):floor((N_lines+1)/2*pitch_px)),1));
            % if the maxiumum is after the center, make sure the first line
            % is not a cut line
            if i_offset>pitch_px/2
                i_offset = i_offset - pitch_px/2;
            end
            
            i_offset = i_offset-offset_px;
            
            % extract all the lines
            img_lines = cell(1,N_lines-1);
            for i=1:(N_lines-1)
                try
                    img_lines{i} = img(:,(floor(((1+(i-1)*pitch_px):i*pitch_px)-(i_offset)+(pitch_px)/2)));
                catch
                    img_lines{i} = [];
                end
            end
        end
        
        %FIXME make sure the line is vertical and positive
        function [ x1, x2 ] = extract_ler( img_line, threshold, interp_factor, opt_arg)
            %EXTRACT_LER Extract the left and righ edge of a line
            %   [ left_edge, right_edge ] = extract_ler( img_line, threshold, interp_factor )
            %       where img_line is an image containg only one line
            %             threshold is the threshold value
            %             interp_factor the optional interpolation factor
            %
            % This function uses linear interpolation for finding the edges
            % in a robust an efficient way
            %
            % See also Sharp.extract_lines
            
            if isa(img_line,'cell')
                error('Sharp.extract_ler > please provide a valid single image');
            end
            
            
            if threshold>max(img_line(:))
                warning('Sharp.extract_ler > threshold higher than the max value');
            end
            
            if threshold<min(img_line(:))
                warning('Sharp.extract_ler > threshold lower than the min value');
                if min(img_line(:))>500
                    warning('you might want to check background subtraction');
                end
            end
            

            
            % size of th imaage
            x_px = size(img_line,1);
            y_px = size(img_line,2);
            % if some interpolation is needed, resize the imgae (brute force)
            if nargin<3
                interp_factor = 1;
            else
                img_line = Sharp.resize(img_line,...
                    x_px*interp_factor,y_px*interp_factor);
            end
            
            % position of the edge
            x1 = zeros(1,x_px*interp_factor);
            x2 = zeros(1,x_px*interp_factor);
            for i=1:x_px*interp_factor
                % for every vertical position, find the threshold cut
                test = img_line(i,:);
                bounds = find(test>threshold);
                % here are the coarse estimate of threshold crossing
                ixl_e = min(bounds);
                ixr_e = max(bounds);
                
                % size increment
                dx = 1;
                % refine the threasold crossing estimate using
                % explicit linear interpolation
                try
                    % left edge
                    if ixl_e>1 %make sure there is a left edge
                        xl = ixl_e-(test(ixl_e)-threshold)/(test(ixl_e)-test(ixl_e-1))*dx;
                    else %otherwise, pupulate missing edge as NaNs
                        xl = NaN;
                    end
                    % right edge
                    if ixr_e<length(test)
                        xr = ixr_e-(test(ixr_e)-threshold)/(test(ixr_e+1)-test(ixr_e))*dx;
                    else
                        xr = NaN;
                    end
                catch err
                    rethrow(err)
                end
                
                % output
                x1(i) = xl;
                x2(i) = xr;
                
                if nargin == 4
                    if strcmp(opt_arg,'detrend')
                        x1 = detrend(x1);
                        x2 = detrend(x2);
                    end
                end
            end
            
            idx_l = round(x_px/2)*2;
            if nargout==0
                plot((1:size(img_line,2))*15-idx_l/2, img_line(round(end/2),:),'k.-',...
                    x1(round(end/2))*15-idx_l/2,interp1([floor(x1(idx_l)),ceil(x1(idx_l))],...
                    [img_line(end/2,floor(x1(idx_l))),img_line(end/2,ceil(x1(idx_l)))],x1(idx_l)),'rx',...
                    x2(round(end/2))*15-idx_l/2,interp1([floor(x2(idx_l)),ceil(x2(idx_l))],...
                    [img_line(end/2,floor(x2(idx_l))),img_line(end/2,ceil(x2(idx_l)))],x2(idx_l)),'rx')
                imagesc(img_line)
                axis  off
                hold on
                plot(x1,1:length(x1),'b',x2',1:length(x2),'r')
                hold off
            end
%             x_nm = ((1:size(img_line,2))-1)*15-215;
%             mm = max(img_line(1,:));
%             plot(x_nm,img_line(1,:)/mm,'k-o',xl*15-237.5,threshold/mm,'r+',xr*15-237.5,threshold/mm,'b+')
%             axis tight
%             xlabel('position [nm]')
%             ylabel('intensity [a.u.]')
%             xlim([-160 160])
            
        end
        
        function [ x_d, I_d, cd_m, hp_m, NILS ] = extract_cd( x_m, I_ct, threshold_ct )
            %EXTRACT_CD extraction of CD from a measurement
            %   [ x_d, I_d, cd_m, hp_m, NILS ] = extract_cd( x_m, I_ct, threshold_ct )
            %
            % it gives you the spatial coordinates at threshold crossing (x_d) and the
            % value at this point (I_d should be threshold!).
            % It also spits out the cd in nm for each line (if the actual spatial scale is given)
            % and the (signed) NILS for each edge.
            
            if size(I_ct,1)>1 && size(I_ct,2)>1
                error('please provide 1D data')
            end
            i_l = logical(abs([diff(double(I_ct>threshold_ct)) 0]));
            i_r = logical(abs([0 diff(double(I_ct>threshold_ct))]));
            x_l = x_m(i_l);
            I_l = I_ct(i_l);
            I_r = I_ct(i_r);
            
            frac = (threshold_ct-I_l)./(I_r-I_l);
            dx_m = x_m(2)-x_m(1);
            %
            x_d = x_l+frac*dx_m;
            I_d= I_l+frac.*(I_r-I_l);
            

                temp = diff(x_d);
                cd_m = temp(1:2:end);

            

                hp_m = mean(temp);

            

                NILS = interp1(x_m-dx_m/2,[0 mean(cd_m)*diff(log(I_ct))/dx_m],x_d);

            
            if nargout==0
                plot(x_m,abs([0 mean(cd_m)*diff(log(I_ct))/dx_m]),x_d,abs(NILS),'o')
                
            end

        end
        
        function [ lw, xl, xr ] = extract_lw( array_line, threshold)
            %EXTRACT_LW
            %   [ lw, xl, xr ] = extract_lw( array_line, threshold)
            % See also Sharp.extract_ler
            
            if isa(array_line,'cell')
                error('Sharp.extract_ler > please provide a valid single image');
            end
            
            
            if threshold>max(array_line(:))
                warning('Sharp.extract_ler > threshold higher than the max value');
            end
            
            if threshold<min(array_line(:))
                warning('Sharp.extract_ler > threshold lower than the min value');
                if min(array_line(:))>500
                    warning('you might want to check background subtraction');
                end
            end
            
            % for every vertical position, find the threshold cut
            bounds = find(array_line>threshold);
            % here are the coarse estimate of threshold crossing
            ixl_e = min(bounds);
            ixr_e = max(bounds);
            
            % explicit linear interpolation
            try
                % left edge
                if ixl_e>1 %make sure there is a left edge
                    xl = ixl_e-(array_line(ixl_e)-threshold)/(array_line(ixl_e)-array_line(ixl_e-1));
                else %otherwise, pupulate missing edge as NaNs
                    xl = NaN;
                end
                % right edge
                if ixr_e<length(array_line)
                    xr = ixr_e-(array_line(ixr_e)-threshold)/(array_line(ixr_e+1)-array_line(ixr_e));
                else
                    xr = NaN;
                end
            catch err
                rethrow(err)
            end
            
            lw = xr-xl;
            
        end
        

        
        function [ rlow ] = randlow( N,M, freq )
            % RANDPHASE Generation of Low Frequency Noise
            %   [ rlow ] = Sharp.randlow( N,M, freq )
            %        generates a low frequency noise matrix
            %        of dimension NxM, with max frequency freq
            %        (freq should be <1)
            % Example :
            %   lfn = randlow(100,100,0.01)

            [X,Y] = meshgrid(linspace(-1,1,N),linspace(-1,1,M));
            
            rlow = abs(ifft2(exp(-abs(X/freq)).*exp(-abs(Y/freq)).*exp(1i*2*pi*rand(N,M))));
            %imagesc(rlow); axis image
        end
        
        %FIXME make sure legit....
        function [ img_out ] = invert( img_in )
            % INVERT    Image inversion
            %   img_inverse = Sharp.invert(img)
            
            M = max(img_in(:));
            m = min(img_in(:));
            img_out = ((img_in-m)/(M-m)-0.5)*(-1)*(M-m)+m;
        end
        
        function [x_d,y_d] = register(img1,img2)
        % REGISTER Sub-pixel registration
        %   [xd, yd] = Sharp.register(img1,img2) give the distance between
        %   two images
        %
        if size(img1,1) ~= size(img2,1) && size(img1,2) ~= size(img2,2)
            error('Sharp.register : the two images must be the same size')
        end
        
            [a,~] = Sharp.dftregistration(fft2(img1),fft2(img2),15);
            x_d = a(4);
            y_d = a(3);
        end
        
        function [x_d,y_d] = register_fine(img1,img2, subpix)
        % REGISTER_fINE Sub-pixel registration
        %   [xd, yd] = Sharp.register_fine(img1,img2,subpix) give the distance between
        %   two images
        %
        if size(img1,1) ~= size(img2,1) && size(img1,2) ~= size(img2,2)
            error('Sharp.register : the two images must be the same size')
        end
        
            [a,~] = Sharp.dftregistration(fft2(img1),fft2(img2),subpix);
            x_d = a(4);
            y_d = a(3);
        end
        
        %TODO incorporate better, correct zoom
        function [h] = imagesc(varagin)
            % IMAGESC Augmented version of imagesc (keeps zoom etc.)
            %   [h] = Sharp.imagesc(img)
            
            zoom_update = ~isempty(get(gcf,'Children'));
            if zoom_update
                h = gca;
                if ishandle(h)
                    x_lim = get(h,'Xlim');
                    y_lim = get(h,'Ylim');
                end
            end
            try
                imagesc(varagin)
            catch err
                if strcmp(err.identifier,'MATLAB:hg:udd_interface:goSetUDDPointerProp:CannotSetProperty')
                    Sharp.imagec(varagin)
                else
                    rethrow(err)
                end
            end
            axis image off;
            if zoom_update
                set(h,'XLim',x_lim);
                set(h,'YLim',y_lim);
            end
        end
        
        function h = imagespec(img)
        %IMAGESC Displays the direct spectrum intensity of an image, with dc removed
        %   handle = Sharp.imagespec(img);
        %
        %   See also Sharp.imagec
        
            imagesc(Sharp.remove_dc(abs(Sharp.ft(img)).^2));
            axis image off
        end
        
        
        function img_rgb = hsv(img)
            % HSV hsv scaling of the data, returned as a rgb matrix
            % the image
            %   img_rgb = Sharp.hsv(img)
            %   returns the MxNx3 matrix to a HSV scaling of the data.
            
            [N_row,N_col] = size(img);
            
            map = @(x,channel) min(6*mod(x-(channel+1)/3,1),1).*...
                max(min(6*(2/3-mod(x-(channel+1)/3,1)),1),0);
            
            img_abs = abs(img(:)-min(img(:)))/((max(img(:))-min(img(:))));
            
            img_rgb(:,:,1)  = reshape(map(img_abs,1),N_row,N_col);
            img_rgb(:,:,2)  = reshape(map(img_abs,2),N_row,N_col);
            img_rgb(:,:,3)  = reshape(map(img_abs,3),N_row,N_col);
        end
        
        

        function img_rgb = hsv2(img)
            % HSV2 Alternative hsv scaling of the image
            %   img_rgb = Sharp.hsv2(img)
            %   uses a triple sine-coded hsv mapping for the data
            
            [N_row,N_col] = size(img);
            
            %x=0:0.01:1;
            map = @(x,delta) cos(2*pi*(x-(delta-1)/3)).*(1+sign(cos(2*pi*(x-(delta-1)/3))))/2;
            %plot(x,map(x,1),x,map(x,2),x,map(x,3))
            
            
            img_abs = abs(img(:)-min(img(:)))/(max(img(:))-min(img(:)));
            
            %max(img_arg)
            %third dimension
            img_rgb(:,:,1)  = reshape(map(img_abs,3),N_row,N_col);
            img_rgb(:,:,2)  = reshape(map(img_abs,2),N_row,N_col);
            img_rgb(:,:,3)  = reshape(map(img_abs,1),N_row,N_col);
        end
       
        %% Display
        function  jetplot(N)
            % JETPLOT   Plot several graphs with a nice colormap
            %   Sharp.jetplot( N ) should be place before the plot
            set(gca,'ColorOrder',jet(N),'NextPlot','ReplaceChildren')
        end


        function img_rgb = jet(img)
            %SHARP.JET Matlab's Jet colormap
            
            [N_row,N_col] = size(img);
            
            map = @(x,channel) max(min(4*(x+1/8),1),0).*...
                max(min(4*(5/8-x),1),0);
            
            map1 = @(x) map(1-x);
            map2 = @(x) map(1-x-1/4);
            map3 = @(x) map(x);
            
            img_abs = abs(img(:)-min(img(:)))/(max(img(:))-min(img(:)));
            
            %max(img_arg)
            %third dimension
            img_rgb(:,:,1)  = reshape(map1(img_abs),N_row,N_col);
            img_rgb(:,:,2)  = reshape(map2(img_abs),N_row,N_col);
            img_rgb(:,:,3)  = reshape(map3(img_abs),N_row,N_col);
        end
        
        function img_rgb = aims_cmap(img)
            %SHARP.AIMS_CMAP AIMS's Colormap
            
            [N_row,N_col] = size(img);
            
            map = @(x,channel) max(min(4*(x+1/8),1),0).*...
                max(min(4*(5/8-x),1),0);
            
            map1 = @(x) (256*(1.5-x)).*0.5.*(1-sign(0.5-x));
            map2 = @(x) 256*(1-4*abs((x-0.5)))*0.5.*(1+sign((1-4*abs((x-0.5)))));
            map3 = @(x) 256*(0.5+x).*0.5.*(1-sign(x-0.5));
            
            
            img_abs = abs(img(:)-min(img(:)))/(max(img(:))-min(img(:)));
            
            %max(img_arg)
            %third dimension
            img_rgb(:,:,1)  = uint8(reshape(map1(img_abs),N_row,N_col));
            img_rgb(:,:,2)  = uint8(reshape(map2(img_abs),N_row,N_col));
            img_rgb(:,:,3)  = uint8(reshape(map3(img_abs),N_row,N_col));
        end
        
        function img_rgb = aims_cmap2(img)
            %SHARP.AIMS_CMAP AIMS's Colormap
            
            [N_row,N_col] = size(img);
            
            k=8;
            sm = @(x) 0.5*(1+tanh(-k*x));
            
            map1 = @(x) 256*(1.5-x).*sm(0.5-x);
            map2 = @(x) 256*(1-4*abs((x-0.5)))*0.5.*(1+sign((1-4*abs((x-0.5)))));
            map3 = @(x) 256*(0.5+x).*sm(x-0.5);
            
            
            img_abs = abs(img(:)-min(img(:)))/(max(img(:))-min(img(:)));
            
            %max(img_arg)
            %third dimension
            img_rgb(:,:,1)  = uint8(reshape(map1(img_abs),N_row,N_col));
            img_rgb(:,:,2)  = uint8(reshape(map2(img_abs),N_row,N_col));
            img_rgb(:,:,3)  = uint8(reshape(map3(img_abs),N_row,N_col));
        end
%% Coordinates
       
        function plot_coorderror(target_x,target_y,mask_x,mask_y,scale)
        %PLOT_COORD_ERROR Plots the difference between the target position
        %   and the actual position
        %   plot_coorderror(target_x,target_y,mask_x,mask_y,scale)
        
            if nargin<5
                scale = 1000;
            end
            
            clf
            Sharp.jetplot(max(size((target_x))))
            for i=1:max(size((target_x)))
                temp_x = [target_x(i);(mask_x(i)-target_x(i))*scale+target_x(i)];
                temp_y = [target_y(i);(mask_y(i)-target_y(i))*scale+target_y(i)];
                plot(temp_x,temp_y ,'-');
                hold on
            end
            
            plot(target_x,target_y,'k.')
            xlabel('x position on mask [mm]')
            ylabel('y position on mask [mm]')
            title(sprintf('err(%dum/div)',10*1000/scale))
            set(gca,'YTick',[-76,-70:10:70,76])
            set(gca,'XTick',[-76,-70:10:70,76])
            xlim([-80 80])
            ylim([-80 80])
            grid on
            axis square
            hold off
        end


        function [ mx, my ] = coord_transform( sx, sy, ABCD, DXY )
        %COORD_TRANSFORM SHARP coordinate transform from smask to imask
        %   [ mx, my ] = coord_transform( sx, sy, ABCD, DXY )
        %       where ABDC is the mask_{a,b,c,d} matrix 
        %         and DXY is mask_{dz,dy} vector
        %       
        % See also SHARP.IMASK_TRANSFORM
        
            mx = zeros(1,length(sx));
            my = zeros(1,length(sy));
            for i=1:length(sx)
                M = ABCD*([sx(i);sy(i)]-DXY);
                mx(i)= M(1);
                my(i)= M(2);
            end
            if size(sx,1)>size(sx,2)
                mx = mx';
                my = my';
            end
        end
        
        function [cimask_x, cimask_y] = imask_correction(imaskraw_x, imaskraw_y)
            %IMASK_CORRECTION Correction of interferometer readings 
            %   [cimask_x, cimask_y] = imask_correction(imaskraw_x, imaskraw_y)
            %
            % See also SHARP.IMASK_TRANSFORM
            
            cimask_x = (imaskraw_x-polyval(Sharp.px,imaskraw_y*1e-3)*1e3);
            cimask_y = (imaskraw_y-polyval(Sharp.py,imaskraw_x*1e-3)*1e3);
        end
        
        function [tismask_x, tismask_y] = coord_rigid(cimask_x, cimask_y, dx ,dy, theta_rad)
            % COORD_RIGID Rigid bogy coordinate tranform
            
            theta_y = theta_rad-300e-6+0.01e-3*-1;
            if nargin<3
                theta_rad = -4.3e-3+0.001e-3*-41;
                DXY = [-3.5081; -2.1714]-[-11+0.1*-21;+13+0.1*15]*1e-3;
            else
                DXY = [dx; dy];
            end
            
            ABCD = [cos(theta_rad), sin(theta_rad); -sin(theta_y), cos(theta_y)];
            [tismask_x, tismask_y] = Sharp.coord_transform(cimask_x,cimask_y,ABCD,DXY);
        end
        
        function [tismask_x, tismask_y] = imask_transform(imask_x, imask_y, dx, dy, theta)
        %IMASK_TRANSFORM
        %   [tismask_x, tismask_y] = imask_transform(imask_x, imask_y, dx, dy, theta)
        %
        % See also SHARP.IMASK_CORRECTION, SHARP.IMASK_FIT
        
            [cimask_x,cimask_y] = Sharp.imask_correction(imask_x,imask_y);
            [tismask_x, tismask_y] = Sharp.coord_rigid(cimask_x, cimask_y, dx, dy,theta);
        end
        
        function [ mx, my ] = tilt_shift( sx, sy, dx, dy, theta )
        %TILT_SHIFT SHARP rigid-body coordinate transform
        %   [ mx, my ] = tilt_shift( sx, sy, dx, dy, theta )
        %
        % See also SHARP.COORD_TRANSFORM
            
            mx = zeros(1,length(sx));
            my = zeros(1,length(sy));
            
            T =  [cos(theta), sin(theta), dx;- sin(theta), cos(theta), dy; 0, 0, 1];
            
            for i=1:length(sx)
                M = T*([sx(i);sy(i);1]);
                mx(i)= M(1);
                my(i)= M(2);
            end
            if size(sx,1)>size(sx,2)
                mx = mx';
                my = my';
            end
        end
        
        function [ error ] = imask_fit_fcn( imask_x,imask_y,target_x,target_y,dx,dy,theta )
        %IMASK_FIT_FCN Merit function for imask fit optimization
        %   error  = imask_fit_fcn(imask_x, imask_y, target_x, target_y, dx, dy, theta )
        %
        % See also SHARP.IMASK_FIT
            
            [timask_x, timask_y] = Sharp.tilt_shift(imask_x,imask_y,dx,dy,theta);
            error = sum(sqrt((timask_x(:)-target_x(:)).^2+(timask_y(:)-target_y(:)).^2));
        end
        
        
        function X = imask_fit(target_x, target_y, imask_x, imask_y)
        %IMASK_FIT Merit function for imask fit optimization
        %   X  = imask_fit(target_x, target_y, imask_x, imask_y)
        %       where X = [dx_um, dy_um, theta_rad]
        %
        % See also SHARP.IMASK_FIT_FCN
            
            fun = @(X) Sharp.temp_opt( imask_x,imask_y,target_x,target_y,X(1),X(2),X(3) );
            X0 = [0;0;0];
            X = fminsearch(fun,[0;0;0]);
        end

        function printmat(matrix,precision)
            %PRINTMAT Print matrices to the console in a copyable form
            %   Sharp.printmat(matrix, precision)
            
            if nargin<2
                precision = 3;
            end
            format = sprintf('%s%de','%+1.',precision);
            fprintf('\n[')
            
            if size(matrix,1)==1 || size(matrix,2)==1
                for i=1:length(matrix)
                    if i>1
                        fprintf(', ');
                    end
                    fprintf(format,matrix(i));
                end
            else
                for i=1:size(matrix,1)
                    for j=1:size(matrix,2)
                        if i>1 && j==1
                            fprintf(' ');
                        end
                        if j>1
                            fprintf(', ');
                        end
                        fprintf(format,matrix(i));
                    end
                    if i<size(matrix,1)
                        fprintf(';...\n')
                    end
                end
            end
            fprintf(']\n')
        end
              
        
        %FIXME not working
        function img_rgb = bbr(img)
            % BBR Blue-Black-Red color mapping of an image
            %   img_rgb = Sharp.bbr(img)
            
            [N_row,N_col] = size(img);

            %x=0:0.01:1;
            map = @(x,delta) 0.5*(cos(2*pi*x/delta)+1);
            %plot(x,map(x,1),x,map(x,2),x,map(x,3))
            plot(map(0:0.1:1,1))

            img_abs = abs(img(:)-min(img(:)))/(max(img(:))-min(img(:)));
            
            %max(img_arg)
            %third dimension
            img_rgb(:,:,1)  = reshape(map(img_abs,2),N_row,N_col);
            img_rgb(:,:,2)  = reshape(map(img_abs,3),N_row,N_col);
            img_rgb(:,:,3)  = reshape(map(img_abs,2),N_row,N_col);
        end
        
        
        %FIXME not working
        function img_rgb = bwr(img)
            % BWR   Blue-White-Red color mapping of the data
            %   img_rgb = Sharp.bwr(img)
            %   returs the MxNx3 matrix corresponding to the colormap
            

            %x=0:0.01:1;
%             map = @(x,delta) 0.5*(cos(2*pi*x/delta)+1);
%             %plot(x,map(x,1),x,map(x,2),x,map(x,3))
%             plot(map(0:0.1:1,1))

            img_norm = img./(max(abs(img(:))));
            
            img_n = img_norm.*double((img_norm<0));
            img_p = img_norm.*double((img_norm>0));
            
            %max(img_arg)
            %third dimension
            img_rgb(:,:,1)  = 1-img_p;
            img_rgb(:,:,2)  = 1;
            img_rgb(:,:,3)  = 1-img_n;
        end
        
        
        function img_rgb = imagec( img_complex )
            % IMAGEC Image of complex image with hsv color mapping
            %   imagec( img_complex ) displays an image where
            %   amplitude is mapped on the value and the
            %   phase is mapped on the hue.
            %
            %   See also SHARP.HSV, SHARP.COLORSCALE
            
            [N_row,N_col] = size(img_complex);
            
            map = @(x,channel) min(6*mod(x-(channel+1)/3,1),1).*...
                max(min(6*(2/3-mod(x-(channel+1)/3,1)),1),0);
            
            img_abs = abs(img_complex(:))/(max(abs(img_complex(:))));
            img_arg = angle(img_complex(:))/(2*pi)+0.5;
            
            %max(img_arg)
            %third dimension
            img_rgb(:,:,1)  = reshape(map(img_arg,1).*abs(img_abs),N_row,N_col);
            img_rgb(:,:,2)  = reshape(map(img_arg,2).*abs(img_abs),N_row,N_col);
            img_rgb(:,:,3)  = reshape(map(img_arg,3).*abs(img_abs),N_row,N_col);
            
            if nargout == 0
                image(img_rgb);
            end
            
            axis image
        end
        
        function img_rgb = imagecc( img_complex, ratio )
            % IMAGECC Image of complex image with hsv color mapping
            %   imagecc( img_complex, ratio ) displays an image where
            %   amplitude is mapped on the value and the
            %   phase is mapped on the hue.
            %
            %   See also SHARP.HSV, SHARP.COLORSCALE
            
            if nargin == 1
                ratio = 4; %default = quarterwave
            end
            
            [N_row,N_col] = size(img_complex);
            
            map = @(x,channel) min(6*mod(x-(channel+1)/3,1),1).*...
                max(min(6*(2/3-mod(x-(channel+1)/3,1)),1),0);
            
            img_abs = abs(img_complex(:))/(max(abs(img_complex(:))));
            img_arg = angle(img_complex(:))/(2*pi)*ratio+0.5;
            
            %max(img_arg)
            %third dimension
            img_rgb(:,:,1)  = reshape(map(img_arg,1).*abs(img_abs),N_row,N_col);
            img_rgb(:,:,2)  = reshape(map(img_arg,2).*abs(img_abs),N_row,N_col);
            img_rgb(:,:,3)  = reshape(map(img_arg,3).*abs(img_abs),N_row,N_col);
            
            if nargout == 0
                image(img_rgb);
            end
            
            axis image
        end

        
        %FIXME : axis argument
        function [ h ] = imagezc( img )
            %SHARP.IMAGEZC Zero-Centered image display
            %   [ h ] = imagezc( img )
            h = image(img./max(img(:))*64);
        end

        
        function cscale = colorscale(size,initial_phase,flip)
            % SHARP.COLORSCALE Scaling of the amplitude+phase visaulization
            %   cscale = colorscale(size) outputs an image with the
            %   requested size
            %
            % See also SHARP.IMAGEC
            if nargin < 1;
                size = 512;
                initial_phase = 0;
                flip = false;
            elseif nargin<2
                initial_phase = 0;
                flip = false;
            elseif nargin<3
                flip = false;
            end
            
            [X,Y]=meshgrid(linspace(-1,1,size));
            
            
            
            A = atan2(Y,X);
            R = sqrt(X.^2+Y.^2);
            if flip
                A = -(A);
            end
            
            B = R.*exp(1i*(A+initial_phase));
            mask = R<1;

            
            cscale = Sharp.imagec(B.*mask);
        end
        
        %TODO : system agnostic, 
        %FIXME suffix in folder
        function fiji(argin)
            if ischar(argin)
                folder = argin;
            else
                folder = '/Users/awojdyla/Documents/MATLAB/SHARP/data';
            end
            system(sprintf('open /Applications/Fiji.app %s%s%s',folder,strtrim(meta.image_dir(15:end)),strtrim(meta.image_name)))
        end
        
        function open_folder(argin)
            if ischar(argin)
                folder = argin;
            else
                folder = '/Users/awojdyla/Documents/MATLAB/SHARP/data';
            end
            system(sprintf('open  %s%s',folder,strtrim(meta.image_dir(15:end))))
        end

        
        function addTick(value)
            %SHARP.ADDTICK Add a tick to the current plot on the x-axis
            set(gca,'XTick',sort([get(gca,'XTick') value]))
        end


        %TODO implement properly
        function imgs_out = add_slider(imgs_in)
        % ADD_SLIDER Showing sliders on images of a focus stack
        %   imgs_out = add_slider(imgs_in)
        %   imagesc(Sharp.add_slider(img_in){1})
            N = length(imgs_in);
            roi_size_px = size(imgs_in{1},1);
            imgs_out = imgs_in;
            for idx = 1:N
                wid = floor(roi_size_px/(N));
                pos = floor((idx-1)*wid)+1;
                imgs_out{idx}(end-round(roi_size_px/16):end,:) = 0;
                imgs_out{idx}(end-round(roi_size_px/16):end,pos:(pos+wid)) = max(imgs_out{idx}(:));
            end
        end
        
        function title(varargin)
            %SHARP.TITLE EZ title labelling
            % Sharp.title(i) will add a title to the gca with i printed
            title_arg = '.';
            for i = 1:nargin
                title_arg = strcat(title_arg,...
                    sprintf('. %0+2.2f . ',varargin{i}));
            end
            title(strcat(title_arg,'.'))
        end
        
        
        function [ cmap ] = colormap_bluetored( n_level )
            %DKBLUERED Summary of this function goes here
            %   Detailed explanation goes here
            
            if nargin < 1
                n_level = size(get(gcf,'colormap'),1);
            end
            
            r = linspace(0,1,n_level);
            g = zeros(1,n_level);
            b = linspace(1,0,n_level);
            
            cmap = [r' g' b'];%.*(1-L*0.5);
            
            if nargin == 0
                colormap(cmap)
            end
            
        end
        
        function [ cmap ] = colormap_dkbluered( n_level )
            %DKBLUERED Summary of this function goes here
            %   Detailed explanation goes here
            if nargin < 1
                n_level = size(get(gcf,'colormap'),1);
            end
            
            cmap = ones(n_level,3);
            
            x=1:n_level;
            r = min(0,(x/n_level-1/2))*2;
            g = -(-x/n_level+1/2).*sign(-x/n_level+1/2)*2;
            b = min(0,(-x/n_level+1/2))*2;
            l = abs(x-n_level/2)/n_level;
            L=repmat(l,3,1)';
            
            
            cmap = (cmap + [r' g' b']).*(1-L*0.5);
            
            if nargin == 0
                colormap(cmap)
            end
        end
        

        

        function filename = print(idx)
            % SHARP.PRINT Prints the current figure into the current folder
            %   Sharp.print(idx);
            %   prints the current current figure using the unix timestamp
            %   and its index
            %
            % Example : 
            % x=-2:0.1:2
            % for i=10
            %   plot(1:10, x.^i)
            %   Sharp.print(i)
            % end
            
            if nargin==0
                idx = 0;
            end
            filename = sprintf('%s_%04.0f', datestr(now,'yyyymmddhhMMSS'), idx);
            
            chandles =  findobj(gcf,'type','axes');
            for i=1:length(chandles)
                if length(chandles)==1
                    set(chandles(i),'FontSize',12)
                else
                    set(chandles(i),'FontSize',12)
                end
            end
            

            print(filename,'-dpng');
            
            save_folder = Sharp.save_folder2;
            if exist(save_folder,'dir')~=7
                if ispc
                     save_folder = Sharp.save_folder_win;
                elseif ismac
                    save_folder = Sharp.save_folder_mac;
                end
                
            end
            print(strcat(save_folder,filename),'-dpng');
            h=gcf;
            saveas(h,strcat(save_folder,filesep,filename),'fig')
            saveas(h,strcat(save_folder,filesep,filename),'pdf')
            
        end
        
        function filename = print2(font_size, linewidth)
            % SHARP.PRINT2 Prints the current figure into the current folder
            %   Sharp.print(idx);
            %   prints the current current figure using the unix timestamp
            %   and its index

            filename = sprintf('%s', datestr(now,'yyyymmddhhMMSS'));
            
            chandles =  findobj(gcf,'type','axes');
            for i=1:length(chandles)
                if length(chandles)==1
                    set(chandles(i),'FontSize',font_size)
                else
                    set(chandles(i),'FontSize',font_size)
                end
            end
            
            chandles =  findobj(gcf,'type','line');
            for i=1:length(chandles)
                set(chandles,'LineWidth',linewidth)
            end

            print(filename,'-dpng');
            print(strcat(Sharp.save_folder2,filename),'-dpng');
            h=gcf;
            saveas(h,strcat(Sharp.save_folder2,'/',filename),'fig')
            saveas(h,strcat(Sharp.save_folder2,'/',filename),'pdf')
            
        end

        %% I/O
        
        %FIXME : overwrite check
        function write_png(img, filename, folder)
            current_folder = pwd;
            
            if nargin == 3
                cd(folder)
            end
            try
                if length(filename)<4 || ~strcmp(filename((end-3):end),'.png')
                    filename = strcat(filename,'.png');
                end
                imwrite(uint16(img/max(img(:))*(2^16-1)),filename,'png')
                catch err
                if nargin ==3 
                    cd(current_folder)
                end
                rethrow(err)
            end
            if nargin == 3
                cd(current_folder)
            end
        end
        
        %TODO
        function thumbnail(img)
            meta = Sharp.metadata(img);
            img_roi = Sharp.ROI(img);
            imwrite()        ;    
        end
        
        function thumbnail_jpg(img,filename)
            if size(img,1)==2048
                img = Sharp.resize(img,512);
            end
            imwrite(uint8(img/max(img(:))*(2^8-1)),strcat(filename,'.jpg'),'jpg')
        end
        
        function write_hdf5(img, filename, folder)
            % WRITE_HDF5    Write complex data using HDF5 file format
            %   Sharp.write_hdf5(img, filename)
            %   writes the complex image into the current folder
            %
            %   Sharp.write_hdf5(img, filename, folder)
            %   writes the complex image into a specific folder
            
            %going to the destination folder, if needed
            if nargin == 3
                current_folder = pwd;
                cd(folder)
            end
            
            try
                if ~strcmp(filename((end-2):end),'.h5')
                    filename = strcat(filename,'.h5');
                end
                hdf5write(filename,'/real part',real(img),'/imaginary part',imag(img))
                
            catch err
                if nargin ==3
                    cd(current_folder)
                end
                rethrow(err)
            end
            if nargin == 3
                cd(current_folder)
            end
        end
        
        
        function img = read_hdf5(filename)
            % READ_HDF5 Reads complex data encoded in HDF5 file format
            %   img = Sharp.read_hdf5(filename)
            %   read the complex image using its filename
            %
            % See also Sharp.write_hdf5

            %info  = hdf5info(filename);
            try
                img_real = hdf5read(filename,'/real part');
                img_imag = hdf5read(filename,'/imaginary part');
                img = img_real + 1i*img_imag;
            catch err
                rethrow(err)
            end
        end
        
        %overwrite protection
        function k = write_kif( filename, x, y)
            % WRITE_KIF Write Ken-Iacopo interchange Format files
            %   fid = Sharp.write_kif(filename);
            %   x is the real part of the data and y is the imaginary part (if any)
            %
            % See also Sharp.write_kif, Sharp.read_kif, Sharp.read_kifc
            
            % I Mochi
            
            %converting to a row major environment
            % A wojdyla Sept'15
            x=x';
            y=y';
            
            if nargin == 2
                re = real(x);
                im = imag(x);
                x = re;
                y = im;
            end
            
            if length(filename)<4 || ~strcmp(filename(end-3:end),'.kif')
                filename = strcat(filename,'.kif');
            end
            
            fid = fopen(filename,'wb');
            fwrite(fid,size(x,1),'uint32');
            fwrite(fid,size(x,2),'uint32');
            
            fwrite(fid,x,'float32');
            fwrite(fid,y,'float32');
            
            k=fclose(fid);
        end
        
        
        function k = write_kifc( filename, aeiphi )
            %  WRITE_KIFV Write Ken-Iacopo interchange Format files
            %   fid = Sharp.write_kifc(filename,img_complex);
            %   x is the real part of the data and y is the imaginary part (if any)
           
                 k = Sharp.write_kif( filename, real(aeiphi), imag(aeiphi));
        end
        
        function [x,y] = read_kif( filename )
            % READKIF Read Ken-Iacopo Interchange format
            %   [x,y] = Sharp.read_kif(filename);
            %   x is the real part of the data and y is the imaginary part (if any)
            % See also Sharp.read_kifc, Sharp.write_kif, Sharp.write_kifc
            
            % I Mochi
            
            if length(filename)<4 || ~strcmp(filename(end-3:end),'.kif')
                filename = strcat(filename,'.kif');
            end
            
            fid = fopen(filename,'rb');
            r=fread(fid,1,'uint32');
            c=fread(fid,1,'uint32');
            
            x=fread(fid,[r,c],'float32');
            y=fread(fid,[r,c],'float32');
            
            fclose(fid);
            
            %converting to a column major environment
            % A wojdyla Sept'15
            x=x';
            y=y';
        end
        
        function aeiphi = read_kifc( filename )
            % READKIFC Read Ken-Iacopo Interchange format, complex output
            %   [img_complex] = Sharp.read_kifc(filename);
            %      
            
            [re,im] = Sharp.read_kif(filename);
            aeiphi = re+1i*im;
        end
        
        %TODO
        function write_mat(img, filename, folder)
            % WRITE_MAT     Write the mat-file associated to the variable
            %   write_mat(img, filename, folder)
            
        end
        

        %FIXME : something's wrong here...
        %TODO : work with 3D arrays
        function export2gif(img_cell)
            % EXPORT2GIF    Creates a GIF with the images in the cell
            %   export2gif(img_cell) exports all the frames of img_cell as
            %   a gif in the current folder
            %
            % See also Sharp.tile_cell, Sharp.add_slider
            
            h = figure('Position',[0 0 size(img_cell{1},1) size(img_cell{1},1)]);
            set(gca,'position',[0 0 1 1],'units','normalized');
            colormap gray
            
            filename = sprintf('%s_%04.0f', datestr(now,'yyyymmddhhMMSS'));
            
            for i = 1:length(img_cell)
                imagesc(img_cell{i})
                axis image off
                drawnow
                frame = getframe(h);
                im = frame2im(frame);
                [imind,cm] = rgb2ind(im,256);
                if i == 1;
                    imwrite(imind,cm,filename,'gif', 'Loopcount',inf,'DelayTime',0.2);
                else
                    imwrite(imind,cm,filename,'gif','WriteMode','append');
                end
            end
            close(h)
                
                %             for i=length(images_cell)
                %                 %imagesc(images_cell{i},'Parent',aTemp)
                %                 imagesc(images_cell{i},'Parent',aTemp)
                %                 axis(aTemp,'image','off')
                %                 drawnow
                %
                %                 frame = getframe(hTemp);
                %                 im = frame2im(frame);
                %                 [A,map] = rgb2ind(im,256);
                %                 if i == 1;
                %                     imwrite(A,map,filename,'gif','LoopCount',Inf,'DelayTime',1);
                %                 else
                %                     imwrite(A,map,filename,'gif','WriteMode','append','DelayTime',1);
                %                 end
                %                 close(hTemp)
                %             end
        end
            


        %TODO : refresh
        function tofig(filename)
            % TOFIG     Write the current figure into a fig and a eps file
            %   Sharp.tofig(filename)
            %   writes the current figure into a fig and a eps file
            
            current_path = pwd;
            if nargin == 0
                cd(Sharp.save_folder);
            end
            
            
            set(findobj('Type','Line'),'LineWidth',2);
            set(gca,'Box','off')
            set(gca,'LineWidth',0.5);
            set(get(gca,'YLabel'),'FontSize',20);
            set(get(gca,'XLabel'),'FontSize',20);
            set(get(gca,'ZLabel'),'FontSize',20);
            set(get(gca,'title'),'FontSize',20);
            set(legend, 'FontSize',20)
            
            try
                print(gcf,filename,'-depsc');
                saveas(gcf,filename,'fig')
            catch err
                rethrow(err)
                cd(current_path)
            end
            cd(current_path)
            
        end
        
        
        function  fig2eps( filename )
            % FIG2EPS   Write a figure into an eps file
            %   Sharp.fig2eps(filename)
            %   writes the current figure into an eps file

            % Fontes available :
            % % Avante Garde              Palatino
            % % Bookman                   Symbol
            % % Courier                   Times
            % % Helvetica                 Zapf Chancery
            % % Helvetica Narrow          Zapf Dingbats
            % % New Century Schoolbook
            fonte = 'Bookman';
            
            hx = get(gca,'XLabel');
            hy = get(gca,'YLabel');
            ht = get(gca,'Title');
            set(gca,'FontName',fonte)
            set(hx,'FontName',fonte)
            set(hy,'FontName',fonte)
            set(ht,'FontName',fonte)
            
            exportfig(gcf,filename,'Color','rgb');
        end

        
        % slideshow
        
        % illumination type

        % resolution metrics
        
        % classifier
        
        function t = test(in)
            fprintf('Sharp.test\n')
        end


        function img_resized = resize2(img,x_size_px,y_size_px)
            % RESIZE2    Resize an image using FFT
            %   img_resized = Sharp.resize2(img, size_px)
            %   img_resized = Sharp.resize2(img, x_size_px, y_size_px)
            
            if nargin==2 
                if size(img,1)==size(img,2)
                    y_size_px = x_size_px;
                else
                    error('the image should be square if you only use two arguments.')
                end
            end
            
            if size(img,1)>x_size_px && size(img,2)>y_size_px
                IMG = Sharp.ft(img);
                IMG_CROPPED = Sharp.crop2(IMG,x_size_px,y_size_px);
                img_resized = real(Sharp.ift(IMG_CROPPED));
            else
                IMG = Sharp.ft(img);
                IMG_PADDED = Sharp.pad2(IMG,x_size_px,y_size_px);
                img_resized = real(Sharp.ift(IMG_PADDED));
            end
        end
        
        
        %FIXME there is a small lateral shift after each iterations...:(
        %FIXME not fully tested, expect funny results
        %TODO : allow for complex and NxMx3
        function [ img_rotated ] = rotate2( img, angle_deg )
            %ROTATE2     Rotate an image using HQAFIR algorithm
            %   img_rotate = Sharp.rotate(img,angle_deg)
            %   rotates the image by the specified angle, ccw.
            
            % base on the algorithm described in 
            % "High Quality Alias Free Image Rotation", Charles B Owen (1996)
            % it seems that the rotation causes a one-pixel shift of the
            % image; probably something to do with the (i)fftshifts
            
            if ~isreal(img)
                %error('Sharp.rotate : the image cannot be complex')
                img_rotated = abs(Sharp.rotate2(abs(img),angle_deg)).*...
                              exp(1i.*Sharp.rotate(angle(img),angle_deg));
                          disp('mama')
            else
                angle_rad = angle_deg*pi/180;
                N_row = size(img,1);
                N_col = size(img,2);
                Np =  2*N_row;ceil(N_row*(1+tan(angle_rad/2).^2));
                Npp = 2*N_col;
                
                img_rotated = Sharp.pad2(img,N_row,Np);
                [Fx1,Fy1] = meshgrid((1:Np)-Np/2,(1:N_col)-N_col/2);
                delta = tan(angle_rad/2);
                SKEW1 = exp(2*1i*pi*(Fx1.*Fy1)*delta/Np);
                IMG_ROT = fftshift(fft(ifftshift(img_rotated,2),[],2),2).*SKEW1;
                
                IMG_ROT2 = fftshift(fft(ifftshift(IMG_ROT,1),[],1),1);
                IMG_ROT2_PAD = pad2(IMG_ROT2,Npp,Np);
                IMG_ROT3 = fftshift(ifft(ifftshift(IMG_ROT2_PAD,2),[],2),2);
                delta2 = -2*sin(angle_rad);
                [Fx2,Fy2] = meshgrid((1:Np)-Np/2,(1:Npp)-Npp/2);
                SKEW2 = exp(2*1i*pi*(Fx2.*Fy2)*delta2/Np);
                IMG_ROT4 = IMG_ROT3.*SKEW2;
                
                IMG_ROT5 = fftshift(fft(ifftshift(IMG_ROT4,1),[],1),1);
                IMG_ROT6 = fftshift(ifft(fftshift(IMG_ROT5,2),[],2),2);
                delta3 = -tan(angle_rad/2)/2;
                [Fx3,Fy3] = meshgrid((1:Np)-Np/2,(1:Npp)-Npp/2);
                SKEW3 = exp(-2*1i*pi*(Fx3.*Fy3)*delta3/Npp);
                IMG_ROT7 = IMG_ROT6.*SKEW3;
                
                IMG_ROT8 = fftshift(ifft(ifftshift(IMG_ROT7,2),[],2),2);
                img_rotated = fliplr(flipud(abs(crop2((IMG_ROT8(1:2:end,:)),N_row,N_col))));
            end
        end
        
        function [aerial, params] = import_hyperlith(filename)
        %IMPORT_HYPERLITH Import aerial images from Hyperlith export txt files
        %   [{aerial}, {params}] = import_hyperlith(filename)
        %
        % See also ...
        
            %open the file
            fid = fopen(filename);
            
            %set image index to zero
            idx = 1;
            
            %scan the first line (for template)
            c = fgetl(fid);

            % the first line line looks like
            % 'biasx_nm=-20.0, biasy_nm=-20.0,defocus=-3600.0'
            % so we can split the parameters at the colon
            d = strsplit(c,',');
            % that's how we know how many parameters
            N_params = length(d);
            params = {};
            % let's populate the parameters' cell with a struct
            for i=1:N_params
                %split every sub-string at the equal sign
                % e.g. 'biasx_nm=-20.0'
                e = strsplit(d{i},'=');
                %remove spaces (field names should not have leading spaces)
                field = strtrim(strrep(strrep(e{1},']',''),'[',''));
                %populate the first field
                if i==1
                    field1=field;
                    params{idx}.(field) = str2double(e{2});
                else
                    %and all the others
                    params{idx} = setfield(params{idx},field,str2double(e{2}));
                end
            end

            % let's loop over the file
            idx_r = 1;
            c = fgetl(fid);
            
            % if there is still something to read, go on
            while ~(~ischar(c) && c==-1)
                if isempty(c)
                    % do nothing
                else
                    %if the line has non-numeral characters, populated the
                    %params
                    if strcmp(c(1:3),field1(1:3))
                        %increment the counter (new data)
                        idx = idx+1;
                        %there will be a new aerial image,
                        % so set the row count to one
                        idx_r = 1;
                        %see above for explanations
                        d = strsplit(c,',');
                        for i=1:N_params
                            e = strsplit(d{i},'=');
                            field = strtrim(strrep(strrep(e{1},']',''),'[',''));
                            if i==1
                                params{idx}.(field) = str2double(e{2});
                            else
                                params{idx} = setfield(params{idx},field,str2double(e{2}));
                            end
                        end
                    else
                        % read data
                        aerial{idx}(idx_r,:) = str2num(c);
                        %increment row counter (in case it's 2D data)
                        idx_r = idx_r+1;
                    end
                end
                %read next line
                c = fgetl(fid);
            end
            %close the file            
            fclose(fid);
        end
        
        
        function [time_s,pos1_um,pos2_um] = import_pico2(filename, startRow, endRow)
            %IMPORT_PICO2 Import numeric data from a text file as column vectors.
            %   [TIME_S,POS1_UM,POS2_UM] = IMPORTFILE(FILENAME) Reads data from text
            %   file FILENAME for the default selection.
            %
            %   [TIME_S,POS1_UM,POS2_UM] = IMPORTFILE(FILENAME, STARTROW, ENDROW) Reads
            %   data from rows STARTROW through ENDROW of text file FILENAME.
            %
            % Example:
            %   [time_s,pos1_um,pos2_um] = importfile('tap test',2, 31200);
            %
            
            % Auto-generated by MATLAB on 2016/08/11 16:17:42
            
            delimiter = '\t';
            if nargin<=2
                startRow = 2;
                endRow = inf;
            end
            
            formatSpec = '%f%f%f%*s%[^\n\r]';

            fileID = fopen(filename,'r');
            
            dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'EmptyValue' ,NaN,'HeaderLines', startRow(1)-1, 'ReturnOnError', false);
            for block=2:length(startRow)
                frewind(fileID);
                dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'EmptyValue' ,NaN,'HeaderLines', startRow(block)-1, 'ReturnOnError', false);
                for col=1:length(dataArray)
                    dataArray{col} = [dataArray{col};dataArrayBlock{col}];
                end
            end

            fclose(fileID);
            time_s = dataArray{:, 1};
            pos1_um = dataArray{:, 2};
            pos2_um = dataArray{:, 3};
        end
        
        
        
        function [ xc_px, yc_px] = fid_center( img )
            %FID_CENTER Finds the corner of fiducial on an image
            %   [ xc_px, yc_px] = fid_center( img )
            %           
            % See also Sharp.cross_center
            
            if ~isa(img,'cell')
                im_tmp = img;
                img = cell(1);
                img{1} = im_tmp;
            end
            
            roi_size_x = size(img{1},1);
            roi_size_y = size(img{1},2);
            
            
            xc_px = zeros(1,length(img))-roi_size_x/2;
            yc_px = zeros(1,length(img))-roi_size_x/2;
            for i_m =1:length(img)
                img_x = sum(img{i_m},1)/mean(sum(img{i_m},1));
                img_y = sum(img{i_m},2)/mean(sum(img{i_m},2));
                [~, xc_px(i_m)] = max(diff(img_x));
                [~, yc_px(i_m)] = max(diff(img_y));
            end
            xc_px = xc_px-roi_size_x/2;
            yc_px = xc_px-roi_size_x/2;
        end
        
        function [ centro ] = centroid( x,y )
            %CENTROID Centroid computation in one dimension (first moment)
            %   cent = centroid( x,y )
            %
            % A Wojdyla (awojdyla@lbl.gov), June 2013
            centro = trapz(x.*y)./trapz(y);
        end
        
        function [ xc_px, yc_px] = centroid2( img )
            %CENTROID Centroid computation in one dimension (first moment)
            %   cent = centroid( x,y )
            %
            % A Wojdyla (awojdyla@lbl.gov), June 2013
             if ~isa(img,'cell')
                im_tmp = img;
                img = cell(1);
                img{1} = im_tmp;
            end
            
            roi_size_x = size(img{1},1);
            roi_size_y = size(img{1},2);
            x_px = (-roi_size_x/2:roi_size_x/2-1);
            y_px = (-roi_size_y/2:roi_size_y/2-1);
            
            xc_px = zeros(1,length(img));
            yc_px = zeros(1,length(img));
            for i_m =1:length(img)
                img_x = sum(img{i_m},1)/mean(sum(img{i_m},1));
                img_y = sum(img{i_m},2)/mean(sum(img{i_m},2));
                xc_px(i_m) = Sharp.centroid(x_px,img_x);
                yc_px(i_m) = Sharp.centroid(y_px,img_y');
            end
        end
        
        function [ xc_px, yc_px] = cross_center( img )
            %CROSS_CENTER Find the centre of a symmtrical pattern
            %   [ xc_px, yc_px] = cross_center( {img} )
            %
            % See also Sharp.fid_center
            
            if ~isa(img,'cell')
                im_tmp = img;
                img = cell(1);
                img{1} = im_tmp;
            end
            
            roi_size_x = size(img{1},1);
            roi_size_y = size(img{1},2);
            x_px = (-roi_size_x/2:roi_size_x/2-1);
            y_px = (-roi_size_y/2:roi_size_y/2-1);
            
            xc_px = zeros(1,length(img));
            yc_px = zeros(1,length(img));
            for i_m =1:length(img)
                img_x = sum(img{i_m},1)/mean(sum(img{i_m},1));
                img_y = sum(img{i_m},2)/mean(sum(img{i_m},2));
                m_x = img_x>1.5;
                m_y = img_y>1.5;
                xc_px(i_m) = Sharp.centroid(x_px(m_x),img_x(m_x));
                yc_px(i_m) = Sharp.centroid(y_px(m_y),img_y(m_y)');
            end
        end

        
        function [ chr_braille ] = map_braille( chr )
            %MAP_BRAILLE maps a character onto is Braille equivalant
            %   [ chr_braille ] = map_braille( chr )
            %    e.g. map_braille('H') returns 
            %       1 0
            %       1 1
            %       0 0
            
            braille_str = ' A1B\K2L@CIF/MSPxE3H9O6R^DJG>NTQ,*5<-U8V.%[$+X!&;:4\\0Z7(_?W]#Y)=';
            %" A1B'K2L@CIF/MSP\"E3H9O6R^DJG>NTQ,*5<-U8V.%[$+X!&;:4\\0Z7(_?W]#Y)="
            idx = find(braille_str==chr,1);
            
            chr_braille = zeros(3,2);
            if ~isempty(idx)
                bits = dec2bin(hex2dec(strcat('28',dec2hex(idx-1))));
                for k=1:6
                    cur = str2double(bits(end-k+1));
                    chr_braille(k) = cur;
                end
            end
            
            if sum(sum(chr_braille==0))==6
                warning('make sure you are using capital letters')
            end
            
        end


        

        %% Fourier Optics

        function u_out=propTF(u_in,L,lambda,z)
            %SHARP.PROPTF Fourier optics propagation using Transfer Function kernel method
            %   u_out=Sharp.propTF(u_in,L,lambda,z)
            %
            %     propagation - transfer function approach
            %     assumes same x and y side lengths and
            %     uniform sampling
            %     u1 - source plane field
            %     L - source and observation plane side length
            %     lambda - wavelength
            %     z - propagation distance
            %     u2 - observation plane field
            %
            % See also SHARP.PROPIR, SHARP.PROPFF, SHARP.TILT, SHARP.LENS
            
            %December 2012
            %Antoine Wojdyla, CXRO/LBNL awojdyla@lbl.gov
            %adapted from 'Computational Fourier Optics' chap V.2
            
            [M,~]=size(u_in); %get input field array size
            dx=L/M; %sample interval
            k=2*pi/lambda; %wavenumber
            if dx<lambda.*z/L
                warning(sprintf('Fourier Transform propagation kernel is not appropriate (by a factor %1.3f).\nConsider using Impulse Response propagation kernel (propIR) to reduce aliasing\n',lambda.*z/(L*dx)))
            end
            
            fx=-1/(2*dx):1/L:1/(2*dx)-1/L; %freq coords
            [FX,FY]=meshgrid(fx,fx);
            
            H=exp(-1i*pi*lambda*z*(FX.^2+FY.^2)); %trans func
            H=fftshift(H); %shift trans func
            U1=fft2(fftshift(u_in)); %shift, fft src field
            U2=H.*U1; %multiply
            u_out=ifftshift(ifft2(U2)); %inv fft, center obs field
        end
        
        function u_out=propTF1(u_in,L,lambda,z)
            %SHARP.PROPTF Fourier optics propagation using Transfer Function kernel method
            %   u_out=Sharp.propTF(u_in,L,lambda,z)
            %
            %     propagation - transfer function approach
            %     assumes same x and y side lengths and
            %     uniform sampling
            %     u1 - source plane field
            %     L - source and observation plane side length
            %     lambda - wavelength
            %     z - propagation distance
            %     u2 - observation plane field
            %
            % See also SHARP.PROPIR, SHARP.PROPFF, SHARP.TILT, SHARP.LENS
            
            %December 2012
            %Antoine Wojdyla, CXRO/LBNL awojdyla@lbl.gov
            %adapted from 'Computational Fourier Optics' chap V.2
            
            M=length(u_in); %get input field array size
            dx=L/M; %sample interval
            k=2*pi/lambda; %wavenumber
            if dx<lambda.*z/L
                warning(sprintf('Fourier Transform propagation kernel is not appropriate (by a factor %1.3f).\nConsider using Impulse Response propagation kernel (propIR) to reduce aliasing\n',lambda.*z/(L*dx)))
            end
            
            fx=Sharp.fs(dx:dx:L); %freq coords
            
            H=exp(-1i*pi*lambda*z*fx.^2); %trans func
            H=fftshift(H); %shift trans func
            U1=fft(fftshift(u_in)); %shift, fft src field
            U2=H.*U1; %multiply
            u_out=ifftshift(ifft(U2)); %inv fft, center obs field
        end
        
        
        function u2 = propIR(u1,L,lambda,z)
            %SHARP.PROPIR Fourier Optics propagation using Impulse Response kernel method
            %
            %     propagation - impulse response approach
            %     assumes same x and y side lengths and
            %     uniform sampling
            %     u1 - source plane field
            %     L - source and observation plane side length
            %     lambda - wavelength
            %     z - propagation distance
            %     u2 - observation plane field
            %
            % See also SHARP.PROPTF, SHARP.PROPFF, SHARP.TILT, SHARP.LENS
            
            %December 2012
            %Antoine Wojdyla, CXRO/LBNL awojdyla@lbl.gov
            %adapted from 'Computational Fourier Optics' chap V.2
            
            [M,~]=size(u1); %get input field array size
            dx=L/M; %sample interval
            if dx>lambda.*z/L
                fprintf('Impulse Response propagation kernel is not appropriate (by a factor %1.3f).\nConsider using Fourier Transform propagation kernel (propTF) to reduce aliasing\n',1/(lambda.*z/(L*dx)))
            end
            k=2*pi/lambda; %wavenumber
            
            x=-L/2:dx:L/2-dx; %spatial coords
            [X,Y]=meshgrid(x,x);
            
            h=1/(1i*lambda*z)*exp(1i*k/(2*z)*(X.^2+Y.^2)); %impulse
            H=fft2(fftshift(h))*dx^2; %create trans func
            U1=fft2(fftshift(u1)); %shift, fft src field
            U2=H.*U1; %multiply
            u2=ifftshift(ifftshift(ifft2(U2))); %inv fft, center obs field
        end
        
        
        function [uout] = tilt(uin,L,lambda,alpha,theta)
            %SHARP.TILT Tilts the phase modulation equivalent to a tilt
            %   u_out = Sharp.tile(u_in,L,lambda,alpha,theta)
            % where u_in is the inital EM field
            %          L is the screen diameter
            %          lambda is the wavelength (in screen units)
            %          alpha is the azimuthal angle(rad)
            %          theta is the other angle (rad)
            %
            % See also SHARP.LENS, SHARP.PROPTF, SHARP.PROPIR, SHARP.PROPFF
            
            % tilt phasefront
            % uniform sampling assumed
            % uin - input field
            % L - side length
            % lambda - wavelength
            % alpha - tilt angle
            % theta - rotation angle (x axis 0)
            % uout - output field
            
            %from 'Computational Fourier Optics' chap VI.1
            %should be alpha<lambda*.(0.5/dx-B), where B is the bandwidth of the source
            
            [M,N]=size(uin); %get input field array size
            dx=L/M; %sample interval
            k=2*pi/lambda; %wavenumber
            
            x=-L/2:dx:L/2-dx; %coords
            [X,Y]=meshgrid(x,x);
            
            uout=uin.*exp(1i*k*(X*cos(theta)+Y*sin(theta)) *tan(alpha)); %apply tilt
        end
        
        
        function u_out = lens(u_in,L,lambda,zf,diam)
            %SHARP.LENS Creates the phase modulation of a lens
            %   u_out = Sharp.lens(u_in,L,lambda,zf,diam)
            % where u_in is the inital EM field
            %          L is the screen diameter
            %          lambda is the wavelength (in screen units)
            %          zf is the ocal distance  (in screen units)
            %          diam is the diameter of the lens (in screen units)
            %
            % See also SHARP.TILT, SHARP.PROPTF, SHARP.PROPIR, SHARP.PROPFF
            
            % should be zf/diam>dx/lambda
            
            % converging or diverging phase-front
            % uniform sampling assumed
            % uin - input field
            % L - side length
            % lambda - wavelength
            % zf - focal distance (+ converge, - diverge)
            % diam - lens diameter
            % uout - output field
            
            [M,N]=size(u_in); %get input field array size
            dx=L/M; %sample interval
            k=2*pi/lambda; %wavenumber
            %
            x=-L/2:dx:L/2-dx; %coords
            [X,Y]=meshgrid(x,x);
            
            u_out=u_in.*exp(-1i*k/(2*zf)*(X.^2+Y.^2)).*double(sqrt(X.^2+Y.^2)<diam/2); %apply focus
        end


        %% Phase rertrieval


        
        %%
        function [uq, xq_m] = propHF(up, xp_m, xq_m, lambda_m, z_m, option)
            % PROPFH Huygens-Fresnel propagation
            % u_out = propHF(u_in, x_in_m, x_out_m, lambda, z_m)
            % u_out = propHF(u_in, x_in_m, x_out_m, lambda, z_m, 'off')
            %   removes the normalization
            Np = length(xp_m);
            Nq = length(xq_m);
            uq = zeros(1,Nq);
            
            % propagation
            for iq = 1:Nq
                r_m = sqrt((xq_m(iq)+xp_m).^2+z_m.^2);
                uq(iq) = sum(up.*exp(1i*2*pi*r_m/lambda_m)./r_m);
            end
            
            % normalization
            if nargin<6 || ~strcmp(option,'off')
                uq = uq./sqrt(sum(abs(uq).^2));
                uq = uq.*sqrt(sum(abs(up).^2));
            end
        end
        
        function A = polygen(N_points, N_order)
        % Generate polynomilas over[-1;1]
        % A = polygen(Npts, N_order)
            
            x  =linspace(-1,1,N_points);
            A = zeros(N_points,N_order);
            
            for i=1:N_order
                A(:,i) = x.^(i-1);
            end

        end
        
        function Q = gs_basis(A)
        % Gram-Schmidt decomposition    
            [m,n] = size(A);
            
            Q = zeros(m,n);
            R = zeros(n,n);
            for i = 1:n
                %
                v = A(:,i);
                for j = 1:i-1
                    R(j,i) = Q(:,j)'*A(:,i);
                    v = v-R(j,i)*Q(:,j);
                end
                R(i,i) = norm(v);
                Q(:,i) = v/R(i,i);
            end
        end
        
        function [enc, x_px, fraction] = enc_energy(Intensity, size_px)
            %ENC_ENERGY 
            %   enc = enc_energy(intensity_1D)
            %   [enc, x_px, fraction] = enc_energy(Intensity, size_px)
            if size(Intensity,1)>size(Intensity,2)
                I = Intensity';
            else
                I = Intensity;
            end
            
            Npx = length(I);
            
            % if the signal is center-symmetric
            if mod(Npx,2)==0
                enc = [0 cumsum(I((Npx/2+1):Npx)+I((Npx/2):-1:1))]./sum(I);
            else
                enc_left  = ( cumsum(I(((Npx+1)/2): 1:(Npx-1)))+ cumsum( I(((Npx+1)/2+1): 1: Npx)) )/2;
                enc_right = ( cumsum(I(((Npx+1)/2):-1:      2))+ cumsum( I(((Npx+1)/2-1):-1:   1)) )/2;
                enc = [0 enc_left+enc_right];
                enc = enc./sum(I);
            end
            
            %radius
            x_px = 0:(length(enc)-1);
            
            fraction = 1;
            if nargin==2 && size_px<x_px(end)
                fraction  = interp1(x_px,enc,size_px);
            end
            
        end
        
        function [fwhm_px, xl, xr] = fwhm( signal, thr )
            %FHWM determines the FWHM
            %   [fwhm_px, xl, xr] = fwhm( signal, thr )
            
            % error if 2D
            if (size(signal,1)==1 && size(signal,2)==1)
                error('data has to be 1D')
            end
            
            % reshaping if needed
            if size(signal,1)>size(signal,2)
                signal = signal';
            end
            
            % allow a different threshold
            if nargin==1
                thr = max(signal(:)/2);
            end
            
            
            bounds = find(signal>thr);
            % here are the coarse estimate of threshold crossing
            ixl_e = min(bounds);
            ixr_e = max(bounds);
            
            % refine the threasold crossing estimate using
            % explicit linear interpolation
            % left edge
            if ixl_e>1 %make sure there is a left edge
                xl = ixl_e-(signal(ixl_e)-thr)/(signal(ixl_e)-signal(ixl_e-1));
            else %otherwise, pupulate missing edge as NaNs
                xl = NaN;
            end
            % right edge
            if ixr_e<length(signal)
                xr = ixr_e-(signal(ixr_e)-thr)/(signal(ixr_e+1)-signal(ixr_e));
            else
                xr = NaN;
            end
            
            fwhm_px = abs(xr-xl);
        end
        


        %% Aberrations

        %TODO Test and polish
        function Z = Zernikes(W,J)
            % Sharp.Zernike Zernike polynomial generator:
            %   Z = Zernikes(W,J)
            %       W : pixel number
            %       J : order (Noll)
            
            %  Iacopo Mochi
            
            switch J
                case 9
                    J=11;
                case 10
                    J=9;
                case 11
                    J=10;
            end
            
            [x,y] = meshgrid(-1:2/(W-1):1);
            
            r = sqrt(x.^2+y.^2);
            theta = atan2(y,x);
            c = 0;
            n = -1;
            j = 0;
            
            while c==0
                n=n+1;
                for m=0:n
                    d=n-m;
                    if mod(d,2)==0
                        if m>0
                            j=j+2;
                        else
                            j=j+1;
                        end
                    end
                    if j>=J
                        c=1;
                        break
                    end
                end
            end
            
            Z = zeros(W);
            for s=0:(n-m)/2
                Z=Z+r.^(n-2*s).*(-1)^s*factorial(n-s)/...
                    (factorial(s)*factorial((n+m)/2-s)*factorial((n-m)/2-s));
            end
            
            if m > 0
                if mod(J,2)==0
                    Z=Z.*cos(theta*m);
                else
                    Z=Z.*sin(theta*m);
                end
            end
            
            Z( r > 1 )=0;
        end 
        

        
        

        
    end





    
    
    methods (Static, Hidden)
                function [ x_roi, y_roi] = getSweetSpot( img, roi_size_px )
            %SHARP. GETSWEETSPOT Retrieves the sweetspot position from image metadata
            %   [x_roi, y_roi] = sharpGetSweetSpot( img, {roi_size_px} )
            %
            % See also SHARP.ROI
            
            % A Wojdyla, July 2014
            
            % is the first argument a full series or a single image
            if isa(img,'cell')
                img_temp = img{1};
            else
                img_temp = img;
            end
            
            % make sure the provided image has the right size
            if size(img_temp,1)==2048 && size(img_temp,2)==2048
                meta = Sharp.metadata(img_temp);
                sweet = eval(strrep(strrep(meta.image_sweetspot,'(','['),')',']'));
                
                if nargin == 1
                    x_roi = sweet(1);
                    y_roi = 2048-sweet(2);
                elseif nargin==2
                    b_roi = (1:roi_size_px)-(roi_size_px+mod(roi_size_px,2))/2;
                    x_roi = b_roi+sweet(1);
                    % !!! +33 only to account for 5x extension
                    y_roi = 2048+b_roi-sweet(2)+33;
                else
                    error('too many arguments')
                end
            else
                error('The provided image do not have the right size.')
            end
        end
        
        function [path, mask_name, series, number] = parseSharpFilepath(filepath)
            %SHARP.PARSESHARPFILEPATH   Utility to parse common Sharp filepath
            
            [path, file_name] = fileparts(filepath);
            %parts = strsplit(path)
            %cfolder = parts(end-1);
            name_parts = textscan(strrep(file_name,'-',' '),'%s %s %s %s');
            
            mask_name = cell2mat(name_parts{1});
            series =    str2double(cell2mat(name_parts{3}));
            number =    str2double(cell2mat(name_parts{4}));
        end
        
        function bg = getBG (img)
            bg = sum(sum(img(1537:1792,1:256)))/(256.^2);
        end
        
        %BACKWARD COMPATIBILITY
        function [ img_wo_bg , bg ] = removeBG( img )
            [ img_wo_bg , bg ] = Sharp.remove_bg( img );
        end
        
        function img_shifted = spshift( img,x_shift_px, y_shift_px )
            img_shifted = Sharp.spshift2( img,x_shift_px, y_shift_px );
        end
        
        function img_out = removeDC(img_in)
            img_out = Sharp.remove_dc(img_in);
        end
        
        function v_offsets = offAxis_correction(var1, mag, dz_m, cra_deg)
            if nargin == 1
                v_offsets = Sharp.offaxis_correction(var1);
            else
                v_offsets = Sharp.offaxis_correction(var1, mag, dz_m, cra_deg);
            end
        end
        
        
        %FIXME clean up, check dependencies
        function out = abdom_fn(coef)
        %SHARP.ABDOM_FN Generate an aberrated generator using Zernike coeff. 
        %   aberrated_domain = Sharp.abdom_fn(DOMAIN, coefficients)
        %   creates an aberrated domain specified the coefficients vector COEFF
        %   If COEF is a N x 1 vector, ABDOM takes this as coefficients
        %   of Zernike polynomials 0 -> N - 1.  If COEF is a N x 2 vector, the first
        %   column specifies the desired aberration number starting with 0 for
        %   piston, and the second column specifies the magnitude
        %
        % See also Sharp.zgen
        % alt : Sharp.Zernikes
        
        % 8/2012 R Miyakawa
            fnCoef = [];
            [sr, sc] = size(coef);
            switch sc
                case 1
                    for k = 1:sr
                        fnAr{k} = Sharp.zgen([], k-1, 'fn');
                        fnCoef(k) = coef(k);
                    end
                case 2
                    for k = 1:sr
                        fnAr{k} = Sharp.zgen([], coef(k,1), 'fn');
                        fnCoef(k) = coef(k,2);
                    end
                otherwise
                    if sr == 1 % then this is case 1 but vector is transposed:
                        out = abdom(coef');
                    end
            end
            out = @(x,y) fnSum(fnAr, fnCoef, x, y);
        end
        
        function out = fnSum(fnAr, fnCoef, x, y)
            out = 0;
            for k = 1:length(fnAr)
                fnk = fnAr{k};
                out = out + fnCoef(k)*fnk(x,y);
            end
        end



        function out = zgen(domain, order, flag)
        % 8/2012, function out = zgen(domain, order)
        %
        % This function (has been long coming) creates a zernike polynomial over
        % the domain specified by DOMAIN at an arbitrary order ORDER.  If DOMAIN is
        % a single number, it is taken to be a pinhole of resolution specified by
        % that number.  Flag specifies:
        %
        %     pt: computes OUT to be a list of values specified by coordinate pairs
        %     specified by a N x 2 vector inputed as DOMAIN
        % 
        %     fn: returns a handle to a function to compute zernikes over DOMAIN
        if nargin == 2
            flag = 'none';
        end

        switch flag
            case 'pt'
                xind = domain(:,1);
                yind = domain(:,2);
                [TH R] = cart2pol(xind, yind);
                [n, m] = j2nm(order);
                Z = ZgenNM(n, m);
                out = Z(R, TH);
                out(R>1) = 0;
            case 'fn'
                [n, m] = j2nm(order);
                zRTH = ZgenNM(n,m);
                out = @(x,y) zRTH(sqrt(x.^2 + y.^2), atan2(y,x));
            case 'none'
                if length(domain) == 1
                    % assume DOMAIN is specifying resolution
                    domain = pinhole(domain(1));
                end
                [sy, sx] = size(domain);
                xind = linspace(-1, 1, sx);
                yind = linspace(-1, 1, sy);
                [X Y] = meshgrid(xind, yind);
                idx = find(domain);
                [TH R] = cart2pol(X(idx), Y(idx));
                [n, m] = Sharp.j2nm(order);
                Z = Sharp.ZgenNM(n, m);
                z = Z(R, TH);
                out = domain;
                out(idx) = z;
        end
        end


        function Z = ZgenNM(n,m)
            R = Sharp.RgenNM(n,abs(m));
            A = Sharp.AgenNM(m);
            Z = @(r, phi) R(r).*A(phi);
        end

        % Forms radial function based on n,m
        function f = RgenNM(n,m)
            p = (n-m)/2;
            f = @(r) 0;
            for k = 0:p
                f = @(r) f(r) +  (-1)^k*nchoosek(n-k,k)*nchoosek(n-2*k,(n-m)/2-k)*r.^(n - 2*k);
            end
        end

        % Forms azimuthal function based on n,m
        function g = AgenNM(m)
            if m > 0
                g = @(phi) cos(m*phi);
            elseif m < 0
                g = @(phi) -sin(m*phi);
            else
                g = @(phi) 1;
            end
        end

        function [n,m] = j2nm(j)
            % j = j+1; % 0 is piston
            smct = 2;
            ct = 0;
            numInSmct = 3;
            while j > 0
                smct = smct + 2;
                j = j - numInSmct;
                numInSmct = smct + 1;
                ct = ct + 1;
            end
            n = 2*ct;
            m = 0;

            for k = 1:abs(j)
                if mod(k,2)~=0
                    n = n - 1;
                    m = m + 1;
                end
            end
            if mod(abs(j),2)~=0
                m = -m;
            end

        end





        
        function [V,T] = GS_basis(B, domain)
        % 11/2012 Ryan Miyakawa
        % function [V,T] = GS_basis(B, domain)
        %
        % Extends orthobasis to consider an arbitrary basis set of vectors.
        % Returns the basis vectors V and the transformation matrix T to get from
        % the input basis to the orthonormal basis.  Update 11/2012, verified
        % transformation matrix and made algorithm more numerically stable
            if nargin < 2
                domain = ones(size(B{1}));
            end
            
            numV = length(B);
            for k = 1:numV
                if ~all(size(domain) == size(B{k}))
                    error('Domain and basis vectors must all be the same size');
                end
            end
            
            % New GS algorithm from wikipedia:
            V = B;
            for k = 1:length(V)
                kkNorm = 1/sqrt(mipc(V{k}, V{k}, domain));
                V{k} = V{k}*kkNorm;
                
                for m = k+1:length(V)
                    V{m} = V{m} - proj(V{k}, V{m}, domain);
                end
                
            end
            
            % Generate transformation.
            D = zeros(numV);
            for k = 1:length(V)
                for m = 1:k
                    D(k,m) = mipc(B{k}, V{m}, domain);
                end
            end
            T = (inv(D))'; % Ref: page 80 Example 2 of Friedberg, Insel and Spence
            
        end
        
        function prod = mipc(A,B, domain)  %matrix inner product
            if nargin == 2
                B = A;
            end
            prod = sum(sum(A.*conj(B).*domain));
        end
            
        function out = proj(u,v, domain)
            out = u*mipc(u,v,domain)/mipc(u,u,domain);
        end



        
        
        function [orders, VT] = zerndecomp(wave, N, mask, flag, VT)
        %Zerndecomp Ryan Miyakawa, 8/2006
        %Modified 6/01/2007 for display functionality
        %Modified 6/08/2007 to fix scaling issues
        % Modified 11/2012 to use general GS and projection
        %
        %function orders = zerndecomp(wave, N, flag)
        %
        %Decomposes a wavefront into a basis of Zernike polynomials and returns the
        %coeffficients in number of waves over the NA.  Since the discrete zernike functions are not orthogonal,
        %an orthogonal set is constructed using the gram-schmidt algorithm.  The
        %coefficients are matched to this orthogonal basis and then remapped to the
        %Zernikes.  Zerndecom automatically crops the wave input to a circle. The
        %parameter N specifies the number of basis vectors to in the constructed basis set.  It is
        %important to choose an N larger than the highest order Zernike function
        %you are interested in, since that function will not necessarily be linearly
        %independent with lower orders if a smaller N is chosen
        %
        %WAVE is the target wavefront.
        %
        %FLAG can be set to control display behavior and to monitor progress with a waitbar:
        %               a:  display all coefficients
        %               s:  display all non-trivial coefficients (TOL = 1e-8)
        %               aw:  display all coefficients and show waitbar
        %               sw:  display all non-trivial coefficients and show waitbar
        %               w: no display, but show waitbar
        %               no value:  no display
        %
        %Zerndecomp requires functions: orthobasis; To run properly
        
            %% flag logic
            if exist('flag') ~= 1
                flag = 'n';
            end
            if flag(1) == 'w'
                track = 1;
            else
                track = 0;
            end
            if length(flag) == 2 && flag(2) == 'w'
                track = 1;
                flag = flag(1);
            end
            
            if ~strcmp(flag, 'VT')
                zrnOrders = 0:N;
                B = cell(1,length(zrnOrders));
                for k = 1:length(zrnOrders)
                    thisZrnOrder = zrnOrders(k);
                    B{k} = zgen(mask, thisZrnOrder);
                end
                
                [V T] = GS_basis(B, mask);
                VT = {V T};
            end
            
            orders = projectOnBasis(wave, VT, mask, 'VT');
            
            
            if flag == 'a'
                orders = round(1e14*orders)/1e14; %eliminates roe precision inconsistencies
            else
                orders = round(1e8*orders)/1e8; %eliminates roe precision inconsistencies
            end
            
            % display
            if exist('flag') == 1 && (strcmp(flag, 's') || strcmp(flag, 'a'))
                load aberrations.mat %cell array with the names of the aberrations
                str =  ['ABERRATIONS:' sprintf('\n')];
                if flag == 'a'
                    str = ['COMPLETE CHARACTERIZATION:' sprintf('\n')];
                elseif flag == 's'
                    str = ['NONTRIVIAL ABERRATION COEFFICIENTS:' sprintf('\n')];
                end
                for j = 0:N
                    if flag == 's' && abs(orders(j+1)) < 1e-8
                        continue
                    end
                    str = [str  'Z_' num2str(j) ' (' [aberrations{j+1} sprintf('): %0.5g\n', orders(j+1))] ];
                end
                disp(str);
            end
            
            if track
                close(h);
            end
        end


        function [output, Greg] = dftregistration(buf1ft,buf2ft,usfac)
            % function [output Greg] = dftregistration(buf1ft,buf2ft,usfac);
            % Efficient subpixel image registration by crosscorrelation. This code
            % gives the same precision as the FFT upsampled cross correlation in a
            % small fraction of the computation time and with reduced memory
            % requirements. It obtains an initial estimate of the crosscorrelation peak
            % by an FFT and then refines the shift estimation by upsampling the DFT
            % only in a small neighborhood of that estimate by means of a
            % matrix-multiply DFT. With this procedure all the image points are used to
            % compute the upsampled crosscorrelation.
            % Manuel Guizar - Dec 13, 2007
            
            % Portions of this code were taken from code written by Ann M. Kowalczyk
            % and James R. Fienup.
            % J.R. Fienup and A.M. Kowalczyk, "Phase retrieval for a complex-valued
            % object by using a low-resolution image," J. Opt. Soc. Am. A 7, 450-458
            % (1990).
            
            % Citation for this algorithm:
            % Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
            % "Efficient subpixel image registration algorithms," Opt. Lett. 33,
            % 156-158 (2008).
            
            % Inputs
            % buf1ft    Fourier transform of reference image,
            %           DC in (1,1)   [DO NOT FFTSHIFT]
            % buf2ft    Fourier transform of image to register,
            %           DC in (1,1) [DO NOT FFTSHIFT]
            % usfac     Upsampling factor (integer). Images will be registered to
            %           within 1/usfac of a pixel. For example usfac = 20 means the
            %           images will be registered within 1/20 of a pixel. (default = 1)
            
            % Outputs
            % output =  [error,diffphase,net_row_shift,net_col_shift]
            % error     Translation invariant normalized RMS error between f and g
            % diffphase     Global phase difference between the two images (should be
            %               zero if images are non-negative).
            % net_row_shift net_col_shift   Pixel shifts between images
            % Greg      (Optional) Fourier transform of registered version of buf2ft,
            %           the global phase difference is compensated for.
            
            % Default usfac to 1
            if exist('usfac')~=1, usfac=1; end
            
            % Compute error for no pixel shift
            if usfac == 0,
                CCmax = sum(sum(buf1ft.*conj(buf2ft)));
                rfzero = sum(abs(buf1ft(:)).^2);
                rgzero = sum(abs(buf2ft(:)).^2);
                error = 1.0 - CCmax.*conj(CCmax)/(rgzero*rfzero);
                error = sqrt(abs(error));
                diffphase=atan2(imag(CCmax),real(CCmax));
                output=[error,diffphase];
                
                % Whole-pixel shift - Compute crosscorrelation by an IFFT and locate the
                % peak
            elseif usfac == 1,
                [m,n]=size(buf1ft);
                CC = ifft2(buf1ft.*conj(buf2ft));
                [max1,loc1] = max(CC);
                [max2,loc2] = max(max1);
                rloc=loc1(loc2);
                cloc=loc2;
                CCmax=CC(rloc,cloc);
                rfzero = sum(abs(buf1ft(:)).^2)/(m*n);
                rgzero = sum(abs(buf2ft(:)).^2)/(m*n);
                error = 1.0 - CCmax.*conj(CCmax)/(rgzero(1,1)*rfzero(1,1));
                error = sqrt(abs(error));
                diffphase=atan2(imag(CCmax),real(CCmax));
                md2 = fix(m/2);
                nd2 = fix(n/2);
                if rloc > md2
                    row_shift = rloc - m - 1;
                else
                    row_shift = rloc - 1;
                end
                
                if cloc > nd2
                    col_shift = cloc - n - 1;
                else
                    col_shift = cloc - 1;
                end
                output=[error,diffphase,row_shift,col_shift];
                
                % Partial-pixel shift
            else
                
                % First upsample by a factor of 2 to obtain initial estimate
                % Embed Fourier data in a 2x larger array
                [m,n]=size(buf1ft);
                mlarge=m*2;
                nlarge=n*2;
                CC=zeros(mlarge,nlarge);
                CC(m+1-fix(m/2):m+1+fix((m-1)/2),n+1-fix(n/2):n+1+fix((n-1)/2)) = ...
                    fftshift(buf1ft).*conj(fftshift(buf2ft));
                
                % Compute crosscorrelation and locate the peak
                CC = ifft2(ifftshift(CC)); % Calculate cross-correlation
                [max1,loc1] = max(CC);
                [max2,loc2] = max(max1);
                rloc=loc1(loc2);cloc=loc2;
                CCmax=CC(rloc,cloc);
                
                % Obtain shift in original pixel grid from the position of the
                % crosscorrelation peak
                [m,n] = size(CC); md2 = fix(m/2); nd2 = fix(n/2);
                if rloc > md2
                    row_shift = rloc - m - 1;
                else
                    row_shift = rloc - 1;
                end
                if cloc > nd2
                    col_shift = cloc - n - 1;
                else
                    col_shift = cloc - 1;
                end
                row_shift=row_shift/2;
                col_shift=col_shift/2;
                
                % If upsampling > 2, then refine estimate with matrix multiply DFT
                if usfac > 2,
                    %%% DFT computation %%%
                    % Initial shift estimate in upsampled grid
                    row_shift = round(row_shift*usfac)/usfac;
                    col_shift = round(col_shift*usfac)/usfac;
                    dftshift = fix(ceil(usfac*1.5)/2); %% Center of output array at dftshift+1
                    % Matrix multiply DFT around the current shift estimate
                    CC = conj(Sharp.dftups(buf2ft.*conj(buf1ft),ceil(usfac*1.5),ceil(usfac*1.5),usfac,...
                        dftshift-row_shift*usfac,dftshift-col_shift*usfac))/(md2*nd2*usfac^2);
                    % Locate maximum and map back to original pixel grid
                    [max1,loc1] = max(CC);
                    [max2,loc2] = max(max1);
                    rloc = loc1(loc2); cloc = loc2;
                    CCmax = CC(rloc,cloc);
                    rg00 = Sharp.dftups(buf1ft.*conj(buf1ft),1,1,usfac)/(md2*nd2*usfac^2);
                    rf00 = Sharp.dftups(buf2ft.*conj(buf2ft),1,1,usfac)/(md2*nd2*usfac^2);
                    rloc = rloc - dftshift - 1;
                    cloc = cloc - dftshift - 1;
                    row_shift = row_shift + rloc/usfac;
                    col_shift = col_shift + cloc/usfac;
                    
                    % If upsampling = 2, no additional pixel shift refinement
                else
                    rg00 = sum(sum( buf1ft.*conj(buf1ft) ))/m/n;
                    rf00 = sum(sum( buf2ft.*conj(buf2ft) ))/m/n;
                end
                error = 1.0 - CCmax.*conj(CCmax)/(rg00*rf00);
                error = sqrt(abs(error));
                diffphase=atan2(imag(CCmax),real(CCmax));
                % If its only one row or column the shift along that dimension has no
                % effect. We set to zero.
                if md2 == 1,
                    row_shift = 0;
                end
                if nd2 == 1,
                    col_shift = 0;
                end
                output=[error,diffphase,row_shift,col_shift];
            end
            
            % Compute registered version of buf2ft
            if (nargout > 1)&&(usfac > 0),
                [nr,nc]=size(buf2ft);
                Nr = ifftshift([-fix(nr/2):ceil(nr/2)-1]);
                Nc = ifftshift([-fix(nc/2):ceil(nc/2)-1]);
                [Nc,Nr] = meshgrid(Nc,Nr);
                Greg = buf2ft.*exp(1i*2*pi*(-row_shift*Nr/nr-col_shift*Nc/nc));
                Greg = Greg*exp(1i*diffphase);
            elseif (nargout > 1)&&(usfac == 0)
                Greg = buf2ft*exp(1i*diffphase);
            end
            return
        end
            
            function out=dftups(in,nor,noc,usfac,roff,coff)
                % function out=dftups(in,nor,noc,usfac,roff,coff);
                % Upsampled DFT by matrix multiplies, can compute an upsampled DFT in just
                % a small region.
                % usfac         Upsampling factor (default usfac = 1)
                % [nor,noc]     Number of pixels in the output upsampled DFT, in
                %               units of upsampled pixels (default = size(in))
                % roff, coff    Row and column offsets, allow to shift the output array to
                %               a region of interest on the DFT (default = 0)
                % Recieves DC in upper left corner, image center must be in (1,1)
                % Manuel Guizar - Dec 13, 2007
                % Modified from dftus, by J.R. Fienup 7/31/06
                
                % This code is intended to provide the same result as if the following
                % operations were performed
                %   - Embed the array "in" in an array that is usfac times larger in each
                %     dimension. ifftshift to bring the center of the image to (1,1).
                %   - Take the FFT of the larger array
                %   - Extract an [nor, noc] region of the result. Starting with the
                %     [roff+1 coff+1] element.
                
                % It achieves this result by computing the DFT in the output array without
                % the need to zeropad. Much faster and memory efficient than the
                % zero-padded FFT approach if [nor noc] are much smaller than [nr*usfac nc*usfac]
                
                [nr,nc]=size(in);
                % Set defaults
                if exist('roff')~=1, roff=0; end
                if exist('coff')~=1, coff=0; end
                if exist('usfac')~=1, usfac=1; end
                if exist('noc')~=1, noc=nc; end
                if exist('nor')~=1, nor=nr; end
                % Compute kernels and obtain DFT by matrix products
                kernc=exp((-1i*2*pi/(nc*usfac))*( ifftshift([0:nc-1]).' - floor(nc/2) )*( [0:noc-1] - coff ));
                kernr=exp((-1i*2*pi/(nr*usfac))*( [0:nor-1].' - roff )*( ifftshift([0:nr-1]) - floor(nr/2)  ));
                out=kernr*in*kernc;
                return
            end
        
        function metaData = readParams(logfile, filename)
            %METADATA Read Metadata from a csv file
            flog = fopen(logfile,'r');  % Open log file
            img_text = textscan(flog,'%s',4,'delimiter', '\n'); % Read the first four lines
            header = strsplit(img_text{1,1}{4}, ',');  %line 4 is the header line, split in to cell by ','
            img_text = textscan(flog,'%s',inf,'delimiter', '\n');   % read the rest of the file split by carrage return
            fclose(flog);  %close file
            num_files = size(filename,1);
            metaData = cell(num_files, 1);
            
            for s = 1:num_files  %step through image names and construct struct for each image
                img_cell = strfind(img_text{1, 1}, filename{s});
                line_index = find(~cellfun(@isempty,img_cell));  %find the currect line with the image name in it
                img_line = strsplit(img_text{1,1}{line_index},','); %Split line in to cells by ','
                data_struct = struct();
                strings = ['data_systimestring image_comment image_name image_dir'...
                    'image_roi image_poi image_prefix image_sweetspot ma_pattern mask_name mc pattern plan_comment'...
                    'plan_file ptgrey_camname series_type user_name user_org zp_description mc_pattern ']; %field names that should be left as a string
                
                for m = 1:1:size(header, 2)   %construct structure using names in the header
                    name = header{m};
                    
                    if isempty(name)
                        continue
                    end
                    
                    [data_struct(:).(name)] = deal(0);
                    
                    if strfind(strings, char(name)) % check strings to see if value needs to stay a string
                        data_struct.(name) = img_line{m};
                    else
                        data_struct.(name) = str2double(img_line{m}); % if not a string make it a double
                    end
                end
                
                metaData{s} = data_struct;
            end
        end
    end
    
        
    methods(Hidden)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%HANDLE CLASS METHODS THAT SHOULD BE HIDDEN TO MAKE
        %%AUTO-COMPLETION EASIER
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function lh = addlistener(varargin)
            lh = addlistener@handle(varargin{:});
        end
        function notify(varargin)
            notify@handle(varargin{:});
        end
        function delete(varargin)
            delete@handle(varargin{:});
        end
        function Hmatch = findobj(varargin)
            Hmatch = findobj@handle(varargin{:});
        end
        function p = findprop(varargin)
            p = findprop@handle(varargin{:});
        end
        function TF = eq(varargin)
            TF = eq@handle(varargin{:});
        end
        function TF = ne(varargin)
            TF = ne@handle(varargin{:});
        end
        function TF = lt(varargin)
            TF = lt@handle(varargin{:});
        end
        function TF = le(varargin)
            TF = le@handle(varargin{:});
        end
        function TF = gt(varargin)
            TF = gt@handle(varargin{:});
        end
        function TF = ge(varargin)
            TF = ge@handle(varargin{:});
        end
    end
end





