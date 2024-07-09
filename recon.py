import numpy as np
import matplotlib as plt
from PIL import Image
import os
import glob
import scipy.io
import time
from utils import *

# take arguments from command line
import argparse
parser = argparse.ArgumentParser(description='WFE correction parameter tuning')
parser.add_argument('--alpha', type=float, default=1, help='WFE correction parameter')
args = parser.parse_args()

folder = './real_data/lines2/'
elliptical_pupil = False                 # Whether to use 0.55 elliptical pupil or 0.33 circular pupil
pre_process = True                      # Preprocess the data by applying spectrum support and filtering
equalization = False                     # Enable for real data
wfe_correction = False                   # k-illumination correction
data_format = 'npz'                    # 'npz' 'mat' 'img'
keyword = 'CD60cb'
roi_size_px = 332*2
use_ROI = True
ROI_length = 332
ROI_center =  [int(roi_size_px/2), int(roi_size_px/2)]
init_option = 'plane'                   # 'zernike' 'plane' 'file'
file_name = f'{folder}gt_abe.npy'

recon_alg = 'GN'                        # 'GS', 'GN', 'EPFR'
iters = 100
swap_dim = False

#%% Set up parameters
# wavelength of acquisition
lambda_m = 13.5e-9

# effective pixel size
dx_m = 15e-9
# effective field size
Dx_m = roi_size_px * dx_m

# spatial scales
x_m = np.arange(1, roi_size_px + 1) * dx_m
y_m = np.arange(1, roi_size_px + 1) * dx_m

# angular frequency scale
fs = 1 / (x_m[1] - x_m[0])
Nfft = len(x_m)
df = fs / Nfft
freq_cpm = np.arange(0, fs, df) - (fs - Nfft % 2 * df) / 2
Fx, Fy = np.meshgrid(freq_cpm, freq_cpm)

if elliptical_pupil:
    fc_lens = (np.arcsin(.55/4)/lambda_m)
    a = fc_lens  # semi-major axis
    b = fc_lens / 2  # semi-minor axis
    FILTER = ((Fx/a)**2 + (Fy/b)**2) <= 1

    # # Take 20% obscuration into account
    a_ob = a*0.2
    b_ob = b*0.2
    FILTER[(Fx/a_ob)**2 + (Fy/b_ob)**2 <= 1] = 0
else:
    fc_lens = (np.arcsin(.33/4)/lambda_m)
    FILTER = (Fx**2 + Fy**2) <= fc_lens**2


#%% Load data
if data_format == 'img':
    # Find all .png files in the folder
    files = glob.glob(os.path.join(folder, f"{keyword}_sx*.png"))
    img = []
    sx = []
    sy = []

    for filename in files:
        img_0 = np.array(Image.open(filename)).astype(float)
        img.append(img_0)
        filename = os.path.basename(filename)
        sx_0 = float(filename[-17:-12])
        # sx_0 = float(filename[-20:-15])
        sy_0 = float(filename[-9:-4])
        sx.append(sx_0)
        sy.append(sy_0)

elif data_format == 'mat':
    path = folder + keyword + '.mat'
    data = scipy.io.loadmat(path)
    img = data['I_low']
    img = [img[:,:,i] for i in range(img.shape[2])]
    na_calib = data['na_calib']
    na_calib = na_calib/(fc_lens*lambda_m)
    sx = [na_calib[i, 0] for i in range(na_calib.shape[0])]
    sy = [na_calib[i, 1] for i in range(na_calib.shape[0])]
    
elif data_format == 'npz':
    path = folder + keyword + '.npz'
    data = np.load(path)
    img = data['imgs']
    img = [img[i,:,:] for i in range(img.shape[0])]
    sx = data['sx']
    sy = data['sy']
    
    
print(f"Reconstructing with {len(img)} images")
    
if pre_process:
    binaryMask = (Fx**2 + Fy**2) <= (2*fc_lens)**2
    
    # Taper the edge to avoid ringing effect
    from scipy.ndimage import gaussian_filter
    xsize, ysize = img[0].shape[:2]
    edgeMask = np.zeros((xsize, ysize))
    pixelEdge = 3
    edgeMask[0:pixelEdge, :] = 1
    edgeMask[-pixelEdge:, :] = 1
    edgeMask[:, 0:pixelEdge] = 1
    edgeMask[:, -pixelEdge:] = 1
    edgeMask = gaussian_filter(edgeMask, sigma=5)
    maxEdge = np.max(edgeMask)
    edgeMask = (maxEdge - edgeMask) / maxEdge


    for i in range(len(img)):
        ftTemp = ft(img[i])
        noiseLevel = max(np.finfo(float).eps, np.mean(np.abs(ftTemp[~binaryMask])))
        ftTemp = ftTemp * np.abs(ftTemp) / (np.abs(ftTemp) + noiseLevel)
        img[i] = ift(ftTemp * binaryMask) * edgeMask

if swap_dim:
    sx, sy = sy, sx
    
if equalization:
    # Equalize each patch
    patch_size = 332
    for im in img:
        # calculate the energy of central patch
        cen_energy = np.sum(img[0][patch_size: 2*patch_size, patch_size: 2*patch_size])
        # equalize the energy of all patches
        for i in range(3):
            for j in range(3):
                im[i*patch_size: (i+1)*patch_size, j*patch_size: (j+1)*patch_size] *= cen_energy / np.sum(
                    im[i*patch_size: (i+1)*patch_size, j*patch_size: (j+1)*patch_size])
    # Equalize the energy of all images
    energy = np.sum(img[0])
    for im in img:
        im *= energy / np.sum(im)

if wfe_correction:
    sx = np.array(sx)
    sy = np.array(sy)
    kx = sx * fc_lens
    ky = sy * fc_lens
    k = 1 / lambda_m
    kz = np.sqrt(k**2 - kx**2 - ky**2)
    phi = np.arccos(kz / k)

    xc_m = (ROI_center[0]-int(roi_size_px/2))*dx_m
    yc_m = (ROI_center[1]-int(roi_size_px/2))*dx_m
    r_m = np.sqrt(xc_m**2 + yc_m**2)
    delta_phi = 2*8*lambda_m/((10*1e-6)**2)*r_m
            
    delta_sx = delta_phi/(fc_lens*lambda_m)*yc_m/r_m
    delta_sy = delta_phi/(fc_lens*lambda_m)*xc_m/r_m
    sx_corrected = sx + args.alpha * delta_sx
    sy_corrected = sy + args.alpha * delta_sy

    sx, sy = sx_corrected, sy_corrected


#%% Crop ROI
if use_ROI:
    print(f'Using ROI of size {ROI_length}')
    x_m = x_m[ROI_center[0]-int(ROI_length/2):ROI_center[0]+int(ROI_length/2)]
    y_m = y_m[ROI_center[1]-int(ROI_length/2):ROI_center[1]+int(ROI_length/2)]
    img = [i[ROI_center[0]-int(ROI_length/2):ROI_center[0]+int(ROI_length/2), 
             ROI_center[1]-int(ROI_length/2):ROI_center[1]+int(ROI_length/2)] for i in img]
    X_full = [sx*fc_lens*Dx_m for sx in sx]
    Y_full = [sy*fc_lens*Dx_m for sy in sy]
    X = [int(x/roi_size_px*ROI_length) for x in X_full]
    Y = [int(y/roi_size_px*ROI_length) for y in Y_full]
     
    roi_size_px = min(ROI_length, roi_size_px)
    Dx_m = ROI_length * dx_m
    Nfft = len(x_m)
    df = fs / Nfft
    freq_cpm = np.arange(0, fs, df) - (fs - Nfft % 2 * df) / 2
    Fx, Fy = np.meshgrid(freq_cpm, freq_cpm)
    FILTER = (Fx**2 + Fy**2) <= fc_lens**2
else:
    X = [int(x*fc_lens*Dx_m) for x in sx]
    Y = [int(y*fc_lens*Dx_m) for y in sy]

lens_init = get_lens_init(FILTER, init_option, file_name)
spectrum_guess = ft(np.sqrt(img[0]))
object_guess = ift(spectrum_guess)
lens_guess = np.complex128(lens_init)

#%% GS Reconstruction
if recon_alg == 'GS':
    N_img = len(img) # assuming img is a list
    start_time = time.time()
    for k in range(iters): # general loop
        for i in range(N_img):
            idx = i

            S_n = object_guess
            S_p = ft(S_n)

            X0 = round(sx[idx]*fc_lens*Dx_m)
            Y0 = round(sy[idx]*fc_lens*Dx_m)

            mask = circshift2(FILTER, X0, Y0)
            aberrated_mask = circshift2(lens_guess, X0, Y0)
            phi_n = aberrated_mask * S_p
            Phi_n = ift(phi_n)
            Phi_np = np.sqrt(img[idx]) * np.exp(1j*np.angle(Phi_n))
            phi_np = ft(Phi_np) # undo aberration

            # S_p[mask] = phi_np[mask]
            step = np.conj(aberrated_mask)/np.max(np.abs(aberrated_mask)**2)*(phi_np-phi_n)
            S_p[mask] = S_p[mask] + step[mask]
            S_np = ift(S_p)
            object_guess = S_np
    end_time = time.time()
    print(f'GS reconstruction took {end_time-start_time} seconds')
#%% GN reconstruction
elif recon_alg == 'GN':
    alpha = 1
    beta = 1000
    step_size = 0.1
    paddingHighRes = 4

    # Upsample
    Np = roi_size_px
    N_obj = roi_size_px * paddingHighRes
    def downsamp(x, cen, Np):
        start_row = cen[0] - np.floor(Np/2)
        end_row = start_row + Np
        start_col = cen[1] - np.floor(Np/2)
        end_col = start_col + Np
        return x[int(start_row):int(end_row), int(start_col):int(end_col)]

    def upsamp(x, N_obj, Np):
        pad_height = ((N_obj - Np) // 2, (N_obj - Np) - ((N_obj - Np) // 2))
        pad_width = pad_height
        return np.pad(x, (pad_height, pad_width), mode='constant')

    O = upsamp(spectrum_guess, N_obj, Np)
    O.max()
    P = lens_guess
    cen0 = [(N_obj+1)//2, (N_obj+1)//2]

    start_time = time.time()
    for _ in range(iters):
        for idx in range(len(img)):
            I_mea = img[idx]
            X0 = X[idx]
            Y0 = Y[idx]
            # flip the order to stay consistent with GS and EPFR
            cen = cen0 - np.array([Y0, X0])
            Psi0 = downsamp(O, cen, Np) * P * FILTER
            psi0 = ift(Psi0)

            I_est = np.abs(psi0) ** 2
            Psi = ft(np.sqrt(I_mea) * np.exp(1j * np.angle(psi0)))
            
            # Projection 2
            dPsi = Psi - Psi0
            Omax = np.abs(O[cen0[0], cen0[1]])

            start_row = cen[0] - np.floor(Np/2)
            end_row = start_row + Np
            start_col = cen[1] - np.floor(Np/2)
            end_col = start_col + Np
        
            O1 = downsamp(O, cen, Np)
            dO = step_size * 1 / np.max(np.abs(P)) * np.abs(P) * np.conj(P) * dPsi / (np.abs(P) ** 2 + alpha)
            O[int(start_row):int(end_row), int(start_col):int(end_col)] += dO
            dP = 1 / Omax * (np.abs(O1) * np.conj(O1)) * dPsi / (np.abs(O1) ** 2 + beta) * FILTER
            P += dP
            
    end_time = time.time()
    object_guess = ift(O)
    lens_guess = P
    print(f'GN reconstruction took {end_time-start_time} seconds')   

#%% EPFR reconstruction
elif recon_alg == 'EPFR':
    alpha = 1
    beta = 1

    N_img = len(img) # assuming img is a list
    start_time = time.time()
    for k in range(iters):
        for i in range(N_img):
            idx = i

            S_n = object_guess
            P_n = lens_guess

            # X0 = round(sx[idx]*fc_lens*Dx_m)
            # Y0 = round(sy[idx]*fc_lens*Dx_m)
            X0 = X[idx]
            Y0 = Y[idx]

            phi_n = P_n * circshift2(ft(S_n), -X0, -Y0)

            Phi_n = ift(phi_n)
            Phi_np = np.sqrt(img[idx]) * np.exp(1j*np.angle(Phi_n))
            phi_np = ft(Phi_np)

            S_np = ift(ft(S_n) + alpha * \
                (np.conj(circshift2(P_n, X0, Y0)) / np.max(np.abs(circshift2(P_n, X0, Y0))**2)) * \
                (circshift2(phi_np, X0, Y0) - circshift2(phi_n, X0, Y0)))

            P_np = P_n + beta * \
                (np.conj(circshift2(ft(S_n), -X0, -Y0)) / np.max(np.abs(circshift2(ft(S_n), -X0, -Y0))**2)) * \
                (phi_np - phi_n)

            object_guess = S_np
            lens_guess = P_np * FILTER
    end_time = time.time()
    print(f'EPFR reconstruction took {end_time-start_time} seconds')
#%% Display
if not os.path.exists(f'{folder}/result'):
    os.makedirs(f'{folder}/result')

np.save(f'{folder}/result/{recon_alg}_recon.npy', object_guess)
np.save(f'{folder}/result/{recon_alg}_abe.npy', np.angle(lens_guess)*FILTER)


plt.imshow(np.abs(object_guess), extent=[x_m[0]*1e9, x_m[-1]*1e9, y_m[0]*1e9, y_m[-1]*1e9], cmap='gray')
plt.xlabel('x position (nm)')
plt.ylabel('y position (nm)')
plt.title(f'{recon_alg} reconstruction (amplitude)')
plt.savefig(f'{folder}/result/obj_recon_amp.png', bbox_inches='tight')

# save with wfr correction parameters
plt.savefig(f'{folder}/result/alpha{args.alpha}.png', bbox_inches='tight')

# save to full resolution image
import imageio
norm_image = np.abs(object_guess)**2
norm_image = (255*norm_image/np.max(norm_image)).astype(np.uint8) 
imageio.imwrite(f'{folder}/result/recon.png', norm_image)


plt.imshow(np.angle(object_guess), extent=[x_m[0]*1e9, x_m[-1]*1e9, y_m[0]*1e9, y_m[-1]*1e9], cmap='gray')
plt.xlabel('x position (nm)')
plt.ylabel('y position (nm)')
plt.title(f'{recon_alg} reconstruction (phase)')
plt.savefig(f'{folder}/result/obj_recon_phase.png', bbox_inches='tight')

plt.imshow(np.abs(lens_guess), cmap='gray')
plt.title(f'{recon_alg} pupil reconstruction (amplitude)')
plt.savefig(f'{folder}/result/pupil_recon_amp.png', bbox_inches='tight')

plt.imshow(np.angle(lens_guess)*FILTER, cmap='gray')
plt.title(f'{recon_alg} pupil reconstruction (phase)')
plt.savefig(f'{folder}/result/pupil_recon_phase.png', bbox_inches='tight')

#%% No high-res upsampling version
# O = spectrum_guess
# P = lens_guess
# cen0 = Np//2


# for _ in range(iters):
#     for idx in range(len(img)):
#         I_mea = img[idx]
#         X0 = round(sx[idx]*fc_lens*Dx_m)
#         Y0 = round(sy[idx]*fc_lens*Dx_m)     
    
#         Psi0 = circshift2(O, -X0, -Y0)*FILTER*P
#         psi0 = ift(Psi0)

#         I_est = np.abs(psi0) ** 2
#         Psi = ft(np.sqrt(I_mea) * np.exp(1j * np.angle(psi0)))
        
#         # Projection 2
#         dPsi = Psi - Psi0
#         Omax = np.abs(O[cen0, cen0])

#         O1 = circshift2(O, -X0, -Y0)*FILTER
#         P = P*FILTER
#         dO = step_size * 1 / np.max(np.abs(P)) * np.abs(P) * np.conj(P) * dPsi / (np.abs(P) ** 2 + alpha)
#         O += circshift2(dO, X0, Y0)
#         dP = 1 / Omax * (np.abs(O1) * np.conj(O1)) * dPsi / (np.abs(O1) ** 2 + beta)
#         P += dP * FILTER