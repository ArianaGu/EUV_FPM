import numpy as np
import matplotlib as plt
from PIL import Image
import os
import glob
import scipy.io
import time
from utils import *


folder = './real_data/enlarged_defect/'
use_mat = False
keyword = 'bprp_abe00'
roi_size_px = 332*3
use_ROI = True
ROI_length = 332
ROI_center =  [int(roi_size_px/2), int(roi_size_px/2)]
init_option = 'plane'                   # 'exp' or 'plane'
file_name = f'{folder}/gt_abe.npy'

recon_alg = 'GS'                        # 'GS', 'GN', 'EPFR'
iters = 100
swap_dim = True

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

# frequency cut-off of the lens (0.33 4xNA lens)
fc_lens = (np.arcsin(.33/4)/lambda_m)
# lens pupil filter in reciprocal space
Fx, Fy = np.meshgrid(freq_cpm, freq_cpm)
FILTER = (Fx**2 + Fy**2) <= fc_lens**2

#%% Load data
if not use_mat:
    # Find all .png files in the folder
    files = glob.glob(os.path.join(folder, "*.png"))
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

else:
    path = folder + keyword + '.mat'
    data = scipy.io.loadmat(path)
    img = data['I_low']
    img = [img[:,:,i] for i in range(img.shape[2])]
    na_calib = data['na_calib']
    na_calib = na_calib/(fc_lens*lambda_m)
    sx = [na_calib[i, 0] for i in range(na_calib.shape[0])]
    sy = [na_calib[i, 1] for i in range(na_calib.shape[0])]
    
print(f"Reconstructing with {len(img)} images")
    
if swap_dim:
    sx, sy = sy, sx

#%% Crop ROI
if use_ROI:
    print(f'Using ROI of size {ROI_length}')
    x_m = x_m[ROI_center[0]-int(ROI_length/2):ROI_center[0]+int(ROI_length/2)]
    y_m = y_m[ROI_center[1]-int(ROI_length/2):ROI_center[1]+int(ROI_length/2)]
    img = [i[ROI_center[0]-int(ROI_length/2):ROI_center[0]+int(ROI_length/2), 
             ROI_center[1]-int(ROI_length/2):ROI_center[1]+int(ROI_length/2)] for i in img]
       
    roi_size_px = min(ROI_length, roi_size_px)
    Dx_m = ROI_length * dx_m
    Nfft = len(x_m)
    df = fs / Nfft
    freq_cpm = np.arange(0, fs, df) - (fs - Nfft % 2 * df) / 2
    Fx, Fy = np.meshgrid(freq_cpm, freq_cpm)
    FILTER = (Fx**2 + Fy**2) <= fc_lens**2
    
    X = [(i*fc_lens*Dx_m) for i in sx]
    Y = [(i*fc_lens*Dx_m) for i in sy]
    
else:
    X = [(i*fc_lens*Dx_m) for i in sx]
    Y = [(i*fc_lens*Dx_m) for i in sy]
    
lens_init = get_lens_init(FILTER, init_option, file_name)
spectrum_guess = ft(np.sqrt(img[0]))
object_guess = ift(spectrum_guess)
lens_guess = np.complex128(FILTER)

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
            X0 = round(X[idx])
            Y0 = round(Y[idx])
            cen = cen0 - np.array([X0, Y0])
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
            dP = 1 / Omax * (np.abs(O1) * np.conj(O1)) * dPsi / (np.abs(O1) ** 2 + beta)
            P += dP * FILTER
            
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

            X0 = round(sx[idx]*fc_lens*Dx_m)
            Y0 = round(sy[idx]*fc_lens*Dx_m)

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


plt.imshow(np.abs(object_guess), extent=[x_m[0]*1e9, x_m[-1]*1e9, y_m[0]*1e9, y_m[-1]*1e9], cmap='gray')
plt.xlabel('x position (nm)')
plt.ylabel('y position (nm)')
plt.title(f'{recon_alg} reconstruction (amplitude)')
plt.savefig(f'{folder}/result/obj_recon_amp.png', bbox_inches='tight')

plt.imshow(np.angle(object_guess), extent=[x_m[0]*1e9, x_m[-1]*1e9, y_m[0]*1e9, y_m[-1]*1e9], cmap='gray')
plt.xlabel('x position (nm)')
plt.ylabel('y position (nm)')
plt.title(f'{recon_alg} reconstruction (phase)')
plt.savefig(f'{folder}/result/obj_recon_phase.png', bbox_inches='tight')

plt.imshow(np.abs(lens_guess), cmap='gray')
plt.title(f'{recon_alg} pupil reconstruction (amplitude)')
plt.savefig(f'{folder}/result/pupil_recon_amp.png', bbox_inches='tight')

plt.imshow(np.angle(lens_guess), cmap='gray')
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