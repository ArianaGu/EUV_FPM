import numpy as np
import matplotlib as plt
from PIL import Image
import os
import glob
import scipy.io
import time
from utils import *
import cv2
from cv2 import seamlessClone


folder = './real_data/full_tile/'
equalization = False                     # Enable for real data
use_mat = False
keyword = 'bprp_abe00'
roi_size_px = 332*5
ROI_length = 332
init_option = 'plane'                   # 'exp' or 'plane'
file_name = f'{folder}/gt_abe.npy'

recon_alg = 'GN'                        # 'GS', 'GN', 'EPFR'
iters = 20
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

if equalization:
    patch_size = 332
    for im in img:
        # calculate the energy of central patch
        cen_energy = np.sum(img[0][patch_size: 2*patch_size, patch_size: 2*patch_size])
        # equalize the energy of all patches
        for i in range(3):
            for j in range(3):
                im[i*patch_size: (i+1)*patch_size, j*patch_size: (j+1)*patch_size] *= np.sum(
                    im[i*patch_size: (i+1)*patch_size, j*patch_size: (j+1)*patch_size])/cen_energy

#%% Crop ROI
def GN_recon(img, X, Y, spectrum_guess, lens_guess, iters):
    alpha = 1
    beta = 1000
    step_size = 0.1
    paddingHighRes = 4

    # Upsample
    Np = img[0].shape[0]
    N_obj = Np * paddingHighRes
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

    for _ in range(iters):
        for idx in range(len(img)):
            I_mea = img[idx]
            X0 = round(X[idx])
            Y0 = round(Y[idx])
            # flip the oder to stay consistent with GS and EPFR
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
            dP = 1 / Omax * (np.abs(O1) * np.conj(O1)) * dPsi / (np.abs(O1) ** 2 + beta)
            P += dP * FILTER
    
    O = downsamp(O, cen0, Np)
    object_guess = ift(O)
    lens_guess = P
    return object_guess, lens_guess


def GS_recon(img, X, Y, object_guess, lens_guess, iters):
    N_img = len(img) # assuming img is a list
    for k in range(iters): # general loop
        for i in range(N_img):
            idx = i

            S_n = object_guess
            S_p = ft(S_n)

            X0 = round(X[idx])
            Y0 = round(Y[idx])

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
    return object_guess, lens_guess


def EPFR_recon(img, X, Y, object_guess, lens_guess, iters):
    alpha = 1
    beta = 1

    N_img = len(img) # assuming img is a list
    for k in range(iters):
        for i in range(N_img):
            idx = i

            S_n = object_guess
            P_n = lens_guess
            
            X0 = round(X[idx])
            Y0 = round(Y[idx])

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
    return object_guess, lens_guess

#%% Reconstruct
patch_num = 6
Dx_m = ROI_length * dx_m
Nfft = ROI_length
df = fs / Nfft
freq_cpm = np.arange(0, fs, df) - (fs - Nfft % 2 * df) / 2
Fx, Fy = np.meshgrid(freq_cpm, freq_cpm)
FILTER = (Fx**2 + Fy**2) <= fc_lens**2
X = [(i*fc_lens*Dx_m) for i in sx]
Y = [(i*fc_lens*Dx_m) for i in sy]
lens_init = get_lens_init(FILTER, init_option, file_name)
lens_guess = np.complex128(lens_init)
object_full = np.zeros((roi_size_px, roi_size_px), dtype=np.complex128)
weight_map = np.zeros((roi_size_px, roi_size_px))
# create 2D Gaussian mask
mask = np.zeros((ROI_length, ROI_length))
# Define Gaussian parameters
sigma_x = sigma_y = ROI_length / 6  # Standard deviation (adjust as needed)
mu_x = mu_y = ROI_length / 2  # Center of the Gaussian
# Create 2D Gaussian mask
y, x = np.indices((ROI_length, ROI_length))
mask = np.exp(-(((x - mu_x) ** 2) / (2 * sigma_x ** 2) + ((y - mu_y) ** 2) / (2 * sigma_y ** 2)))
mask = mask / np.max(mask)

start_time = time.time()

start_pt = int(ROI_length/2)
end_pt = int(roi_size_px-ROI_length/2)
pts = np.linspace(start_pt, end_pt, patch_num, dtype=int)

pupil_array = np.zeros((patch_num, patch_num, ROI_length, ROI_length))

for i, center_x in enumerate(pts):
    for j, center_y in enumerate(pts):
        print(f'Processing patch {i*patch_num+j+1}/{patch_num**2}')
        ROI_center =  [center_x, center_y]
        x_m = x_m[ROI_center[0]-int(ROI_length/2):ROI_center[0]+int(ROI_length/2)]
        y_m = y_m[ROI_center[1]-int(ROI_length/2):ROI_center[1]+int(ROI_length/2)]
        patch_img = [im[ROI_center[0]-int(ROI_length/2):ROI_center[0]+int(ROI_length/2), 
                    ROI_center[1]-int(ROI_length/2):ROI_center[1]+int(ROI_length/2)] for im in img]
        spectrum_guess = ft(np.sqrt(patch_img[0]))
        object_guess = ift(spectrum_guess)
        if recon_alg == 'GS':
            object_guess, lens_guess = GS_recon(patch_img, sx, sy, object_guess, lens_guess, iters)
        elif recon_alg == 'GN':
            object_guess, lens_guess = GN_recon(patch_img, X, Y, spectrum_guess, lens_guess, iters)
        elif recon_alg == 'EPFR':
            object_guess, lens_guess = EPFR_recon(patch_img, X, Y, object_guess, lens_guess, iters)
        else:
            raise ValueError('Invalid reconstruction algorithm') 
        object_full[ROI_center[0]-int(ROI_length/2):ROI_center[0]+int(ROI_length/2),
                    ROI_center[1]-int(ROI_length/2):ROI_center[1]+int(ROI_length/2)] += object_guess * mask
        weight_map[ROI_center[0]-int(ROI_length/2):ROI_center[0]+int(ROI_length/2),
                    ROI_center[1]-int(ROI_length/2):ROI_center[1]+int(ROI_length/2)] += mask
        pupil_array[i, j] = lens_guess

object_full = object_full / weight_map
end_time = time.time()
print(f'Full GN reconstruction took {end_time-start_time} seconds')



np.save(f'{folder}/result/{recon_alg}_full_recon.npy', object_full)    
np.save(f'{folder}/result/{recon_alg}_full_pupil.npy', pupil_array)
        
        






# def blend(patch, full, center, mask, alpha):
#     patch = patch * mask
#     full = full * (1-mask)
#     full[center[0]-int(ROI_length/2):center[0]+int(ROI_length/2), 
#                     center[1]-int(ROI_length/2):center[1]+int(ROI_length/2)] += patch
#     return full

# amp_full = np.zeros((roi_size_px, roi_size_px), dtype=np.uint8)
# phase_full = np.zeros((roi_size_px, roi_size_px), dtype=np.uint8)
# def blend(patch, full, center):
#     # convert to 8bit image
#     patch = np.uint8(patch/np.max(patch)*255)
#     mask = np.zeros_like(full)
#     mask[center[0]-int(ROI_length/2):center[0]+int(ROI_length/2), 
#                     center[1]-int(ROI_length/2):center[1]+int(ROI_length/2)] = 1
#     full = seamlessClone(patch, full, mask, center, cv2.NORMAL_CLONE)
#     return full
#         # blend the amplitude and phase separately
#         amp = np.abs(object_guess)
#         phase = np.angle(object_guess)
#         amp_full = blend(amp, amp_full, ROI_center)
#         phase_full = blend(phase, phase_full, ROI_center)
# object_full = amp_full * np.exp(1j*phase_full)
