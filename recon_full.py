import numpy as np
import matplotlib as plt
from PIL import Image
import os
import glob
import scipy.io
import yaml
import argparse
import time
from utils import *
# import cv2
# from cv2 import seamlessClone


config_path = './configs/recon_full/BPRA0.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)  # returns a dict
# Unpack config
folder = config["folder"]
phase_object = config["phase_object"]
elliptical_pupil = config["elliptical_pupil"]
pre_process = config["pre_process"]
equalization = config["equalization"]
wfe_correction = config["wfe_correction"]
wfe_alpha = config["wfe_alpha"]
data_format = config["data_format"]
keyword = config["keyword"]
index_to_exclude = config["index_to_exclude"]
roi_size_px = config["roi_size_px"]
init_option = config["init_option"]
file_name = config["file_name"]
ROI_length = config["ROI_length"]
patch_num = config["patch_num"]
recon_alg = config["recon_alg"]
abe_correction = config["abe_correction"]
iters = config["iters"]
swap_dim = config["swap_dim"]


#%% Set up parameters
# wavelength of acquisition
lambda_m = 13.5e-9

if elliptical_pupil:
    dx_m = 10.8e-9
else:
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
    
if index_to_exclude is not None:
    img = [img[i] for i in range(len(img)) if i+1 not in index_to_exclude]
    sx = [sx[i] for i in range(len(sx)) if i+1 not in index_to_exclude]
    sy = [sy[i] for i in range(len(sy)) if i+1 not in index_to_exclude]

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

    
if equalization:
    # Equalize each patch
    patch_size = 332
    for im in img:
        # calculate the energy of central patch
        cen_energy = np.sum(img[0][patch_size: 2*patch_size, patch_size: 2*patch_size])
        # equalize the energy of all patches
        N_patch = roi_size_px // patch_size
        for i in range(N_patch):
            for j in range(N_patch):
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
    center = [roi_size_px//2, roi_size_px//2]
    xc_m = (center[0]-int(roi_size_px/2))*dx_m
    yc_m = (center[1]-int(roi_size_px/2))*dx_m
    r_m = np.sqrt(xc_m**2 + yc_m**2)
    delta_phi = 2*8*lambda_m/((10*1e-6)**2)*r_m
            
    delta_sx = delta_phi/(fc_lens*lambda_m)*yc_m/r_m
    delta_sy = delta_phi/(fc_lens*lambda_m)*xc_m/r_m
    sx_corrected = sx + wfe_alpha * delta_sx
    sy_corrected = sy + wfe_alpha * delta_sy

    sx, sy = sx_corrected, sy_corrected


if elliptical_pupil:
    a = fc_lens  # semi-major axis
    b = fc_lens / 2  # semi-minor axis
    X_full = [sx*a*Dx_m for sx in sx]
    Y_full = [sy*b*Dx_m for sy in sy]
else:
    X_full = [sx*fc_lens*Dx_m for sx in sx]
    Y_full = [sy*fc_lens*Dx_m for sy in sy]
X = [int(x/roi_size_px*ROI_length) for x in X_full]
Y = [int(y/roi_size_px*ROI_length) for y in Y_full]
    

Dx_m = ROI_length * dx_m
Nfft = ROI_length
df = fs / Nfft
freq_cpm = np.arange(0, fs, df) - (fs - Nfft % 2 * df) / 2
Fx, Fy = np.meshgrid(freq_cpm, freq_cpm)
if elliptical_pupil:
    fc_lens = (np.arcsin(.55/4)/lambda_m)
    a = fc_lens
    b = fc_lens / 2
    FILTER = ((Fx/a)**2 + (Fy/b)**2) <= 1
    a_ob = a*0.2
    b_ob = b*0.2
    FILTER[(Fx/a_ob)**2 + (Fy/b_ob)**2 <= 1] = 0
else:
    fc_lens = (np.arcsin(.33/4)/lambda_m)
    FILTER = (Fx**2 + Fy**2) <= fc_lens**2

print('FILTER shape:', FILTER.shape)

if swap_dim:
    X, Y = Y, X


#%% Crop ROI
def GN_recon(patch_img, X, Y, spectrum_guess, lens_guess, iters):
    alpha = 1
    beta = 1000
    step_size = 0.1
    paddingHighRes = 4

    # Upsample
    Np = patch_img[0].shape[0]
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
        for idx in range(len(patch_img)):
            I_mea = patch_img[idx]
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
            if abe_correction:
                dP = 1 / Omax * (np.abs(O1) * np.conj(O1)) * dPsi / (np.abs(O1) ** 2 + beta) * FILTER
                P += dP
    
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
lens_init = get_lens_init(FILTER, init_option, file_name)
lens_init = np.complex128(lens_init)

# create 2D Gaussian mask
mask = np.zeros((ROI_length, ROI_length))
# Define Gaussian parameters
sigma_x = sigma_y = ROI_length / 4  # Standard deviation (adjust as needed)
mu_x = mu_y = ROI_length / 2  # Center of the Gaussian
# Create 2D Gaussian mask
y, x = np.indices((ROI_length, ROI_length))
mask = np.exp(-(((x - mu_x) ** 2) / (2 * sigma_x ** 2) + ((y - mu_y) ** 2) / (2 * sigma_y ** 2)))
mask = mask / np.max(mask)

start_pt = int(ROI_length/2)
end_pt = int(roi_size_px-ROI_length/2)
pts = np.linspace(start_pt, end_pt, patch_num, dtype=int)

object_full = np.zeros((roi_size_px, roi_size_px), dtype=np.complex128)
weight_map = np.zeros((roi_size_px, roi_size_px))
pupil_array = np.zeros((patch_num, patch_num, ROI_length, ROI_length), dtype=np.complex128)
lens_guess = lens_init.copy()

def spiral_pts(pts):
    """
    Given a 1D array of coordinates (pts), construct a 2D grid
    and return the (x,y) patch centers in an order that goes
    from the center outward in a spiral-like pattern.
    """
    N = len(pts)
    cx = (N - 1) / 2.0
    cy = (N - 1) / 2.0
    
    coords = []
    for i in range(N):
        for j in range(N):
            r = np.hypot(i - cx, j - cy)
            theta = np.arctan2(j - cy, i - cx)
            coords.append((r, theta, i, j))
    
    coords.sort(key=lambda x: (x[0], x[1]))
    
    spiral_order = []
    for (_, _, i, j) in coords:
        spiral_order.append((i, j, pts[i], pts[j]))
    
    return spiral_order


# Get the spiral (center-out) sequence of (x,y) centers
spiral_list = spiral_pts(pts)

start_time = time.time()
for n, (i, j, center_x, center_y) in enumerate(spiral_list):
    print(f"Processing patch {n+1}/{patch_num**2}: (i,j)=({i},{j}), ROI=({center_x},{center_y})")
    ROI_center = [center_x, center_y]
    
    patch_img = [
        im[ROI_center[0]-int(ROI_length/2):ROI_center[0]+int(ROI_length/2), 
        ROI_center[1]-int(ROI_length/2):ROI_center[1]+int(ROI_length/2)]
        for im in img
    ]

    spectrum_guess = ft(np.sqrt(patch_img[0]))
    object_guess = ift(spectrum_guess)
    if recon_alg == 'GS':
        object_guess, lens_guess = GS_recon(patch_img, sx, sy, object_guess, lens_init.copy(), iters)
    elif recon_alg == 'GN':
        object_guess, lens_guess = GN_recon(patch_img, X, Y, spectrum_guess, lens_init.copy(), iters)
    elif recon_alg == 'EPFR':
        object_guess, lens_guess = EPFR_recon(patch_img, X, Y, object_guess, lens_init.copy(), iters)
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

ideal_FILTER = FILTER
for i in range(len(img)):
    ideal_FILTER = ideal_FILTER | circshift2(FILTER, -X[i], -Y[i])

scipy.io.savemat(f'{folder}/result/{keyword}_{recon_alg}_full_recon.mat', 
                    {'obj': object_full, 
                    'pupil_array': pupil_array,
                    'ideal_FILTER': ideal_FILTER,
                    }) 
        



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
