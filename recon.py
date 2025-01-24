import numpy as np
import matplotlib as plt
from PIL import Image
import os
import glob
import scipy.io
from scipy.ndimage import gaussian_filter
import time
import yaml
from utils import *

# take arguments from command line
import argparse
parser = argparse.ArgumentParser(description='WFE correction parameter tuning')
parser.add_argument('--alpha', type=float, default=1, help='WFE correction parameter')
args = parser.parse_args()

config_path = './configs/recon/BPRP.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Unpack configuration settings
folder = config["folder"]
phase_object = config["phase_object"]
elliptical_pupil = config["elliptical_pupil"]
pupil_edge_soften = config["pupil_edge_soften"]
pre_process = config["pre_process"]
equalization = config["equalization"]
wfe_correction = config["wfe_correction"]
data_format = config["data_format"]
keyword = config["keyword"]
index_to_exclude = config["index_to_exclude"]
roi_size_px = config["roi_size_px"]
use_ROI = config["use_ROI"]
ROI_length = config["ROI_length"]
ROI_center = config["ROI_center"]
init_option = config["init_option"]
file_name = config["file_name"]
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
    a = fc_lens
    b = fc_lens
    FILTER = (Fx**2 + Fy**2) <= fc_lens**2

# soften the edge of the pupil
if pupil_edge_soften > 0:
    dist_map = (Fx / a)**2 + (Fy / b)**2
    boundary_val = 1.0
    d = dist_map - boundary_val
    delta = boundary_val * pupil_edge_soften
    
    # Hard inside vs. outside
    inside_mask = (d < -delta)
    outside_mask = (d > delta)
    transition_mask = ~(inside_mask | outside_mask)

    FILTER = FILTER.astype(float)  # Ensure we can store fractional values
    FILTER[inside_mask] = 1.0
    FILTER[outside_mask] = 0.0

    # Raised-cosine for the transition zone
    # Map d in [-delta, +delta] → t in [0, 1]
    t = (d[transition_mask] + delta) / (2 * delta)  # 0..1
    FILTER[transition_mask] = 0.5 * (1 + np.cos(np.pi * t))
    
    if elliptical_pupil:
        a_ob = a*0.2
        b_ob = b*0.2
        FILTER[(Fx/a_ob)**2 + (Fy/b_ob)**2 <= 1] = 0

plt.figure()
plt.imshow(FILTER, cmap='gray')
plt.axis('off')
plt.savefig(f'{folder}/result/pupil.png', bbox_inches='tight')
    
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
     
    roi_size_px = min(ROI_length, roi_size_px)
    Dx_m = ROI_length * dx_m
    Nfft = len(x_m)
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
else:
    if elliptical_pupil:
        a = fc_lens
        b = fc_lens / 2
        X = [int(x*a*Dx_m) for x in sx]
        Y = [int(y*b*Dx_m) for y in sy]
    else:
        X = [int(x*fc_lens*Dx_m) for x in sx]
        Y = [int(y*fc_lens*Dx_m) for y in sy]


if swap_dim:
    X, Y = Y, X

lens_init = get_lens_init(FILTER, init_option, file_name)
if phase_object is True:
    spectrum_guess = ft(np.exp(1j*np.pi*np.sqrt(img[0])/np.max(np.sqrt(img[0]))))
else:
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

            mask = circshift2(FILTER, X[idx], Y[idx])
            aberrated_mask = circshift2(lens_guess, X[idx], Y[idx])
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
            if abe_correction:
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

ideal_FILTER = FILTER
for i in range(len(img)):
    ideal_FILTER = ideal_FILTER | circshift2(FILTER, -X[i], -Y[i])

scipy.io.savemat(f'{folder}/result/{keyword}_{recon_alg}_recon.mat', 
                    {'obj': object_guess, 
                    'pupil': lens_guess,
                    'ideal_FILTER': ideal_FILTER,
                    }) 

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

plt.imshow(np.abs(lens_guess)*FILTER, cmap='gray')
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