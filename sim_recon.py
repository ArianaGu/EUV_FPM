import numpy as np
import matplotlib as plt
from PIL import Image
import os
import glob
import scipy.io
from utils import *


folder = './sim_data/bprp_abe00/'
keyword = 'bprp_abe00'
roi_size_px = 332*3
use_ROI = True
ROI_length = 256
ROI_center =  [int(roi_size_px/2), int(roi_size_px/2)]
init_option = 'plane'
file_name = f'{folder}/gt_abe.npy'
iters = 50

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
path = folder + keyword + '.mat'
data = scipy.io.loadmat(path)
img = data['I_low']
img = [img[:,:,i] for i in range(img.shape[2])]
na_calib = data['na_calib']
na_calib = na_calib/(fc_lens*lambda_m)
sx = [na_calib[i, 0] for i in range(na_calib.shape[0])]
sy = [na_calib[i, 1] for i in range(na_calib.shape[0])]


#%%
if use_ROI:
    roi_size_px = min(ROI_length, roi_size_px)
    print(f'Using ROI of size {ROI_length}')
    x_m = x_m[ROI_center[0]-int(ROI_length/2):ROI_center[0]+int(ROI_length/2)]
    y_m = y_m[ROI_center[1]-int(ROI_length/2):ROI_center[1]+int(ROI_length/2)]
    img = [i[ROI_center[0]-int(ROI_length/2):ROI_center[0]+int(ROI_length/2), 
             ROI_center[1]-int(ROI_length/2):ROI_center[1]+int(ROI_length/2)] for i in img]
    Dx_m = x_m.max() - x_m.min()
    dx_m = Dx_m / ROI_length
    fs = 1 / dx_m
    Nfft = len(x_m)
    df = fs / Nfft
    freq_cpm = np.arange(0, fs, df) - (fs - Nfft % 2 * df) / 2
    Fx, Fy = np.meshgrid(freq_cpm, freq_cpm)
    FILTER = (Fx**2 + Fy**2) <= fc_lens**2

lens_init = get_lens_init(FILTER, init_option, file_name)

aberrated_FILTER = lens_init
spectrum_guess = ift(np.sqrt(img[0]))
spectrum_guess = spectrum_guess * FILTER
object_guess = ft(spectrum_guess)
lens_guess = np.double(FILTER)

#%%
N_img = len(img) # assuming img is a list

for k in range(50): # general loop
    for i in range(N_img):
        idx = i

        S_n = object_guess
        S_p = ft(S_n)

        X0 = round(sx[idx]*fc_lens*Dx_m)
        Y0 = round(sy[idx]*fc_lens*Dx_m)

        mask = circshift2(FILTER, X0, Y0)
        aberrated_mask = circshift2(aberrated_FILTER, X0, Y0)
        phi_n = aberrated_mask * S_p
        Phi_n = ift(phi_n)
        Phi_np = np.sqrt(img[idx]) * np.exp(1j*np.angle(Phi_n))
        phi_np = ft(Phi_np) # undo aberration

        # S_p[mask] = phi_np[mask]
        step = np.conj(aberrated_mask)/np.max(np.abs(aberrated_mask)**2)*(phi_np-phi_n)
        S_p[mask] = S_p[mask] + step[mask]
        S_np = ift(S_p)
        object_guess = S_np

GS_recon = object_guess
GS_recon_image = np.abs(object_guess)**2


plt.imshow(np.abs(object_guess)**2, extent=[x_m[0]*1e9, x_m[-1]*1e9, y_m[0]*1e9, y_m[-1]*1e9], cmap='gray')
plt.xlabel('x position (nm)')
plt.ylabel('y position (nm)')
plt.title('GS reconstruction')
plt.savefig(f'{folder}/GS_recon.png')
plt.show()