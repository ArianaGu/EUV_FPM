import numpy as np
import scipy.io
import os
import argparse
from utils import *
from PIL import Image


#%% Simulation parameters
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--obj', type=str, default='bprp', choices=['bprp', 'anime'], help='object type')
parser.add_argument('-im', '--illumination_mode', type=str, default='exp', choices=['grid', 'apic', 'exp'], help='illumination mode')
parser.add_argument('-am', '--aberration_mode', type=str, default='exp', choices=['exp', 'plane'], help='aberration mode')
parser.add_argument('-kwd', '--keyword', type=str, default='bprp_abe00', help='used for saving the simulated data')
parser.add_argument('-f', '--folder', type=str, default='./sim_data/', help='folder to save the simulated data')
parser.add_argument('-ip', '--illumination_perturb', type=float, default=0, help='perturbation of the illumination position')
parser.add_argument('-si', '--save_images', action='store_true', help='save the simulated images')

options = parser.parse_args()

#%% SHARP system parameters
# size of the region of interest
roi_size_px = 332*3
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

#%% Load calibrated aberration data
mat = scipy.io.loadmat('./real_data/aberrations_full_FOV_angleCorrected_512_zernike.mat')
aberr_all = mat['FOV_ABERR']

#%% Create simulate object
if options.obj == 'bprp':
    pattern = bprp(17, 17)

    # BPRP size 17*60nm, corresponding to 17*4 pixels
    magnified_pattern = np.kron(pattern+1, np.ones((4, 4)))

    # 2.5um spacing between patterns
    pattern_size = magnified_pattern.shape[0]
    spacing = round(2.5e-6/dx_m) + pattern_size

    obj_bprp = np.zeros((roi_size_px, roi_size_px))
    width = 3*pattern_size + 2*spacing
    shift = (roi_size_px - width) // 2
    for i in range(3):
        for j in range(3):
            start_x = i * (pattern_size + spacing) + shift
            start_y = j * (pattern_size + spacing) + shift
            obj_bprp[start_x:start_x+pattern_size, start_y:start_y+pattern_size] = magnified_pattern
    obj = obj_bprp

elif options.obj == 'anime':
    intensity = Image.open('./sim_data/intensity.png')
    intensity = intensity.resize((roi_size_px, roi_size_px))
    intensity = np.asarray(intensity.convert('L'))/255

    phase = Image.open('./sim_data/phase.png')
    phase = phase.resize((roi_size_px, roi_size_px))
    phase = np.asarray(phase.convert('L'))/255*np.pi

    obj_anime = intensity * np.exp(1j*phase)
    obj = obj_anime

else:
    raise NotImplementedError

#%% Simulate measurement
meas = []
if options.illumination_mode == 'grid':
    Sx = np.linspace(-1,1,9)
    Sy = np.linspace(-1,1,9)
    # create grid scan pairs (sx, sy)
    for sx in Sx:
        for sy in Sy:
            meas.append([sx, sy])
elif options.illumination_mode == 'apic':
    NA_matching_num = 16
    angles = 2*np.pi*np.arange(NA_matching_num)/NA_matching_num
    for angle in angles:
        meas.append([np.cos(angle)*0.99, np.sin(angle)*0.99])
elif options.illumination_mode == 'exp':
    file_name = 'real_data/enlarged_tile/meas.pkl'
    import pickle
    with open(file_name, 'rb') as f:
        meas = pickle.load(f)

folder = options.folder + options.keyword + '/'
if not os.path.exists(folder):
    os.makedirs(folder)

I_low_stack = []
na_calib = []
na_cal = fc_lens*lambda_m   # NA without dimension
na_rp_cal = fc_lens*Dx_m    # NA in pixels

# Simulate aberration at patch [0,0]
abe = get_pupil_phase(aberr_all[0,0],FILTER)

# Manually introduce defocus
defocus_z = np.zeros(10)
defocus_z[4] = 0.5*np.pi
abe_defocus = get_pupil_phase(defocus_z, FILTER)
# abe = abe + abe_defocus

if options.aberration_mode == 'exp':
    abe_FILTER = FILTER*np.exp(1j*abe)
elif options.aberration_mode == 'plane':
    abe_FILTER = FILTER
else:
    raise NotImplementedError
    
for sx, sy in meas:
    sx_exp = sx + options.illumination_perturb*np.random.randn()
    sy_exp = sy + options.illumination_perturb*np.random.randn()
    X0 = round(sx_exp*fc_lens*Dx_m)
    Y0 = round(sy_exp*fc_lens*Dx_m)
    obj_angle = ift(abe_FILTER*circshift2(ft(obj), -X0, -Y0))
    img_angle = np.abs(obj_angle)**2
    # Fix global exposure
    img_angle = img_angle / 4 * 255
    I_low_stack.append(img_angle)
    na_calib.append([sx*fc_lens*lambda_m, sy*fc_lens*lambda_m])
    
    if save_images:
        sx_str = str(sx).zfill(5)
        sy_str = str(sy).zfill(5)
        filename = f'{folder}/sim_{sx_str}_data_{sy_str}.png'
        img_pil = Image.fromarray(np.uint8(img_angle))
        img_pil.save(filename)

np.save(f'{folder}gt.npy', obj)
np.save(f'{folder}/gt_abe.npy', abe)
I_low = np.stack(I_low_stack, axis=2)
na_calib = np.array(na_calib)
freqXY_calib = na_calib*na_rp_cal/na_cal+roi_size_px/2+1 # Shift to the center of the FOV

scipy.io.savemat(f'{folder}/{options.keyword}.mat', {'I_low': I_low, 'freqXY_calib': freqXY_calib, 'na_calib': na_calib,
                                             'na_cal': na_cal, 'na_rp_cal': na_rp_cal, 'gt_CTF': abe_FILTER})