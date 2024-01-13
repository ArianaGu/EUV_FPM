import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import matplotlib.colors as mcolors
from colorsys import hls_to_rgb
import sympy

def zernikeBasis(deg, X, Y):
    m, n = X.shape

    # Create m's and n's from Noll indices
    npol = int(1/2 * (deg + 1) * (deg + 2))
    nn = np.zeros(npol)
    mm = np.zeros(npol)
    pp = np.zeros(npol)
    
    for ii in range(deg+1):
        idxs = np.arange(ii*(ii+1)//2, (ii*(ii+3))//2 + 1)
        nn[idxs] = ii
        mm[idxs] = np.arange(-ii, ii+1, 2)
        pp[idxs] = np.arange(ii, -1, -1)

    R = np.sqrt(X**2 + Y**2)
    T = np.arctan2(Y, X) * (180/np.pi)  # Convert to degrees

    B = np.zeros((m, n, npol))
    ZE = np.zeros((m*n, npol))
    SE = np.zeros((m*n, npol))

    for ii in range(npol):
        ni = nn[ii]
        mi = abs(mm[ii])
        pi = pp[ii]
        N = np.sqrt((2*(ni+1))/(1 + (mi == 0)))

        tmp = np.zeros((m, n))
        for k in range(0, int((ni-mi)//2)+1):
            tmp += (-1)**k * comb(ni-k, k) * comb(ni-2*k, ((ni-mi)//2)-k) * (R**(ni-2*k))

        if mm[ii] > 0:
            tmp = tmp * np.cos(np.deg2rad(mi*T))
        elif mm[ii] < 0:
            tmp = tmp * np.sin(np.deg2rad(mi*T))

        B[:, :, ii] = N * tmp
        ZE[:, ii] = (N * tmp).ravel()
        tmp2 = X**pi * Y**(ni-pi)
        SE[:, ii] = tmp2.ravel()

    Z = np.linalg.lstsq(SE, ZE, rcond=None)[0]
    return B, Z

def compute_pupil_function(z, deg, X, Y):
    B, _ = zernikeBasis(deg, X, Y)
    pupil = np.sum(B[:, :, :len(z)] * z[np.newaxis, np.newaxis, :], axis=2)
    return pupil

def show_aberration(phase, pupil):
    # color map
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # R -> W -> B
    cmap_name = 'custom_diverging'
    cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

    # calculate rms in the pupil region
    masked_values = phase[pupil == 1]
    rms = np.sqrt(np.mean(masked_values**2))
    
    plt.imshow(phase*pupil, cmap=cm, vmin=-2.0*np.pi, vmax=2.0*np.pi)
    cbar = plt.colorbar(label='Phase (radians)')
    cbar.set_ticks([-2.0*np.pi, 0, 2.0*np.pi])
    cbar.set_ticklabels([r'-2.0$\pi$', '0', r'2.0$\pi$'])
    plt.title('Pupil Function')
    plt.annotate(f'RMS = {rms/(2*np.pi):.4f}*2pi', 
                 xy=(0.5, -0.2),  # Adjusted this value to make the label lower
                 xycoords='axes fraction', 
                 ha='center', va='center', 
                 fontsize=12)
        
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('off')
    plt.show()
    
# Given Zernike coefficients, return the pupil function    
def get_pupil_phase(z, pupil):
    rows = np.any(pupil, axis=1)
    cols = np.any(pupil, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    x = np.linspace(-1, 1, cmax-cmin+1)
    y = np.linspace(-1, 1, rmax-rmin+1)
    X, Y = np.meshgrid(x, y)
    local_pupil = np.where(X**2 + Y**2 <= 1, 1, 0)
    abe_pupil = np.zeros_like(pupil).astype(np.float64)
    abe_pupil[rmin:rmax+1, cmin:cmax+1] = compute_pupil_function(z, len(z)-1, X, Y) * local_pupil
    return abe_pupil

# Modified from MIP library
def bprp(Nx_prime, Ny_prime=None):
    """
    Generate a Binary Pseudo-Random Pattern (a.k.a modified Uniformly Redundant Array)
    
    Parameters:
    Nx_prime, Ny_prime: Integers
        Both should be prime numbers.
        
    Returns:
    ura: 2D numpy array
        2D-BPRP of size Nx_prime x Ny_prime
    
    References:
    Fenimore & Cannon "Coded aperture imaging with uniformly redundant arrays"
    dx.doi.org/10.1364/AO.17.000337
    dx.doi.org/10.1364/AO.28.004344
    dx.doi.org/10.1364/OE.22.019803
    """
    
    if Ny_prime is None:
        Ny_prime = Nx_prime

    if not (sympy.isprime(Nx_prime) and sympy.isprime(Ny_prime)):
        raise ValueError('Needs two prime numbers!')

    # basic array
    ba = np.zeros((Nx_prime, Ny_prime))

    # K is associated with Nx_prime and M is associated with Ny_prime.
    
    # a simple method to implement the equations is to evaluate mod(x^2,r) for
    # all x from 1 to r. The resulting values give the locations (I) in Cr
    # that contains +1. All other terms in Cr are -1.
    Cr = np.ones(Nx_prime) * -1
    cr_idx = sorted(set(np.mod(np.arange(1, Nx_prime + 1) ** 2, Nx_prime)))
    Cr[cr_idx] = 1

    Cs = np.ones(Ny_prime) * -1
    cs_idx = sorted(set(np.mod(np.arange(1, Ny_prime + 1) ** 2, Ny_prime)))
    Cs[cs_idx] = 1

    for ix in range(Nx_prime):
        for jy in range(Ny_prime):
            if ix == 0:
                ba[ix, jy] = 0
            elif ix != 0 and jy == 0:
                ba[ix, jy] = 1
            elif Cr[ix] * Cs[jy] == 1:
                ba[ix, jy] = 1
            else:
                ba[ix, jy] = 0

    # positive array
    pa = ba * 2 - 1
    # b[0, 0] has to be equal to 1 so that the sidelobes are flatter:
    pa[0, 0] = 1

    return pa


# centered Fourier Transform
ft = lambda signal: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(signal)))
ift = lambda signal: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(signal)))

# circular shifting
def circshift2(input, x_shift_px, y_shift_px):
    return np.roll(np.roll(input, round(x_shift_px), axis=1), round(y_shift_px), axis=0)

def colorize(z):
    n,m = z.shape
    c = np.zeros((n,m,3))
    c[np.isinf(z)] = (1.0, 1.0, 1.0)
    c[np.isnan(z)] = (0.5, 0.5, 0.5)

    idx = ~(np.isinf(z) + np.isnan(z))
    A = (np.angle(z[idx]) + np.pi) / (2*np.pi)
    A = (A + 0.5) % 1.0
    B = 1.0 - 1.0/(1.0+abs(z[idx])**0.3)
    c[idx] = [hls_to_rgb(a, b, 0.8) for a,b in zip(A,B)]
    return c

# Complex image visualization
def imagecc(img_complex):
    '''
    imagecc(img_complex) displays an image where amplitude is mapped
    on the value and the phase is mapped on the hue.
    img_rgb =imagecc(img_complex) returns the corresponding RGB data,
    '''

    N_row, N_col = img_complex.shape

    map_func = lambda x, channel: np.minimum(6*np.mod(x-(channel+1)/3,1),1) * \
        np.maximum(np.minimum(6*(2/3-np.mod(x-(channel+1)/3,1)),1),0)

    img_abs = np.abs(img_complex.ravel())/(np.max(np.abs(img_complex.ravel())))
    img_arg = np.angle(img_complex.ravel())/(2*np.pi)+0.5

    img_rgb = np.zeros((N_row,N_col,3))

    img_rgb[:,:,0] = np.reshape(map_func(img_arg,1)*np.abs(img_abs),(N_row,N_col))
    img_rgb[:,:,1] = np.reshape(map_func(img_arg,2)*np.abs(img_abs),(N_row,N_col))
    img_rgb[:,:,2] = np.reshape(map_func(img_arg,3)*np.abs(img_abs),(N_row,N_col))

    return img_rgb


def circsum(img_in):
    if img_in.shape[0] != img_in.shape[1]:
        raise ValueError('the input matrix must be square')

    N_px = img_in.shape[0]
    xc = yc = N_px // 2

    X, Y = np.meshgrid(np.arange(N_px), np.arange(N_px))
    dcirc = np.zeros(N_px//2, dtype=complex)
    for i in range(N_px//2):
        domain = ((X-xc)**2+(Y-yc)**2 >= i**2) & ((X-xc)**2+(Y-yc)**2 < (i+1)**2)
        dcirc[i] = np.sum(domain * img_in)

    return dcirc

# Metrics
def frc(img1, img2):
    ft1 = ft(img1)
    ft2 = ft(img2)

    frc_array = circsum(ft1 * np.conj(ft2)) / (np.sqrt(circsum(np.abs(ft1)**2)) * np.sqrt(circsum(np.abs(ft2)**2)))

    n_vox = np.sum(np.ones(img1.shape))
    halfbit_threshold = (0.2071 + 1.9102 / np.sqrt(n_vox)) / (1.2071 + 0.9102 / np.sqrt(n_vox))

    return frc_array, halfbit_threshold


class frc_plot():
    def __init__(self, freq_cpm, roi_size_px):
        self.freq_cpm = freq_cpm
        self.roi_size_px = roi_size_px
    
    def plot_frc_single(self, img1, img2, extra_title=''):
        frc_array, halfbit_threshold = frc(img1, img2)
        plt.figure()
        plt.plot(self.freq_cpm[self.roi_size_px//2:], np.abs(frc_array))
        plt.axhline(halfbit_threshold, color='r', linestyle='--', label='Half-bit threshold')
        plt.title('Fourier ring correlation' + extra_title)
        plt.xlabel('Spatial frequency (nm-1)')
        plt.legend()
        plt.show()
    
    def plot_frc_combined(self, img1_gt, img2_gt, img1_rec, img2_rec, img1_noised_gt, img2_noised_gt, extra_title=''):
        pastel1 = plt.colormaps.get_cmap('Paired')
        plt.figure()

        # Calculate FRC for ground truth vs. reconstruction
        frc_array_gt_vs_rec, halfbit_threshold_gt_vs_rec = frc(img1_gt, img2_gt)
        plt.plot(self.freq_cpm[self.roi_size_px//2:], np.abs(frc_array_gt_vs_rec), label='GT vs. Reconstruction', color=pastel1(0))

        # Calculate FRC for random noised ground truth
        frc_array_noised_gt, halfbit_threshold_noised_gt = frc(img1_noised_gt, img2_noised_gt)
        plt.plot(self.freq_cpm[self.roi_size_px//2:], np.abs(frc_array_noised_gt), label='Noised GT', color=pastel1(1))

        # Calculate FRC for random noised reconstruction
        frc_array_noised_rec, halfbit_threshold_noised_rec = frc(img1_rec, img2_rec)
        plt.plot(self.freq_cpm[self.roi_size_px//2:], np.abs(frc_array_noised_rec), label='Noised Reconstruction', color=pastel1(2))

        # Assuming the half-bit threshold is constant for all, otherwise plot each separately.
        plt.axhline(halfbit_threshold_gt_vs_rec, color='gray', linestyle='--', label='Half-bit threshold')

        plt.title('Fourier Ring Correlation' + extra_title)
        plt.xlabel('Spatial frequency (nm-1)')
        plt.legend()
        plt.show()

def plot_gt_cmp(object_guess, gt_obj, x_m, y_m, freq_cpm, roi_size_px, frc=False):
    ### Amplitude and phase comparison
    plt.figure(figsize=(12, 8))  # Increase the figure size

    plt.subplot(221)
    plt.imshow(np.abs(object_guess), extent=[x_m[0]*1e9, x_m[-1]*1e9, y_m[0]*1e9, y_m[-1]*1e9])
    plt.xlabel('x position (nm)', fontsize=10)
    plt.ylabel('y position (nm)', fontsize=10)
    plt.title('Reconstructed amplitude', fontsize=12)

    plt.subplot(222)
    plt.imshow(np.angle(object_guess), extent=[x_m[0]*1e9, x_m[-1]*1e9, y_m[0]*1e9, y_m[-1]*1e9])
    plt.xlabel('x position (nm)', fontsize=10)
    plt.ylabel('y position (nm)', fontsize=10)
    plt.title('Reconstructed phase', fontsize=12)

    plt.subplot(223)
    plt.imshow(np.abs(gt_obj), extent=[x_m[0]*1e9, x_m[-1]*1e9, y_m[0]*1e9, y_m[-1]*1e9])
    plt.xlabel('x position (nm)', fontsize=10)
    plt.ylabel('y position (nm)', fontsize=10)
    plt.title('Ground truth amplitude', fontsize=12)

    plt.subplot(224)
    plt.imshow(np.angle(gt_obj), extent=[x_m[0]*1e9, x_m[-1]*1e9, y_m[0]*1e9, y_m[-1]*1e9])
    plt.xlabel('x position (nm)', fontsize=10)
    plt.ylabel('y position (nm)', fontsize=10)
    plt.title('Ground truth phase', fontsize=12)

    plt.subplots_adjust(wspace=0.5, hspace=0.3)  # Increase space between subplots
    plt.tight_layout()  # Automatically adjust subplot parameters for better fit
    plt.show()
    
    
    ### Complex object comparison
    plt.figure(figsize=(10,5))  # Increase the figure size

    plt.subplot(121)
    plt.imshow(imagecc(gt_obj), extent=[x_m[0]*1e9, x_m[-1]*1e9, y_m[0]*1e9, y_m[-1]*1e9])
    plt.xlabel('x position (nm)', fontsize=10)
    plt.ylabel('y position (nm)', fontsize=10)
    plt.title('Ground truth object (bandlimited)', fontsize=12)

    plt.subplot(122)
    plt.imshow(imagecc(object_guess), extent=[x_m[0]*1e9, x_m[-1]*1e9, y_m[0]*1e9, y_m[-1]*1e9])
    plt.xlabel('x position (nm)', fontsize=10)
    plt.ylabel('y position (nm)', fontsize=10)
    plt.title('Reconstructed object', fontsize=12)


    # normalization
    im_guess = np.abs(object_guess)**2
    im_gt = np.abs(gt_obj)**2
    im_guess_normalized = im_guess / np.std(im_guess)
    im_gt_normalized = im_gt / np.std(im_gt)
    rmse_normalized = np.sqrt(np.sum((im_guess_normalized-im_gt_normalized)**2))/np.sqrt(np.sum(im_gt_normalized**2))
    log_rmse_normalized = np.log(rmse_normalized)
    plt.suptitle(f'Log RMSE (normalized): {log_rmse_normalized:.4f}', fontsize=12)
    plt.subplots_adjust(wspace=0.5, hspace=0.3)  # Increase space between subplots
    # remove the gap between suptitle and subplots
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    
    plt.figure()
    plt.plot(x_m*1e9, np.abs(im_guess_normalized[160,:]))
    plt.plot(x_m*1e9, np.abs(im_gt_normalized[160,:]))
    plt.title('Slice 160')
    plt.xlabel('x position (nm)')
    plt.ylabel('normalized intensity (a.u.)')
    plt.legend(['image guess', 'ground truth'])
    plt.show()
        
    ### FRC comparison
    if frc:
        img1_noised_gt = im_gt_normalized + 0.1 * np.random.random((roi_size_px, roi_size_px))
        img2_noised_gt = im_gt_normalized + 0.1 * np.random.random((roi_size_px, roi_size_px))
        img1_noised_rec = im_guess_normalized + 0.1 * np.random.random((roi_size_px, roi_size_px))
        img2_noised_rec = im_guess_normalized + 0.1 * np.random.random((roi_size_px, roi_size_px))
        
        # Call the combined plot function
        f = frc_plot(freq_cpm, roi_size_px)
        f.plot_frc_combined(im_guess_normalized, im_gt_normalized,
                        img1_noised_gt, img2_noised_gt,
                        img1_noised_rec, img2_noised_rec,
                        ' (Comparison)')


def zernike_polynomial(n, m, pupil):
    def polar_coords():
        """Generate polar coordinates for a given size"""
        x = np.linspace(-1, 1, cmax-cmin+1)
        y = np.linspace(-1, 1, rmax-rmin+1)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        T = np.arctan2(Y, X)
        T = np.where(T < 0, T + 2*np.pi, T)
        return R, T

    def radial_poly(n, m, R):
        """Calculate the radial polynomial"""
        radial = np.zeros_like(R)
        for s in range((n - abs(m)) // 2 + 1):
            coef = (-1)**s * np.math.factorial(n - s)
            coef /= np.math.factorial(s) * np.math.factorial((n + abs(m)) // 2 - s) * np.math.factorial((n - abs(m)) // 2 - s)
            radial += coef * R**(n - 2 * s)
        return radial

    rows = np.any(pupil, axis=1)
    cols = np.any(pupil, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    R, T = polar_coords()
    Radial = radial_poly(n, m, R)
    if m > 0:
        Z = np.sqrt(2*n+2) * Radial * np.cos(m * T)
    elif m < 0:
        Z = -np.sqrt(2*n+2) * Radial * np.sin(m * T)
    else:
        Z = np.sqrt(n+1) * Radial

    # Pad the Zernike polynomial to match the full pupil size
    padded_Z = np.zeros(pupil.shape)
    padded_Z[rmin:rmax+1, cmin:cmax+1] = Z
    return padded_Z


def get_lens_init(FILTER, option='plane', file_name=""):
    FILTER = np.double(FILTER)
    if option == 'plane':
        lens_init = FILTER
        abe = np.zeros(FILTER.shape)
    elif option == 'zernike':
        defocus_coef = 0.15
        coma_coef = [0.075,0.075]
        defocus = zernike_polynomial(2, 0, FILTER)
        coma1 = zernike_polynomial(3, 1, FILTER)
        coma2 = zernike_polynomial(3, -1, FILTER)
        abe = (defocus_coef*defocus+ coma_coef[0]*coma1 + coma_coef[1]*coma2)*FILTER
        lens_guess = FILTER * np.exp(1j*abe)
        lens_init = lens_guess
    elif option == 'file':
        abe = np.load(file_name)
        lens_init = FILTER * np.exp(1j*abe)
    else:
        raise ValueError('option must be either plane, zernike, or file')
    # show_aberration(abe, FILTER)
    return lens_init