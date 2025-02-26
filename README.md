# EUV_FPM

## **Basic Usage**  

### **Single-Patch FP Reconstruction**  
Run `recon.py` with a selected configuration file from `configs/recon/`:  

1. Choose a configuration file in `configs/recon/`.  
2. Set the `config_name` parameter in `recon.py` to the target configuration file name.  

### **Full-FOV FP Reconstruction (with Automatic Patch Division)**
Run `recon_full.py` for a full-FOV reconstruction using automatically divided patches:

1. Choose a configuration file in `configs/recon_full/`.
2. Set the `config_name` parameter in `recon_full.py` to the target configuration file name.


## **Configuration File Key Fields**  

Below are a few key parameters in the configuration files:  

- **`elliptical_pupil`**:  
  - Set to `True` for anamorphic 0.55/4× NA with 20% central obscuration and 1250× magnification.  
  - Set to `False` for isomorphic 0.33/4× NA with 900× magnification.  

- **`ROI_length`**:  
  - Defines the reconstruction patch size.  

- **`algorithm`**:  
  - Specifies the reconstruction algorithm. Choose from:  
    - `'GS'` – Gerchberg-Saxton 
    - `'GN'` – Gauss-Newton 
    - `'EPFR'` – Embedded Pupil Function Recovery
  

Example datasets are available upon reasonable request. 
Contact: chaoying_gu@berkeley.edu
