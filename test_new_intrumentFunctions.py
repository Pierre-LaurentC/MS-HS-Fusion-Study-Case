import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.ndimage import zoom
from importlib import reload
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF

import sys
sys.path.append('/home/plcristille/dev/utils')
import plot
from plot import load_dataset_SYNTHETIC_NOISY, load_unmixed_dataset_SYNTHETIC, plot_sampled_images, plot_spec, norm_hypercube, return_mixture, noise_data_for_snr, add_poisson_noise
sys.path.append('/home/plcristille/dev/JWST/fusion_inverse/MS-HS-Fusion-Study-Case')
import instrument_models
from instrument_models import Spectro_Model_3, Mirim_Model_For_Fusion, Mirim_Model_For_Fusion_2, Spectro_Model_3_2, Mirim_Model_For_Fusion_new


sys.path.append('/home/plcristille/dev/JWST/Instrument_func/')
import functions_for_fusion_end_to_end
from functions_for_fusion_end_to_end import Mirim_Model_Cube, Spectro_Model_Cube, Mirim_Model_Cube_for_Tensors, Spectro_Model_Cube_for_Tensors, maps_to_cube, np_to_var, min_not_zero, rescale_0_1, var_to_np


wavelength_NIRSpec = np.load("/home/plcristille/dev/JWST/NIRCam_NIRSpec/NIRSpec_wave.npy")
NIRCam_pce = np.load("/home/plcristille/dev/JWST/NIRCam_NIRSpec/PCE/NIRCam/NIRCam_PCE.npy")
NIRCam_PSF = np.load("/home/plcristille/dev/Webb_PSF/Saved_PSFs/instNIRCam_psfs_pixscale0.15000000000000002_fov6.000000000000001_nb28.npy")
NIRSpec_PSF = np.load("/home/plcristille/dev/Webb_PSF/Saved_PSFs/instNIRSpec_psfs_pixscale0.15000000000000002_fov6.000000000000001_nb1106.npy")
L_pce_NIRSpec = np.load("/home/plcristille/dev/JWST/NIRCam_NIRSpec/PCE/NIRSpec/NIRSpec_PCE.npy")
mixture_specs = np.load("/home/plcristille/dev/JWST/NIRCam_NIRSpec/NIRSpec_spectra/test_signatures.npy")
size = (40,40)
di, dj = 4, 4
margin = 8


x_old = np.linspace(0, 1, mixture_specs.shape[1])
x_new = np.linspace(0, 1, L_pce_NIRSpec.shape[0])

interp_func = interp1d(x_old, mixture_specs, kind='linear', axis=1, fill_value="extrapolate")
mixture_specs = interp_func(x_new)


spat_ss = 2

fname_true_maps = "/home/plcristille/dev/JWST/Instrument_func/Abundance_maps_NEW.fits"
fits_cube = fits.open(fname_true_maps)
true_maps = np.asarray(fits_cube[0].data, dtype=np.float32)[:, ::spat_ss, :: spat_ss]
true_maps.shape

# MODIFYING ABUNDANCE MAP 1

true_maps[0][true_maps[0] > 0.8] = 0.8

# MODIFYING ABUNDANCE MAP 4

n_map = 3

map4 = true_maps[n_map]
# plt.imshow(map4)

d = 20
i1, j1 = 104, 202
# star1 = map4[i1 - d : i1 + d, j1 - d : j1 + d]
i2, j2 = 121, 318
# star2 = map4[i2 - d : i2 + d, j2 - d : j2 + d]
i3, j3 = 113, 345
# star3 = map4[i3 - d : i3 + d, j3 - d : j3 + d]
# star3.shape

mask = np.zeros((2 * d, 2 * d))
mask.shape

# plt.imshow(star3)

map4[i1 - d : i1 + d, j1 - d : j1 + d] = mask
map4[i2 - d : i2 + d, j2 - d : j2 + d] = mask
map4[i3 - d : i3 + d, j3 - d : j3 + d] = mask

# plt.imshow(map4)


# changing values of map 4
map4[map4 <= 0.35] = 0
min_not_zero_map_4 = min_not_zero(map4)
map4[map4 == 0] = min_not_zero_map_4
map4_rescaled = rescale_0_1(map4)

map4_rescaled_blurred = gaussian_filter(map4_rescaled, 1.4)

map4_rerescaled = rescale_0_1(map4_rescaled_blurred)

true_maps[n_map] = map4_rerescaled

shape_target = true_maps.shape[1:]

rect_ld = (310,100,size[0],size[1])

true_maps = true_maps[:, rect_ld[1]:rect_ld[1]+size[1], rect_ld[0]:rect_ld[0]+size[0]]


imager_model_2 = Mirim_Model_For_Fusion_new(
    NIRSpec_PSF, NIRCam_pce, wavelength_NIRSpec, size, mixture_specs.shape[0], di, dj
)

MS_2 = imager_model_2.forward(true_maps, mixture_specs)[:, margin//2:-margin//2, margin//2:-margin//2]
