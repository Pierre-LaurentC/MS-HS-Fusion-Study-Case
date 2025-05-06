#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 20:05:21 2025

@author: dpineau
"""


# import aljabr
from linop import dottest, fwadjtest
import numpy as np
import matplotlib.pyplot as plt
import time
import os

main_path = "/home/plcristille/dev/JWST/fusion_inverse/MS-HS-Fusion-Study-Case"
# data_path = "/home/dpineau/Bureau/data/mycube"
data_path = main_path + "/data"
output_path = main_path + "/output"

os.chdir(main_path)
from data_retrieval import inputs_for_models, abundance_maps_inputs
from criteria_definitions import QuadCriterion2, optimal_res_dicho_quad2, mu_instru, SemiQuad_Criterion_Fusion, optimal_res_dicho_semiquad2
import plot_tools
import noise_data
from reco_quality_metrics import Maps_Quality_Metrics_2
from save_and_load import Saver
from instrument_models import Spectro_Model_3, Mirim_Model_For_Fusion


# %% IMPORT DES DONNÉES ORIGINALES


(
    psfs_monoch,
    L_pce_mirim,
    L_pce_spectro,
    lamb_cube,
    L_specs,
    decim,
) = inputs_for_models(data_path)
true_maps, _ = abundance_maps_inputs(data_path)

# sous échantillonnage des cartes
sous_ech = 2

true_maps = true_maps[:, ::sous_ech, ::sous_ech]

# réduction de la dimension des cartes en 40x40
origin_area = (40, 153)
size_i = 40
size_j = 40

true_maps = true_maps[:, origin_area[0] : origin_area[0] + size_i, origin_area[1] : origin_area[1] + size_j]
shape_target = true_maps.shape[1:]

# normalisation des cartes entre 0 et 1
true_maps /= np.max(true_maps)
print(true_maps.min(), true_maps.max())

# sous échantillonnage spectral pour réduire la dimension des données

# sous_ech_spec = 10
# sous_ech_spec = 5 # original
sous_ech_spec = 2
# sous_ech_spec = 1
L_specs = L_specs[:, ::sous_ech_spec]
L_specs.shape

lamb_cube = lamb_cube[::sous_ech_spec]

psfs_monoch = psfs_monoch[::sous_ech_spec, :, :]

L_pce_mirim = L_pce_mirim[:, ::sous_ech_spec]
L_pce_spectro = L_pce_spectro[::sous_ech_spec]

# plot spectra templates
# plot_tools.plot_spectra(lamb_cube, L_specs)

# plot one map
# plt.imshow(true_maps2[1])

# plot all maps
# plot_tools.plot_true_maps(true_maps)


#%% CREATION DES MODÈLES INSTRUMENTS

# facteurs de décimations spatiales pour le modèle spectro
di = decim
dj = decim

# =============================================================================
# MIRIM MODEL
# =============================================================================

# reshape mirim model for the fusion model
mirim_model_for_fusion = Mirim_Model_For_Fusion(
    psfs_monoch, L_pce_mirim, lamb_cube, L_specs, shape_target, di, dj
)

# =============================================================================
# SPECTRO MODEL
# =============================================================================

# creation of spectro model from scratch
spectro_model = Spectro_Model_3(
    psfs_monoch, L_pce_spectro, di, dj, lamb_cube, L_specs, shape_target
)

print(dottest(spectro_model))
print(fwadjtest(spectro_model))


#%% DONNÉES INSTRUMENTS BRUITÉES

L_snr = [50]
print(f"SNR = {L_snr[0]}")

y_imager, L_std_mirim = noise_data.make_obs_data_with_snr(mirim_model_for_fusion, true_maps, L_snr[0])
y_spectro, L_std_spectro = noise_data.make_obs_data_with_snr(spectro_model, true_maps, L_snr[0])

mu_imager = mu_instru(L_std_mirim)
mu_spectro = mu_instru(L_std_spectro)

print("y_imager.shape = {}".format(y_imager.shape))
print("y_spectro.shape = {}".format(y_spectro.shape))

# plt.imshow(helpers.var_to_np(y_imager)[7])
# plt.imshow(helpers.var_to_np(y_spectro)[5])

#%% save ms and hs data

saver = Saver(output_path + "/")

saver.save_numpy_file("y_imager", y_imager)
saver.save_numpy_file("y_spectro", y_spectro)


#%% class to evaluate reconstruction quality metrics

MQM = Maps_Quality_Metrics_2(true_maps, L_specs)

#%%

# =============================================================================
# L2 REGULARIZATION METHOD
# =============================================================================

#%% search for optimal reg parameter (mu) wrt best mse

quadcriterion_test = QuadCriterion2(
    mu_imager,
    y_imager,
    mirim_model_for_fusion,
    mu_spectro,
    y_spectro,
    spectro_model,
    0,
    printing = False,
    gradient = "separated"
)

# initial range of search
list_of_mu = [100, 500]
best_mse, best_mu = optimal_res_dicho_quad2(quadcriterion_test, MQM.get_mse, list_of_mu)

# for snr 50 :
# best_mse, best_mu = 0.02156771921471, 201.5625

#%% result from the previous search

best_mu = 201.5625

#%% run solution with the best mu

quadcriterion = QuadCriterion2(
    mu_imager,
    y_imager,
    mirim_model_for_fusion,
    mu_spectro,
    y_spectro,
    spectro_model,
    best_mu,
    printing = False,
    gradient = "separated"
)


# minimization = "explicit"
minimization = "iterative" # like Guilloteau et al. (2020)
# minimization = "iterative_with_metric" # add perf_crit argument to have access to a chosen metric wrt iterations

t1 = time.time()
if minimization == "explicit" :
    quadcrit_rec_maps = quadcriterion.run_expsol()
elif minimization == "iterative" :
    res = quadcriterion.run_lcg(maximum_iterations = 1000)
    quadcrit_rec_maps = res.x
elif minimization == "iterative_with_metric" :
    res = quadcriterion.run_lcg(maximum_iterations = 1000, perf_crit = MQM.get_psnr)
    quadcrit_rec_maps = res.x
    
    plt.plot(quadcriterion.L_perf_crit)
    plt.yscale("log")
print("Time = ", time.time() - t1)

# compare map recontructions
# plot_tools.plot_maps(true_maps, quadcrit_rec_maps)
print("MSE", MQM.get_mse(quadcrit_rec_maps))
print("PSNR", MQM.get_psnr(quadcrit_rec_maps))
print("SSIM", MQM.get_ssim(quadcrit_rec_maps))
print("SAM", MQM.get_sam(quadcrit_rec_maps))



#%%

# =============================================================================
# L2/L1 REGULARIZATION METHOD
# =============================================================================

#%%

# increasing values of maps so that the Huber function works as expected
mult_factor = 100
true_maps_100 = true_maps * mult_factor

y_imager_100, L_std_mirim = noise_data.make_obs_data_with_snr(mirim_model_for_fusion, true_maps_100, L_snr[0])
y_spectro_100, L_std_spectro = noise_data.make_obs_data_with_snr(spectro_model, true_maps_100, L_snr[0])

mu_imager = mu_instru(L_std_mirim)
mu_spectro = mu_instru(L_std_spectro)

MQM_100 = Maps_Quality_Metrics_2(true_maps_100, L_specs)

#%% search for optimal Huber threshold and mu

list_of_mu = [0.02, 0.15]

scale = 0.5
n_iter_max_crit = 1000
L_thresh = [0, 1, 2, 3]
L_best_mu = []
L_best_mse = []
for t in L_thresh:
    print("t", t)
    sq_criterion_test = SemiQuad_Criterion_Fusion(
        mu_imager,
        y_imager_100,
        mirim_model_for_fusion,
        mu_spectro,
        y_spectro_100,
        spectro_model,
        mu_reg = 0,
        thresh = t,
        printing = False,
        scale = scale
    )
    
    best_mse, best_mu = optimal_res_dicho_semiquad2(sq_criterion_test,
                                                                    MQM_100.get_mse,
                                                                    list_of_mu,
                                                                    printing=True,
                                                                    max_iter=30,
                                                                    tolerance=0.01,
                                                                    n_iter_max_crit=n_iter_max_crit,
                                                                    diff_min=1e-20)
    L_best_mu.append(best_mu)
    L_best_mse.append(best_mse)

print(L_thresh)
print(L_best_mu)
print(L_best_mse)

# for SNR 50
# thresh = 0, best_mu = 27.85400390625, best_mse = 0.02154791877347902
# thresh = 1, best_mu = 0.057197265625, best_mse = 0.0202774085831295 # bext combo
# thresh = 2, best_mu = 0.030219726562499993, best_mse = 0.02028042324832927
# thresh = 3, best_mu = 0.019964843750000003, best_mse = 0.020802410751074505

#%% results from previous search

best_thresh = 1
best_mu = 0.057197265625

#%%

# check courbe de MSE

sq_criterion = SemiQuad_Criterion_Fusion(
    mu_imager,
    y_imager_100,
    mirim_model_for_fusion,
    mu_spectro,
    y_spectro_100,
    spectro_model,
    mu_reg = best_mu,
    thresh = best_thresh,
    printing = True,
    scale = 0.5
)

minimization = "iterative"
# minimization = "iterative_with_metric"

t1 = time.time()
if minimization == "iterative" :
    rec_maps_sq = sq_criterion.run_expsol(n_iter_max = 2000, diff_min = 1e-25, calc_crit = False)
elif minimization == "iterative_with_metric" :
    rec_maps_sq = sq_criterion.run_expsol(n_iter_max = 2000, diff_min = 1e-25, calc_crit = False, perf_crit = MQM_100.get_psnr)
    
    L_mse_sq = sq_criterion.L_perf_crit
    plt.plot(L_mse_sq)
    
print(time.time() - t1)


# compare map recontructions
# plot_tools.plot_maps(true_maps, rec_maps_sq)
print("MSE", MQM_100.get_mse(rec_maps_sq))
print("PSNR", MQM_100.get_psnr(rec_maps_sq))
print("SSIM", MQM_100.get_ssim(rec_maps_sq))
print("SAM", MQM_100.get_sam(rec_maps_sq))



#%%

# =============================================================================
# COMPARISON OF RECONSTRUCTIONS
# =============================================================================

#%% calculate reconstructed cube and true cube

# change window of comparison to avoid borders effects
new_origin = (4, 4)
new_size_i = 32
new_size_j = 32

l2_cube = plot_tools.maps_to_cube(quadcrit_rec_maps, L_specs)[:, new_origin[0] : new_origin[0] + new_size_i, new_origin[1] : new_origin[1] + new_size_j]
l21_cube = plot_tools.maps_to_cube(rec_maps_sq, L_specs)[:, new_origin[0] : new_origin[0] + new_size_i, new_origin[1] : new_origin[1] + new_size_j] / mult_factor
true_cube = plot_tools.maps_to_cube(true_maps, L_specs)[:, new_origin[0] : new_origin[0] + new_size_i, new_origin[1] : new_origin[1] + new_size_j]

#%% save cubes

name_file = "l2_cube"
saver.save_numpy_file(name_file, l2_cube)

name_file = "l21_cube"
saver.save_numpy_file(name_file, l21_cube)

name_file = "true_cube"
saver.save_numpy_file(name_file, true_cube)

#%% compare reconstructions

# chosen wavelength
lamb = 120

fig = plt.figure()

n_rows, n_cols = 2, 3
i = 1

plt.subplot(n_rows, n_cols, i)
plt.imshow(y_imager[8, new_origin[0] : new_origin[0] + new_size_i, new_origin[1] : new_origin[1] + new_size_j])
plt.colorbar()
plt.title("MS data")

i += 1
plt.subplot(n_rows, n_cols, i)
plt.imshow(y_spectro[lamb, new_origin[0] // di : new_origin[0] // di + new_size_i // di, new_origin[1] // dj : new_origin[1] // dj + new_size_j // dj])
plt.colorbar()
plt.title("HS data")

i += 1
i += 1
plt.subplot(n_rows, n_cols, i)
plt.imshow(l2_cube[lamb])
plt.colorbar()
plt.title(f"l2 reg. (PSNR = {round(MQM.get_psnr(quadcrit_rec_maps), 2)})")

i += 1
plt.subplot(n_rows, n_cols, i)
plt.imshow(l21_cube[lamb])
plt.colorbar()
plt.title(f"l21 reg. (PSNR = {round(MQM_100.get_psnr(rec_maps_sq), 2)})")

i += 1
plt.subplot(n_rows, n_cols, i)
plt.imshow(true_cube[lamb])
plt.colorbar()
plt.title("Original")


#%% save figure

name_file = "comp_rec"
saver.save_figure(name_file, fig)


