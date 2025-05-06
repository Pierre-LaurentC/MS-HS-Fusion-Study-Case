#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 22:00:13 2025

@author: dpineau
"""

import numpy as np
from numpy.linalg import norm
import os

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nrmse
# from sklearn.metrics import mean_squared_error as mse

main_path = "/home/plcristille/dev/JWST/fusion_inverse/MS-HS-Fusion-Study-Case"
os.chdir(main_path)

from noise_data import get_snr

#%%


# used for conversion of Maps to Hyperspectral Cube
def maps_to_cube(maps, L_specs):  # L_specs.shape = (K, N) = (5, 300)
    new_L_specs = L_specs[..., np.newaxis, np.newaxis]  # (5, 300, 1, 1)
    new_maps = maps[:, np.newaxis, ...]  # (5, 1, 250, 500)
    cube = np.sum(new_L_specs * new_maps, axis=0)
    return cube


# used for the Spectral Angle Mapper
def get_list_of_cube_spectra(cube):  # cube.shape = (lambda, alpha, beta)
    moveaxis_cube = np.moveaxis(cube, source=0, destination=2)
    return np.concatenate(moveaxis_cube, axis=0)  # (alpha * beta, lambda)


# def angle_between_spectra(sp1, sp2):
#     # scalar_product = np.dot(sp1, sp2)
#     scalar_product = np.sum(sp1 * sp2)
#     return np.arccos(scalar_product / (norm(sp1) * norm(sp2)))

# same as MQM, but with hyperspectral cubes (alpha, beta, lambda)
class Maps_Quality_Metrics_2:
    def __init__(self, true_maps, L_specs):
        true_cube = maps_to_cube(true_maps, L_specs)  # (300, 250, 500)
        self.true_cube = true_cube
        self.L_specs = L_specs

        # for data_range in ssim
        self.min_of_true_cube = np.min(true_cube)
        self.max_of_true_cube = np.max(true_cube)

        # for list of true spectra for sam
        self.L_true_spectra = get_list_of_cube_spectra(true_cube)

    def get_mse(self, rec_maps):  # Normalised Root Mean Squared Error, between 0 and 1, the smaller the better
        rec_cube = maps_to_cube(rec_maps, self.L_specs)
        # formula for NMSE (see functions_phd.py by Amine) :
        # return np.sum((self.true_cube - rec_cube) ** 2) / np.sum((self.true_cube) ** 2)
        return nrmse(
            self.true_cube, rec_cube
        )  # scikit implementation of Normalized Root MSE, ORDER OF ARGUMENTS MATTER WITH SKIMAGE !!
    
    # def get_mse(self, rec_maps):  # Mean Squared Error, the smaller the better
    #     rec_cube = maps_to_cube(rec_maps, self.L_specs)
    #     # formula for NMSE (see functions_phd.py by Amine) :
    #     # return np.sum((self.true_cube - rec_cube) ** 2) / np.sum((self.true_cube) ** 2)
    #     return mse(rec_cube, self.true_cube)  # scikit implementation of MSE

    def get_ssim(
        self, rec_maps
    ):  # Structural Similarity Index Measure, from -1 to 1, the higher the better
        rec_cube = maps_to_cube(rec_maps, self.L_specs)
        min_of_rec_cube = np.min(rec_cube)
        max_of_rec_cube = np.max(rec_cube)
        max_data_range = np.max([self.max_of_true_cube, max_of_rec_cube]) - np.min(
            [self.min_of_true_cube, min_of_rec_cube]
        )
        # by default, win_size = 7

        SSIM = ssim(self.true_cube, rec_cube, data_range=max_data_range, channel_axis=0)
        return SSIM

    def get_dssim(
        self, rec_maps
    ):  # Structural DISSimilarity Index Measure, from 0 to 1, the SMALLER the better
        SSIM = self.get_ssim(rec_maps)
        DSSIM = (1 - SSIM) / 2
        return DSSIM
    
    def get_dssim_per_lambda(
            self, rec_maps
            ):
        rec_cube = maps_to_cube(rec_maps, self.L_specs)
        min_of_rec_cube = np.min(rec_cube)
        max_of_rec_cube = np.max(rec_cube)
        max_data_range = np.max([self.max_of_true_cube, max_of_rec_cube]) - np.min(
            [self.min_of_true_cube, min_of_rec_cube]
        )
        
        L_ssim = np.array([ssim(self.true_cube[i], rec_cube[i], data_range=max_data_range, channel_axis=0) for i in range(rec_cube.shape[0])])
        return (1 - L_ssim) / 2
        

    def get_sam(
        self, rec_maps
    ):  # Spectral Angle Mapper, from 0 to pi (= output of arccos), the smaller the better
        rec_cube = maps_to_cube(rec_maps, self.L_specs)
        L_rec_spectra = get_list_of_cube_spectra(rec_cube)
        
        scalar_products = np.sum(
            L_rec_spectra * self.L_true_spectra, axis=1
        )  # shape = 250 * 500 = 125000
        return np.mean(
            np.arccos(
                scalar_products / (norm(L_rec_spectra, axis = 1) * norm(self.L_true_spectra, axis = 1))
            )
        )

    def get_psnr(
        self, rec_maps
    ):  # Peak Signal to Noise Ratio = 10 log (max(rec_maps)^2 / MSE), the higher the better
        rec_cube = maps_to_cube(rec_maps, self.L_specs)
        min_of_rec_cube = np.min(rec_cube)
        max_of_rec_cube = np.max(rec_cube)
        max_data_range = np.max([self.max_of_true_cube, max_of_rec_cube]) - np.min(
            [self.min_of_true_cube, min_of_rec_cube]
        )
        return psnr(self.true_cube, rec_cube, data_range=max_data_range)
    
    def get_snrdb(self, rec_maps):
        rec_cube = maps_to_cube(rec_maps, self.L_specs)
        return get_snr(sig_ref = self.true_cube, sig_nsy = rec_cube, log = True)
    
    def get_mse_per_lambda(self, rec_maps):  # Normalised Root Mean Squared Error, between 0 and 1, the smaller the better
        rec_cube = maps_to_cube(rec_maps, self.L_specs)
        L_mse = []
        for i in range(rec_cube.shape[0]):
            L_mse.append(nrmse(self.true_cube[i], rec_cube[i]))
        return L_mse


class Cube_Quality_Metrics:
    def __init__(self, true_cube):
        self.true_cube = true_cube

        # for data_range in ssim
        self.min_of_true_cube = np.min(true_cube)
        self.max_of_true_cube = np.max(true_cube)

        # for list of true spectra for sam
        self.L_true_spectra = get_list_of_cube_spectra(true_cube)

    def get_mse(self, rec_cube):  # Normalised Root Mean Squared Error, between 0 and 1, the smaller the better
        # formula for NMSE (see functions_phd.py by Amine) :
        # return np.sum((self.true_cube - rec_cube) ** 2) / np.sum((self.true_cube) ** 2)
        return nrmse(
            self.true_cube, rec_cube
        )  # scikit implementation of Normalized Root MSE, ORDER OF ARGUMENTS MATTER WITH SKIMAGE !!
    

    def get_ssim(
        self, rec_cube
    ):  # Structural Similarity Index Measure, from -1 to 1, the higher the better
        min_of_rec_cube = np.min(rec_cube)
        max_of_rec_cube = np.max(rec_cube)
        max_data_range = np.max([self.max_of_true_cube, max_of_rec_cube]) - np.min(
            [self.min_of_true_cube, min_of_rec_cube]
        )
        # by default, win_size = 7

        SSIM = ssim(self.true_cube, rec_cube, data_range=max_data_range, channel_axis=0)
        return SSIM

    def get_dssim(
        self, rec_cube
    ):  # Structural DISSimilarity Index Measure, from 0 to 1, the SMALLER the better
        SSIM = self.get_ssim(rec_cube)
        DSSIM = (1 - SSIM) / 2
        return DSSIM
    
    def get_dssim_per_lambda(
            self, rec_cube
            ):
        min_of_rec_cube = np.min(rec_cube)
        max_of_rec_cube = np.max(rec_cube)
        max_data_range = np.max([self.max_of_true_cube, max_of_rec_cube]) - np.min(
            [self.min_of_true_cube, min_of_rec_cube]
        )
        
        L_ssim = np.array([ssim(self.true_cube[i], rec_cube[i], data_range=max_data_range, channel_axis=0) for i in range(rec_cube.shape[0])])
        return (1 - L_ssim) / 2
        

    def get_sam(
        self, rec_cube
    ):  # Spectral Angle Mapper, from 0 to pi (= output of arccos), the smaller the better
        L_rec_spectra = get_list_of_cube_spectra(rec_cube)
        
        scalar_products = np.sum(
            L_rec_spectra * self.L_true_spectra, axis=1
        )  # shape = 250 * 500 = 125000
        return np.mean(
            np.arccos(
                scalar_products / (norm(L_rec_spectra, axis = 1) * norm(self.L_true_spectra, axis = 1))
            )
        )

    def get_psnr(
        self, rec_cube
    ):  # Peak Signal to Noise Ratio = 10 log (max(rec_maps)^2 / MSE), the higher the better
        min_of_rec_cube = np.min(rec_cube)
        max_of_rec_cube = np.max(rec_cube)
        max_data_range = np.max([self.max_of_true_cube, max_of_rec_cube]) - np.min(
            [self.min_of_true_cube, min_of_rec_cube]
        )
        return psnr(self.true_cube, rec_cube, data_range=max_data_range)
    
    def get_snrdb(self, rec_cube):
        return get_snr(sig_ref = self.true_cube, sig_nsy = rec_cube, log = True)
    
    def get_mse_per_lambda(self, rec_cube):  # Normalised Root Mean Squared Error, between 0 and 1, the smaller the better
        L_mse = []
        for i in range(rec_cube.shape[0]):
            L_mse.append(nrmse(self.true_cube[i], rec_cube[i]))
        return L_mse
