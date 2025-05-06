#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 22:21:06 2025

@author: dpineau
"""

import numpy as np
import os

from udft import laplacian, irdftn, rdft2, ir2fr, diff_ir, dftn, idft2, dft2
from einops import einsum, rearrange, reduce, repeat

main_path = "/home/plcristille/dev/JWST/fusion_inverse/MS-HS-Fusion-Study-Case"
os.chdir(main_path)

from instrument_models import concatenating2, Spectro_Model_3, partitioning_einops2, Mirim_Model_For_Fusion, make_iHtH_spectro



#%% INPUTS FOR MODELS

# psfs_monoch, L_pce_mirim, L_pce_spectro, lamb_cube, L_specs, decim = input_templates.inputs_for_models_1_2()
# true_maps, shape_target = input_templates.abundance_maps_inputs_1_2()

#%%

# input in freq and not part, output in freq
def apply_hessian_freq(hess_spec_freq, di, dj, shape_target, x_freq):
    # partitionnement de x
    part_x_freq = partitioning_einops2(x_freq, di, dj)  # (5, 25, 50, 100)

    # produit de HtH avec x
    HtH_x_freq = hess_spec_freq * part_x_freq[np.newaxis, :, np.newaxis, :, :, :]
    # (5, 5, 25, 25, 50, 100) * (1, 5, 1, 25, 50, 100) = (5, 5, 25, 25, 50, 100)

    HtH_x_freq_sum = einsum(HtH_x_freq, "ti tj di dj h w -> ti di h w")

    # reconstitution des cartes en freq
    concat_HtH_x_freq = concatenating2(HtH_x_freq_sum, shape_target, di, dj)
    # (5, 25, 50, 100) --> (5, 250, 500) --> (5, 250, 251)

    return concat_HtH_x_freq

# separation of classes for Regul and Inversion because Regul without inversion will be useful for iterative methods        
class Regul_Fusion_Model2():
    def __init__(self, mirim_model_for_fusion:Mirim_Model_For_Fusion, spectro_model:Spectro_Model_3, L_mu_reg, mu_mirim, mu_spectro, gradient="joint"):
        # print("Use of class {}.".format("Regul_Fusion_Model"))
        # making hessian for fusion
        hessian_fusion = mu_mirim * mirim_model_for_fusion.part_hess_mirim_freq_full + mu_spectro * spectro_model.hess_spec_freq
        
        # adding regularization
        shape_target = spectro_model.shape_target
        if gradient == "joint":
            diff_kernel = laplacian(2)
            D_freq = ir2fr(diff_kernel, shape=shape_target, real=False)
            part_D_freq = partitioning_einops2(D_freq[np.newaxis, ...], spectro_model.di, spectro_model.dj)[0]
        
            regul_hess_fusion = np.copy(hessian_fusion)
            n_spec, _, di_times_dj, _, h_block, w_block = regul_hess_fusion.shape
            for k in range(n_spec):
                for i in range(di_times_dj):
                    regul_hess_fusion[k, k, i, i, :, :] += (
                        L_mu_reg[k] * (np.abs(part_D_freq[i]) ** 2)
                    )
                    
        elif gradient == "separated":
            diff_kernel_row = (np.array([-1, 1]))[..., np.newaxis]
            diff_kernel_col = (np.array([-1, 1]))[np.newaxis, ...]

            D_freq_row = ir2fr(diff_kernel_row, shape=shape_target, real=False)
            D_freq_col = ir2fr(diff_kernel_col, shape=shape_target, real=False)
            
            # partitioning
            part_D_freq_row = partitioning_einops2(D_freq_row[np.newaxis, ...], spectro_model.di, spectro_model.dj)[0]
            part_D_freq_col = partitioning_einops2(D_freq_col[np.newaxis, ...], spectro_model.di, spectro_model.dj)[0]
            
            # summing on diagonal of hessian
            regul_hess_fusion = np.copy(hessian_fusion)
            n_spec, _, di_times_dj, _, h_block, w_block = regul_hess_fusion.shape
            for k in range(n_spec):
                for i in range(di_times_dj):
                    coeff = L_mu_reg[k]
                    regul_hess_fusion[k, k, i, i, :, :] += (
                        coeff * (np.abs(part_D_freq_row[i])**2 + np.abs(part_D_freq_col[i])**2)
                    )
                
        self.regul_hess_fusion = regul_hess_fusion
        self.di = spectro_model.di
        self.dj = spectro_model.dj
        self.shape_target = shape_target
        
        self.mirim_model_for_fusion = mirim_model_for_fusion
        self.mu_mirim = mu_mirim
        
        self.spectro_model = spectro_model
        self.mu_spectro = mu_spectro
        

class Inv_Regul_Fusion_Model2():
    def __init__(self, regul_fusion_model:Regul_Fusion_Model2):
        # print("Use of class {}.".format("Inv_Regul_Fusion_Model"))
        # make inverse of hessian        
        inv_hess_fusion = make_iHtH_spectro(regul_fusion_model.regul_hess_fusion)
        self.inv_hess_fusion = inv_hess_fusion
        
        # real_inv_hessian = idft2(inv_hess_fusion)
        # inv_hess_fusion_half = rdft2(real_inv_hessian)
        # self.inv_hess_fusion_half = inv_hess_fusion_half
        
        self.mirim_model_for_fusion = regul_fusion_model.mirim_model_for_fusion
        self.mu_mirim = regul_fusion_model.mu_mirim
        
        self.spectro_model = regul_fusion_model.spectro_model
        self.mu_spectro = regul_fusion_model.mu_spectro
        
        self.di = regul_fusion_model.di
        self.dj = regul_fusion_model.dj
        
        self.shape_target = regul_fusion_model.shape_target
        
    def map_reconstruction(self, mirim_obs_data, spectro_obs_data):
        # mirim_adjoint_data = self.mirim_model_for_fusion.adjoint_freq_full(mirim_obs_data)
        # spectro_adjoint_data = self.spectro_model.adjoint_freq_full(spectro_obs_data)
        # adjoint_data_freq = self.mu_mirim * mirim_adjoint_data + self.mu_spectro * spectro_adjoint_data
        
        # t1 = time.time()
        
        mirim_adjoint_data = self.mirim_model_for_fusion.adjoint(mirim_obs_data)
        spectro_adjoint_data = self.spectro_model.adjoint(spectro_obs_data)
        adjoint_data_freq = dft2(self.mu_mirim * mirim_adjoint_data + self.mu_spectro * spectro_adjoint_data)
        
        # t2 = time.time()
        
        res = np.real(idft2(apply_hessian_freq(self.inv_hess_fusion, self.di, self.dj, self.shape_target, adjoint_data_freq)))
        
        # t3 = time.time()
        
        # print("map_reconstruction : temps calcul de b =", t2 - t1)
        # print("map_reconstruction : temps calcul de Q-1 b =", t3 - t2)
        
        return res 
    

#%%


# separation of classes for Regul and Inversion because Regul without inversion will be useful for iterative methods        
class Regul_Fusion_Model3():
    def __init__(self, mirim_model_for_fusion:Mirim_Model_For_Fusion, spectro_model:Spectro_Model_3, L_mu_reg, mu_mirim, mu_spectro, scale):
        # making hessian for fusion
        hessian_fusion = mu_mirim * mirim_model_for_fusion.part_hess_mirim_freq_full + mu_spectro * spectro_model.hess_spec_freq
        
        # adding regularization
        shape_target = spectro_model.shape_target
        
        # # instanciate difference operator
        # dosf = Difference_Operator_Sep_Freq(shape_target)
        
        # # kernels in fourier
        # D_freq_row = dosf.D1_freq
        # D_freq_col = dosf.D2_freq
        
        diff_kernel_row = (np.array([-1, 1]))[..., np.newaxis]
        diff_kernel_col = (np.array([-1, 1]))[np.newaxis, ...]
        
        # diff_kernel_row = diff_ir(2, 0) # pour la cohérence avec difference_operator
        # diff_kernel_col = diff_ir(2, 1)

        D_freq_row = ir2fr(diff_kernel_row, shape=shape_target, real=False)
        D_freq_col = ir2fr(diff_kernel_col, shape=shape_target, real=False)
        
        # partitioningdiff_ir(2, 0)
        part_D_freq_row = partitioning_einops2(D_freq_row[np.newaxis, ...], spectro_model.di, spectro_model.dj)[0]
        part_D_freq_col = partitioning_einops2(D_freq_col[np.newaxis, ...], spectro_model.di, spectro_model.dj)[0]
        
        # summing on diagonal of hessian
        regul_hess_fusion = np.copy(hessian_fusion)
        n_spec, _, di_times_dj, _, h_block, w_block = regul_hess_fusion.shape
        for k in range(n_spec):
            for i in range(di_times_dj):
                coeff = L_mu_reg[k] / (2 * scale)
                regul_hess_fusion[k, k, i, i, :, :] += (
                    coeff * (np.abs(part_D_freq_row[i])**2 + np.abs(part_D_freq_col[i])**2)
                )
                # np.abs car nombre * nombre conjugué = np.abs(nombre ** 2)
                # pour avant, quand on travaillait avec gradients joints (laplacien),
                # dans Fourier, D_freq n'avait pas de partie imaginaire, donc par
                # chance, ne pas mettre np.abs marchait. Mais il faut toujours le mettre normalement
                
        self.regul_hess_fusion = regul_hess_fusion
        self.di = spectro_model.di
        self.dj = spectro_model.dj
        self.shape_target = shape_target
        
        self.mirim_model_for_fusion = mirim_model_for_fusion
        self.mu_mirim = mu_mirim
        
        self.spectro_model = spectro_model
        self.mu_spectro = mu_spectro
        
class Inv_Regul_Fusion_Model3():
    def __init__(self, regul_fusion_model:Regul_Fusion_Model3):
        # make inverse of hessian        
        inv_hess_fusion = make_iHtH_spectro(regul_fusion_model.regul_hess_fusion)
        
        self.inv_hess_fusion = inv_hess_fusion
        
        self.mirim_model_for_fusion = regul_fusion_model.mirim_model_for_fusion
        self.mu_mirim = regul_fusion_model.mu_mirim
        
        self.spectro_model = regul_fusion_model.spectro_model
        self.mu_spectro = regul_fusion_model.mu_spectro
        
        self.di = regul_fusion_model.di
        self.dj = regul_fusion_model.dj
        
        self.shape_target = regul_fusion_model.shape_target
        
    # def map_reconstruction(self, mirim_obs_data, spectro_obs_data): # pour faire reco sur critère quadratique avec regul gradients séparés
    #     mirim_adjoint_data = self.mirim_model_for_fusion.adjoint_freq_full(mirim_obs_data)
    #     spectro_adjoint_data = self.spectro_model.adjoint_freq_full(spectro_obs_data)
        
    #     adjoint_data_freq = self.mu_mirim * mirim_adjoint_data + self.mu_spectro * spectro_adjoint_data
        
    #     return np.real(idft2(apply_hessian_freq(self.inv_hess_fusion, self.di, self.dj, self.shape_target, adjoint_data_freq)))
        
    # def apply_Qinv(self, x):
    #     return apply_hessian2(self.inv_hess_fusion, self.di, self.dj, self.shape_target, x)
    
    def apply_Qinv_freq(self, x_freq):
        return apply_hessian_freq(self.inv_hess_fusion, self.di, self.dj, self.shape_target, x_freq)