#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 22:10:22 2025

@author: dpineau
"""


import numpy as np
from udft import ir2fr
from udft import irdftn, rdft2, laplacian, dft2, idft2
from einops import einsum, rearrange
# import aljabr
from linop import Diff, LinOp

from scipy.integrate import trapezoid
import itertools as it




# converts microJansky/arcsec^2, the unit of the cube
# to photons/(s * pixel * microns), the unit of the multispectral data
# lamb in meters (!), pixel_arcsec in arcsec
# def unit_conversion(flux, lamb, pixel_arcsec):
#     print("conversion des unités appliquée: microJansky/arcsec^2 --> photons/(s * pixel * microns)")
#     # Planck constant
#     h = 6.62607015e-34 # J.s
#     # aperture of the jwst
#     A_tel = 25.03 # m^2

#     # solid angle of a pixel
#     omega_pix = pixel_arcsec**2 # arcsec^2.pixel^-1
    
#     new_flux = flux * 1e-38 * (A_tel * omega_pix) / (h * lamb)
    
#     return new_flux

def unit_conversion(flux, lamb, pixel_arcsec):
    # print("conversion des unités non appliquée")
    return flux



# =============================================================================
# IMAGER MODEL
# =============================================================================



def compute_hess_mirim(H_freq):
    """compute H_freq^t * H_freq"""
    n_bands, n_spec, l, k = H_freq.shape
    hess_mirim = np.zeros(shape=(n_spec, n_spec, l, k), dtype=np.complex128)

    for p, q in it.product(range(n_spec), range(n_spec)):
        hess_mirim[p, q] = np.sum(np.conj(H_freq[:, p]) * H_freq[:, q], axis=0)

    return hess_mirim

# compute the equivalent of a dot product between a matrix with a shape equal
# to the hessian of mirim (the hessian itself or its inverse) and an input with
# a shape equal to abundance maps (abundance maps themselves or adjoint of mirim data)

def apply_hess_mirim(hess_mirim, x, shape_target, real=False):
    if real == False:
        x_freq = dft2(x)
    elif real == True:
        x_freq = rdft2(x)

    hess_mirim_x_freq = np.sum(
        hess_mirim * x_freq[np.newaxis, ...], axis=1
    )
    
    if real == False:
        return np.real(idft2(hess_mirim_x_freq))
    elif real == True:
        return irdftn(hess_mirim_x_freq, shape_target)

def partitioning_hess_mirim_freq_full2(hess_mirim_freq_full, di, dj):
    n_spec, _, H, W = hess_mirim_freq_full.shape
    h, w = int(H/di), int(W/dj)
    part_hess_mirim_freq_full = np.zeros(shape=(n_spec, n_spec, di * dj, di * dj, h, w), dtype=np.complex128)
    # (5, 5, 25, 25, 50, 100) comme hess_spec_freq

    for p, q in it.product(range(n_spec), range(n_spec)):
        hess_bloc = hess_mirim_freq_full[p, q]
        bloc_in_line = rearrange(
            hess_bloc, "(dx bx) (dy by) -> (dx dy) bx by", dx=di, dy=dj
        )
        for d in range(di * dj):
            part_hess_mirim_freq_full[p, q, d, d] += bloc_in_line[d]

    return part_hess_mirim_freq_full

# PSFs are full and partitionned to allow the direct sum with the spectro hessian
class Mirim_Model_For_Fusion(LinOp):
    def __init__(
        self, psfs_monoch, L_pce, lamb_cube, L_specs, shape_target, di, dj, pixel_arcsec=0.111
    ):
        assert psfs_monoch.shape[1] <= shape_target[0] # otherwise ir2fr impossible
        assert psfs_monoch.shape[2] <= shape_target[1]
        
        # print("Use of class:", "Mirim_Model_Full_Part")
        n_spec, n_lamb = L_specs.shape
        self.n_spec = n_spec
        self.shape_target = shape_target

        specs = L_specs[np.newaxis, :, :, np.newaxis, np.newaxis]  # (1, 5, 300, 1, 1)
        psfs = psfs_monoch[np.newaxis, np.newaxis, ...]  # (1, 1, 300, 250, 500)
        L_pce = unit_conversion(L_pce, lamb_cube * 1e-6, pixel_arcsec)
        pce = L_pce[:, np.newaxis, :, np.newaxis, np.newaxis]  # (9, 1, 300, 1, 1)
        
        # H_int = trapezoid(specs * psfs * pce, x=lamb_cube, axis=2)  # (9, 300, 250, 500)
        # TODO: new normalisation added here
        # pce_norms = np.sum(L_pce, axis=1)[:, np.newaxis, np.newaxis, np.newaxis]
        pce_norms = trapezoid(L_pce * lamb_cube[np.newaxis, ...], x = lamb_cube, axis = 1)[:, np.newaxis, np.newaxis, np.newaxis]
        new_lamb_cube = lamb_cube[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
        H_int = trapezoid(specs * psfs * pce * new_lamb_cube, x=lamb_cube, axis=2) / pce_norms # (9, 300, 250, 500)

        H_freq_full = ir2fr(H_int, shape_target, real=False)  # (9, 300, 250, 500)
        self.H_freq_full = H_freq_full
        
        H_freq = ir2fr(H_int, shape_target, real=True)
        self.H_freq = H_freq

        hess_mirim_freq_full = compute_hess_mirim(H_freq_full)
        part_hess_mirim_freq_full = partitioning_hess_mirim_freq_full2(hess_mirim_freq_full, di, dj)
        self.part_hess_mirim_freq_full = part_hess_mirim_freq_full  # (5, 5, 25, 25, 50, 100)

        n_bands, _ = L_pce.shape
        self.n_bands = n_bands
        
        self.di = di
        self.dj = dj
        
        self.L_pce = L_pce
        
        self.psfs_monoch = psfs_monoch

        super().__init__(
            ishape=(n_spec, shape_target[0], shape_target[1]),
            oshape=(n_bands, shape_target[0], shape_target[1]),
        )

    def forward(self, x):  # shape of x: (5, 250, 500), costs 2
        return np.real(idft2(np.sum(self.H_freq_full * dft2(x)[np.newaxis, ...], axis=1)))
    
    def forward_freq_to_real(self, x_freq):  # shape of x: (5, 250, 500), input in freq, and output in real, costs 1
        return np.real(idft2(np.sum(self.H_freq_full * x_freq[np.newaxis, ...], axis=1)))
    
    def forward_freq_to_freq(self, x_freq):  # shape of x: (5, 250, 500), input and output in freq, costs 0
        return np.sum(self.H_freq_full * x_freq[np.newaxis, ...], axis=1)

    def adjoint(self, y):  # shape of y: (9, 250, 500)
        return np.real(idft2(np.sum(np.conj(self.H_freq_full) * dft2(y)[:, np.newaxis, ...], axis=0)))
    
    def adjoint_real_to_freq_full(self, y):  # shape of y: (9, 250, 500)
        return np.sum(np.conj(self.H_freq_full) * dft2(y)[:, np.newaxis, ...], axis=0)
    
    def adjoint_real_to_freq(self, y):  # shape of y: (9, 250, 500)
        return np.sum(np.conj(self.H_freq) * rdft2(y)[:, np.newaxis, ...], axis=0)
    
    # def adjoint_freq_full(self, y):  # shape of y: (9, 250, 500), return adjoint in fourier, and real = False
    #     return np.sum(np.conj(self.H_freq_full) * dft2(y)[:, np.newaxis, ...], axis=0)

    def fwadj(self, x):  # shape of x: (5, 250, 500)
        return apply_hessian2(self.part_hess_mirim_freq_full, self.di, self.dj, self.shape_target, x)

class Mirim_Model_For_Fusion_2(LinOp):
    def __init__(
        self, psfs_monoch, L_pce, lamb_cube, shape_target, di, dj, pixel_arcsec=0.111
    ):
        assert psfs_monoch.shape[1] <= shape_target[0] # otherwise ir2fr impossible
        assert psfs_monoch.shape[2] <= shape_target[1]
        
        # print("Use of class:", "Mirim_Model_Full_Part")
        # n_spec, n_lamb = L_specs.shape
        # self.n_spec = n_spec
        self.shape_target = shape_target

        # specs = L_specs[np.newaxis, :, :, np.newaxis, np.newaxis]  # (1, 5, 300, 1, 1)
        psfs = psfs_monoch[np.newaxis, np.newaxis, ...]  # (1, 1, 300, 250, 500)
        L_pce = unit_conversion(L_pce, lamb_cube * 1e-6, pixel_arcsec)
        pce = L_pce[:, np.newaxis, :, np.newaxis, np.newaxis]  # (9, 1, 300, 1, 1)
        
        # H_int = trapezoid(specs * psfs * pce, x=lamb_cube, axis=2)  # (9, 300, 250, 500)
        # TODO: new normalisation added here
        # pce_norms = np.sum(L_pce, axis=1)[:, np.newaxis, np.newaxis, np.newaxis]
        pce_norms = trapezoid(L_pce * lamb_cube[np.newaxis, ...], x = lamb_cube, axis = 1)[:, np.newaxis, np.newaxis, np.newaxis]
        new_lamb_cube = lamb_cube[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
        H_int = trapezoid(psfs * pce * new_lamb_cube, x=lamb_cube, axis=2) / pce_norms # (9, 300, 250, 500)

        H_freq_full = ir2fr(H_int, shape_target, real=False)  # (9, 300, 250, 500)
        self.H_freq_full = H_freq_full
        
        H_freq = ir2fr(H_int, shape_target, real=True)
        self.H_freq = H_freq

        hess_mirim_freq_full = compute_hess_mirim(H_freq_full)
        part_hess_mirim_freq_full = partitioning_hess_mirim_freq_full2(hess_mirim_freq_full, di, dj)
        self.part_hess_mirim_freq_full = part_hess_mirim_freq_full  # (5, 5, 25, 25, 50, 100)

        n_bands, _ = L_pce.shape
        self.n_bands = n_bands
        
        self.di = di
        self.dj = dj
        
        self.L_pce = L_pce
        
        self.psfs_monoch = psfs_monoch

        super().__init__(
            ishape=(self.psfs_monoch.shape[0], shape_target[0], shape_target[1]),
            oshape=(n_bands, shape_target[0], shape_target[1]),
        )

    def forward(self, x):  # shape of x: (5, 250, 500), costs 2
        return np.real(irdftn(np.sum(self.H_freq * rdft2(x), axis=1), shape = self.shape_target))
        # return np.real(idft2(np.sum(self.H_freq_full * dft2(x)[np.newaxis, ...], axis=1)))
    
    def forward_freq_to_real(self, x_freq):  # shape of x: (5, 250, 500), input in freq, and output in real, costs 1
        return np.real(idft2(np.sum(self.H_freq_full * x_freq[np.newaxis, ...], axis=1)))
    
    def forward_freq_to_freq(self, x_freq):  # shape of x: (5, 250, 500), input and output in freq, costs 0
        return np.sum(self.H_freq_full * x_freq[np.newaxis, ...], axis=1)

    def adjoint(self, y):  # shape of y: (9, 250, 500)
        return np.real(idft2(np.sum(np.conj(self.H_freq_full) * dft2(y)[:, np.newaxis, ...], axis=0)))
    
    def adjoint_real_to_freq_full(self, y):  # shape of y: (9, 250, 500)
        return np.sum(np.conj(self.H_freq_full) * dft2(y)[:, np.newaxis, ...], axis=0)
    
    def adjoint_real_to_freq(self, y):  # shape of y: (9, 250, 500)
        return np.sum(np.conj(self.H_freq) * rdft2(y)[:, np.newaxis, ...], axis=0)
    
    # def adjoint_freq_full(self, y):  # shape of y: (9, 250, 500), return adjoint in fourier, and real = False
    #     return np.sum(np.conj(self.H_freq_full) * dft2(y)[:, np.newaxis, ...], axis=0)

    def fwadj(self, x):  # shape of x: (5, 250, 500)
        return apply_hessian2(self.part_hess_mirim_freq_full, self.di, self.dj, self.shape_target, x)





# =============================================================================
# SPECTROMETER MODEL
# =============================================================================



def partitioning_einops2(cube, di, dj):
    new_cube = rearrange(
        cube, "wl (dx bx) (dy by) -> wl (dx dy) bx by", dx=di, dy=dj
    )
    return new_cube

def partitioning_einops_2D(image, di, dj):
    new_image = rearrange(
        image, "(dx bx) (dy by) -> (dx dy) bx by", dx=di, dy=dj
    )
    return new_image

# diff with concatenating: now works with decim different for both dimensions
def concatenating2(cubef, shape_target, di, dj):
    n_maps, d1_times_d2, h_block, w_block = cubef.shape
    h, w = shape_target
    
    concatenated_cube = np.zeros((n_maps, h, w), dtype=complex)
    k = 0
    for i in range(di):
        for j in range(dj):
            concatenated_cube[:, i * h_block : (i+1) * h_block, j * w_block : (j+1) * w_block] += cubef[:, k, :, :]
            k += 1
    
    return concatenated_cube

# compute the equivalent of a dot product between a matrix with a shape equal
# to the hessian of the spectro (the hessian itself, its inverse, the hessian for fusion or its inverse) and an input with
# a shape equal to abundance maps (abundance maps themselves or adjoint of data, or sum of adjoint data)
# apply_hessian2, diff with apply_hessian: di and dj different, instead of di == dj
def apply_hessian2(hess_spec_freq, di, dj, shape_target, x, x_is_freq_and_part=False):
    if x_is_freq_and_part:
        part_x_freq = x
    else:
        x_freq = dft2(x)

        # partitionnement de x
        part_x_freq = partitioning_einops2(x_freq, di, dj)  # (5, 25, 50, 100)
        # print("part_x_freq", part_x_freq.shape)

    # produit de HtH avec x
    HtH_x_freq = hess_spec_freq * part_x_freq[np.newaxis, :, np.newaxis, :, :, :]
    # (5, 5, 25, 25, 50, 100) * (1, 5, 1, 25, 50, 100) = (5, 5, 25, 25, 50, 100)
    # print("HtH_x_freq", HtH_x_freq.shape)

    # somme avec np.sum
    # HtH_x_freq_sum1 = np.sum(
    #     HtH_x_freq, axis=1
    # )  # (5, x, 25, 25, 50, 100) --> (5, 25, 25, 50, 100)
    # HtH_x_freq_sum2 = np.sum(
    #     HtH_x_freq_sum1, axis=2
    # )  # (5, 25, x, 50, 100) --> (5, 25, 50, 100)
    
    # # print("HtH_x_freq_sum2", HtH_x_freq_sum2.shape)

    # # # reconstitution des cartes en freq
    # # concat_HtH_x_freq = concatenating(HtH_x_freq_sum2, shape_target)[
    # #     :, :, : int(shape_target[1] / 2) + 1
    # # ]
    # # # (5, 25, 50, 100) --> (5, 250, 500) --> (5, 250, 251)

    # # # sortie de Fourier
    # # HtH_x = irdftn(concat_HtH_x_freq, shape_target)
    
    # # reconstitution des cartes en freq
    # concat_HtH_x_freq = concatenating2(HtH_x_freq_sum2, shape_target, di, dj)
    # # (5, 25, 50, 100) --> (5, 250, 500) --> (5, 250, 251)
    # # print("concat_HtH_x_freq", concat_HtH_x_freq.shape)
    
    # somme avec einsum
    HtH_x_freq_sum = einsum(HtH_x_freq, "ti tj di dj h w -> ti di h w")
    # reconstitution des cartes en freq
    concat_HtH_x_freq = concatenating2(HtH_x_freq_sum, shape_target, di, dj)

    # sortie de Fourier
    HtH_x = np.real(idft2(concat_HtH_x_freq))
    # print("HtH_x", HtH_x.shape)

    return HtH_x

# la matrice inclut désormais le filtre en fréquence correspondant à la somme
# de chaque pixel avant la décimation.
def make_H_spec_freq_sum2(array_psfs, L_pce, L_lamb, L_spec, shape_target, di, dj, pixel_arcsec = 0.111):
    # print("Use of function {}.".format("make_H_spec_freq_sum"))
    weighted_psfs = array_psfs * L_pce[..., np.newaxis, np.newaxis] # (300, 250, 500)
    newaxis_weighted_psfs = weighted_psfs[np.newaxis, ...] # (1, 300, 250, 500)
    
    L_spec = unit_conversion(L_spec, L_lamb * 1e-6, pixel_arcsec)
    specs = L_spec[..., np.newaxis, np.newaxis] # (5, 300, 1, 1)

    H_spec = newaxis_weighted_psfs * specs # (5, 300, 250, 500)

    H_spec_freq = ir2fr(H_spec, shape_target)
    
    # différence par rapport à make_H_spec_freq est ici
    kernel_for_sum = np.ones((di, dj)) # le flux est bien intégré sur toute la surface du pixel, sans normalisation
    # print("avant ir2fr de make_H_spec_freq")
    kernel_for_sum_freq = ir2fr(kernel_for_sum, shape_target)[np.newaxis, np.newaxis, ...] # (1, 1, 250, 251)
    # print("après ir2fr de make_H_spec_freq")
    
    return H_spec_freq * kernel_for_sum_freq # (5, 300, 250, 251)

def make_H_spec_freq_sum2_2(array_psfs, L_pce, L_lamb, shape_target, di, dj, pixel_arcsec = 0.111):
    # print("Use of function {}.".format("make_H_spec_freq_sum"))
    weighted_psfs = array_psfs * L_pce[..., np.newaxis, np.newaxis] # (300, 250, 500)
    newaxis_weighted_psfs = weighted_psfs[np.newaxis, ...] # (1, 300, 250, 500)
    
    # L_spec = unit_conversion(L_spec, L_lamb * 1e-6, pixel_arcsec)
    # specs = L_spec[..., np.newaxis, np.newaxis] # (5, 300, 1, 1)

    H_spec = newaxis_weighted_psfs # (5, 300, 250, 500)

    H_spec_freq = ir2fr(H_spec, shape_target)
    
    # différence par rapport à make_H_spec_freq est ici
    kernel_for_sum = np.ones((di, dj)) # le flux est bien intégré sur toute la surface du pixel, sans normalisation
    # print("avant ir2fr de make_H_spec_freq")
    kernel_for_sum_freq = ir2fr(kernel_for_sum, shape_target)[np.newaxis, np.newaxis, ...] # (1, 1, 250, 251)
    # print("après ir2fr de make_H_spec_freq")
    
    return H_spec_freq * kernel_for_sum_freq # (5, 300, 250, 251)

def make_H_spec_freq_sum_full(array_psfs, L_pce, L_lamb, L_spec, shape_target, di, dj, pixel_arcsec = 0.111):
    # print("Use of function {}.".format("make_H_spec_freq_sum"))
    weighted_psfs = array_psfs * L_pce[..., np.newaxis, np.newaxis] # (300, 250, 500)
    newaxis_weighted_psfs = weighted_psfs[np.newaxis, ...] # (1, 300, 250, 500)
    
    L_spec = unit_conversion(L_spec, L_lamb * 1e-6, pixel_arcsec)
    specs = L_spec[..., np.newaxis, np.newaxis] # (5, 300, 1, 1)

    H_spec = newaxis_weighted_psfs * specs # (5, 300, 250, 500)

    H_spec_freq = ir2fr(H_spec, shape_target, real=False)
    
    # différence par rapport à make_H_spec_freq est ici
    kernel_for_sum = np.ones((di, dj)) # le flux est bien intégré sur toute la surface du pixel, sans normalisation
    kernel_for_sum_freq = ir2fr(kernel_for_sum, shape_target, real=False)[np.newaxis, np.newaxis, ...] # (1, 1, 250, 251)
    
    return H_spec_freq * kernel_for_sum_freq # (5, 300, 250, 251)


# =============================================================================
# NEW FUNCTIONS RELATED TO PADDING
# =============================================================================

# cube has shape: (wavelength, i, j)
def pad_cube_to_fit_decim(cube, di, dj):
    padding1 = 0
    if cube.shape[1] % di != 0:
        while (cube.shape[1] + padding1) % di != 0:
            padding1 += 1
    
    padding2 = 0
    if cube.shape[2] % dj != 0:
        while (cube.shape[2] + padding2) % dj != 0:
            padding2 += 1
    
    if padding1 % 2 == 0: # si padding1 pair
        pad1_1 =  padding1 // 2
        pad1_2 =  pad1_1
    else: # si padding1 impair
        pad1_1 = padding1 // 2
        pad1_2 = padding1 - pad1_1
    
    if padding2 % 2 == 0: # si padding2 pair
        pad2_1 =  padding2 // 2
        pad2_2 =  pad2_1
    else: # si padding2 impair
        pad2_1 = padding2 // 2
        pad2_2 = padding2 - pad2_1
    
    # mode by default pads with 0
    # padded_cube = np.pad(cube, ((0, 0), (pad1_1, pad1_2), (pad2_1, pad2_2)))
    
    # mode = "edge" pads by repeating values on the borders: not okay when padding adds a lot of data, increases flux!
    padded_cube = np.pad(cube, ((0, 0), (pad1_1, pad1_2), (pad2_1, pad2_2)), mode = "linear_ramp")
    new_shape_target = padded_cube.shape[1:]
    
    return padded_cube, new_shape_target

def crop_cube_to_fit_decim(cube, di, dj):
    cropping1 = 0
    if cube.shape[1] % di != 0:
        while (cube.shape[1] - cropping1) % di != 0:
            cropping1 += 1
    
    cropping2 = 0
    if cube.shape[2] % dj != 0:
        while (cube.shape[2] - cropping2) % dj != 0:
            cropping2 += 1
    
    # print(cropping1, cropping2)
    
    if cropping1 == 0 and cropping2 == 0: # if not crop needed
        return cube, cube.shape[1:]
    
    elif cropping1 != 0 and cropping2 == 0:
        if cropping1 % 2 == 0: # si cropping1 pair
            crop1_1 =  cropping1 // 2
            crop1_2 =  crop1_1
        else: # si cropping1 impair
            crop1_1 = cropping1 // 2
            crop1_2 = cropping1 - crop1_1
        cropped_cube = cube[:, crop1_1 : - crop1_2, :]
        
    elif cropping1 == 0 and cropping2 != 0:
        if cropping2 % 2 == 0: # si cropping2 pair
            crop2_1 =  cropping2 // 2
            crop2_2 =  crop2_1
        else: # si cropping2 impair
            crop2_1 = cropping2 // 2
            crop2_2 = cropping2 - crop2_1
        cropped_cube = cube[:, :, crop2_1 : - crop2_2]
        
    elif cropping1 != 0 and cropping2 != 0:
        if cropping1 % 2 == 0: # si cropping1 pair
            crop1_1 =  cropping1 // 2
            crop1_2 =  crop1_1
        else: # si cropping1 impair
            crop1_1 = cropping1 // 2
            crop1_2 = cropping1 - crop1_1
        
        if cropping2 % 2 == 0: # si cropping2 pair
            crop2_1 =  cropping2 // 2
            crop2_2 =  crop2_1
        else: # si cropping2 impair
            crop2_1 = cropping2 // 2
            crop2_2 = cropping2 - crop2_1
        cropped_cube = cube[:, crop1_1 : - crop1_2, crop2_1 : - crop2_2]
    
    new_shape_target = cropped_cube.shape[1:]
    
    return cropped_cube, new_shape_target

def shape_target_to_fit_decim(shape_target, di, dj):
    padding1 = 0
    if shape_target[0] % di != 0:
        while (shape_target[0] + padding1) % di != 0:
            padding1 += 1
    
    padding2 = 0
    if shape_target[1] % dj != 0:
        while (shape_target[1] + padding2) % dj != 0:
            padding2 += 1
    
    new_shape_target = (shape_target[0] + padding1, shape_target[1] + padding2)
    return new_shape_target

class Spectro_Model_3(LinOp):
    def __init__(
        self, psfs_monoch, L_pce, di:int, dj:int, lamb_cube, L_specs, shape_target, pixel_arcsec=0.111 # size of pixels before integration and decimation of spectro
    ):
        assert shape_target[0] % di == 0
        assert shape_target[1] % dj == 0
        
        assert psfs_monoch.shape[1] <= shape_target[0] # otherwise ir2fr impossible
        assert psfs_monoch.shape[2] <= shape_target[1]
        
        kernel_for_sum = np.ones((di, dj)) # le flux est bien intégré sur toute la surface du pixel, sans normalisation
        kernel_for_sum_freq = ir2fr(kernel_for_sum, shape_target, real=False)[np.newaxis, ...] # (1, 250, 500)
    
        psfs_freq = ir2fr(
            psfs_monoch * L_pce[:, np.newaxis, np.newaxis],
            shape=shape_target,
            real=False,
        ) * kernel_for_sum_freq
        
        # # translation dans Fourier pour sauvegarde de la convolution en haut à gauche
        # # MÉTHODE 1
        decal = np.zeros(shape_target)
        dsi = int((di-1)/2)
        dsj = int((dj-1)/2)
        # if ds != 0:
        #     ds = 0
        # print("dsi", dsi, "dsj", dsj)
        decal[- dsi, - dsj] = np.sqrt(shape_target[0] * shape_target[1]) # surement pour annuler les normalisations qui arrivent dans le passage de Fourier ?
        decalf = dft2(decal)

        h_block, w_block = int(shape_target[0] / di), int(shape_target[1] / dj)
        
        # partitionnement
        # part_psfs_freq_full = partitioning_einops2(psfs_freq, di, dj)
        part_psfs_freq_full = partitioning_einops2(psfs_freq * decalf, di, dj)
        # print("part_psfs_freq_full", part_psfs_freq_full.shape)

        # conjugué des psfs partitionnées
        conj_part_psfs_freq_full = np.conj(part_psfs_freq_full)
        # print("conj_part_psfs_freq_full", conj_part_psfs_freq_full.shape)

        # produit des psfs avec les conjuguées
        # (300, 1, 25, 50, 100) * (300, 25, 1, 50, 100) = (300, 25, 25, 50, 100)
        mat = (
            (1 / (di * dj))
            * part_psfs_freq_full[:, np.newaxis, ...]
            * conj_part_psfs_freq_full[:, :, np.newaxis, ...]
        )
        # print("mat", mat.shape)

        # création de HtH
        L_specs_converted = unit_conversion(L_specs, lamb_cube * 1e-6, pixel_arcsec)
        specs = L_specs_converted[
            :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis
        ]  # (5, 300, 1, 1)
        # print("check1")
        n_spec = specs.shape[0]
        HtH_freq = np.zeros(
            (n_spec, n_spec, di * dj, di * dj, h_block, w_block), dtype=complex
        )
        # print("check2")
        for k1 in range(n_spec):
            for k2 in range(k1, n_spec):
                HtH_freq[k1, k2] += np.sum(specs[k1] * specs[k2] * mat, axis=0)

        # print("check3")
        # utilisation de la symétrie de HtH
        for k1 in range(n_spec):
            for k2 in range(k1):
                HtH_freq[k1, k2] += HtH_freq[k2, k1]

        self.hess_spec_freq = HtH_freq

        # H_spec_freq utile pour forward et adjoint
        # self.H_spec_freq = make_H_spec_freq_sum2(
        #     psfs_monoch, L_pce, lamb_cube, L_specs, shape_target, di, dj
        # )
        
        # print("2 x H_spec_freq enlevés !!")
        
        self.H_spec_freq = make_H_spec_freq_sum2(
            psfs_monoch, L_pce, lamb_cube, L_specs, shape_target, di, dj
        ) * rdft2(decal)[np.newaxis, np.newaxis, :, :]
        
        # utile pour forward_freq_to_freq et forward_freq_to_real
        self.H_spec_freq_full = make_H_spec_freq_sum_full(
            psfs_monoch, L_pce, lamb_cube, L_specs, shape_target, di, dj
        ) * decalf[np.newaxis, np.newaxis, :, :]
        
        self.di = di
        self.dj = dj
        self.shape_target = shape_target
        self.n_lamb = lamb_cube.shape[0]
        self.n_spec = n_spec
        self.L_pce = L_pce
        self.psfs_monoch = psfs_monoch

        super().__init__(
            ishape=(self.n_spec, shape_target[0], shape_target[1]),
            oshape=(self.n_lamb, shape_target[0] // di, shape_target[1] // dj),
        )

    def forward(self, x): # input and output in real, costs 1
        assert x.shape == self.ishape
        
        x_freq = rdft2(x)[:, np.newaxis, ...]  # (5, 1, 250, 251)
        H_spec_x_freq = np.sum(
            self.H_spec_freq * x_freq, axis=0
        )  # (5, 300, 250, 251) * (5, 1, 250, 251) = (300, 250, 251))
        convoluted_cube = irdftn(H_spec_x_freq, self.shape_target)  # (300, 250, 500)

        # make decimated cube
        decimated_cube = convoluted_cube[
            :, :: self.di, :: self.dj
        ]  # (300, 50, 100)
        return decimated_cube
    
    def forward_freq_to_freq(self, x_freq): # input and output in freq, costs 2
        assert x_freq.shape == self.ishape
        
        H_spec_x_freq = np.sum(
            self.H_spec_freq_full * x_freq[:, np.newaxis, ...], axis=0
        )  # (5, 300, 250, 251) * (5, 1, 250, 251) = (300, 250, 251))
        convoluted_cube = idft2(H_spec_x_freq)  # (300, 250, 500)

        # make decimated cube
        decimated_cube = convoluted_cube[
            :, :: self.di, :: self.dj
        ]  # (300, 50, 100)
        return dft2(decimated_cube)
    
    def forward_freq_to_real(self, x_freq): # input in freq, output in real, costs 1
        assert x_freq.shape == self.ishape
        
        H_spec_x_freq = np.sum(
            self.H_spec_freq_full * x_freq[:, np.newaxis, ...], axis=0
        )  # (5, 300, 250, 251) * (5, 1, 250, 251) = (300, 250, 251))
        convoluted_cube = idft2(H_spec_x_freq)  # (300, 250, 500)

        # make decimated cube
        decimated_cube = convoluted_cube[
            :, :: self.di, :: self.dj
        ]  # (300, 50, 100)
        return decimated_cube

    def adjoint(self, y):
        assert y.shape == self.oshape
        
        # bourrage de zéros
        original_cube = np.zeros(
            (self.n_lamb, self.shape_target[0], self.shape_target[1])
        )  # (300, 250, 500)
        original_cube[:, :: self.di, :: self.dj] = y
        

        # make convolution with conjugated weighted psfs
        original_cube_freq = rdft2(original_cube)[np.newaxis, ...]  # (1, 300, 250, 251)
        # H_spec_x_freq = np.sum(
        #     np.conj(self.H_spec_freq) * original_cube_freq, axis=1
        # )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        H_spec_x_freq = einsum(
            np.conj(self.H_spec_freq) * original_cube_freq, "t l i j -> t i j"
        )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        maps = irdftn(H_spec_x_freq, self.shape_target)  # (5, 250, 500)
        
        return maps  # shape = 5, 250, 500
    
    def adjoint_real_to_freq_full(self, y):
        assert y.shape == self.oshape
        
        # bourrage de zéros
        original_cube = np.zeros(
            (self.n_lamb, self.shape_target[0], self.shape_target[1])
        )  # (300, 250, 500)
        original_cube[:, :: self.di, :: self.dj] = y

        # make convolution with conjugated weighted psfs
        original_cube_freq = dft2(original_cube)[np.newaxis, ...]  # (1, 300, 250, 251)
        # H_spec_x_freq = np.sum(
        #     np.conj(self.H_spec_freq_full) * original_cube_freq, axis=1
        # )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        
        H_spec_x_freq = einsum(
            np.conj(self.H_spec_freq_full) * original_cube_freq, "t l i j -> t i j"
        )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        
        return H_spec_x_freq  # shape = 5, 250, 500
    
    def adjoint_real_to_freq(self, y):
        assert y.shape == self.oshape
        
        # bourrage de zéros
        original_cube = np.zeros(
            (self.n_lamb, self.shape_target[0], self.shape_target[1])
        )  # (300, 250, 500)
        original_cube[:, :: self.di, :: self.dj] = y

        # make convolution with conjugated weighted psfs
        original_cube_freq = rdft2(original_cube)[np.newaxis, ...]  # (1, 300, 250, 251)
        # H_spec_x_freq = np.sum(
        #     np.conj(self.H_spec_freq_full) * original_cube_freq, axis=1
        # )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        
        H_spec_x_freq = einsum(
            np.conj(self.H_spec_freq) * original_cube_freq, "t l i j -> t i j"
        )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        
        return H_spec_x_freq  # shape = 5, 250, 500
    
    # def adjoint_freq_full(self, y): # return adjoint in fourier, and real = False
    #     assert y.shape == self.oshape
        
    #     # bourrage de zéros
    #     original_cube = np.zeros(
    #         (self.n_lamb, self.shape_target[0], self.shape_target[1])
    #     )  # (300, 250, 500)
    #     original_cube[:, :: self.di, :: self.dj] = y
        

    #     # make convolution with conjugated weighted psfs
    #     original_cube_freq = dft2(original_cube)[np.newaxis, ...]  # (1, 300, 250, 251)
    #     H_spec_x_freq = np.sum(
    #         np.conj(self.H_spec_freq_full) * original_cube_freq, axis=1
    #     )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        
    #     return H_spec_x_freq  # shape = 5, 250, 500

    def fwadj(self, x):
        assert x.shape == self.ishape
        return apply_hessian2(self.hess_spec_freq, self.di, self.dj, self.shape_target, x)

class Spectro_Model_3_2(LinOp):
    def __init__(
        self, psfs_monoch, L_pce, di:int, dj:int, lamb_cube, shape_target, pixel_arcsec=0.111 # size of pixels before integration and decimation of spectro
    ):
        assert shape_target[0] % di == 0
        assert shape_target[1] % dj == 0
        
        assert psfs_monoch.shape[1] <= shape_target[0] # otherwise ir2fr impossible
        assert psfs_monoch.shape[2] <= shape_target[1]
        
        kernel_for_sum = np.ones((di, dj)) # le flux est bien intégré sur toute la surface du pixel, sans normalisation
        kernel_for_sum_freq = ir2fr(kernel_for_sum, shape_target, real=False)[np.newaxis, ...] # (1, 250, 500)
    
        psfs_freq = ir2fr(
            psfs_monoch * L_pce[:, np.newaxis, np.newaxis],
            shape=shape_target,
            real=False,
        ) * kernel_for_sum_freq
        
        # # translation dans Fourier pour sauvegarde de la convolution en haut à gauche
        # # MÉTHODE 1
        decal = np.zeros(shape_target)
        dsi = int((di-1)/2)
        dsj = int((dj-1)/2)
        # if ds != 0:
        #     ds = 0
        # print("dsi", dsi, "dsj", dsj)
        decal[- dsi, - dsj] = np.sqrt(shape_target[0] * shape_target[1]) # surement pour annuler les normalisations qui arrivent dans le passage de Fourier ?
        # decalf = dft2(decal)

        # h_block, w_block = int(shape_target[0] / di), int(shape_target[1] / dj)
        
        # partitionnement
        # part_psfs_freq_full = partitioning_einops2(psfs_freq, di, dj)
        # part_psfs_freq_full = partitioning_einops2(psfs_freq * decalf, di, dj)
        # print("part_psfs_freq_full", part_psfs_freq_full.shape)

        # conjugué des psfs partitionnées
        # conj_part_psfs_freq_full = np.conj(part_psfs_freq_full)
        # print("conj_part_psfs_freq_full", conj_part_psfs_freq_full.shape)

        # produit des psfs avec les conjuguées
        # (300, 1, 25, 50, 100) * (300, 25, 1, 50, 100) = (300, 25, 25, 50, 100)
        # mat = (
        #     (1 / (di * dj))
        #     * part_psfs_freq_full[:, np.newaxis, ...]
        #     * conj_part_psfs_freq_full[:, :, np.newaxis, ...]
        # )
        # print("mat", mat.shape)

        # création de HtH
        # L_specs_converted = unit_conversion(L_specs, lamb_cube * 1e-6, pixel_arcsec)
        # specs = L_specs_converted[
        #     :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis
        # ]  # (5, 300, 1, 1)
        # print("check1")
        # n_spec = specs.shape[0]
        # HtH_freq = np.zeros(
        #     (n_spec, n_spec, di * dj, di * dj, h_block, w_block), dtype=complex
        # )
        # print("check2")
        # for k1 in range(n_spec):
        #     for k2 in range(k1, n_spec):
        #         HtH_freq[k1, k2] += np.sum(specs[k1] * specs[k2] * mat, axis=0)

        # print("check3")
        # utilisation de la symétrie de HtH
        # for k1 in range(n_spec):
        #     for k2 in range(k1):
        #         HtH_freq[k1, k2] += HtH_freq[k2, k1]

        # self.hess_spec_freq = HtH_freq

        # H_spec_freq utile pour forward et adjoint
        # self.H_spec_freq = make_H_spec_freq_sum2(
        #     psfs_monoch, L_pce, lamb_cube, L_specs, shape_target, di, dj
        # )
        
        # print("2 x H_spec_freq enlevés !!")
        
        self.H_spec_freq = make_H_spec_freq_sum2_2(
            psfs_monoch, L_pce, lamb_cube, shape_target, di, dj
        ) * rdft2(decal)[np.newaxis, np.newaxis, :, :]
        
        # utile pour forward_freq_to_freq et forward_freq_to_real
        # self.H_spec_freq_full = make_H_spec_freq_sum_full(
        #     psfs_monoch, L_pce, lamb_cube, L_specs, shape_target, di, dj
        # ) * decalf[np.newaxis, np.newaxis, :, :]
        
        self.di = di
        self.dj = dj
        self.shape_target = shape_target
        self.n_lamb = lamb_cube.shape[0]
        # self.n_spec = n_spec
        self.L_pce = L_pce
        self.psfs_monoch = psfs_monoch

        super().__init__(
            ishape=(self.psfs_monoch.shape[0], shape_target[0], shape_target[1]),
            oshape=(self.n_lamb, shape_target[0] // di, shape_target[1] // dj),
        )

    def forward(self, x): # input and output in real, costs 1
        assert x.shape == self.ishape
        
        x_freq = rdft2(x)[:, np.newaxis, ...]  # (5, 1, 250, 251)
        H_spec_x_freq = np.sum(
            self.H_spec_freq * x_freq, axis=0
        )  # (5, 300, 250, 251) * (5, 1, 250, 251) = (300, 250, 251))
        convoluted_cube = irdftn(H_spec_x_freq, self.shape_target)  # (300, 250, 500)

        # make decimated cube
        decimated_cube = convoluted_cube[
            :, :: self.di, :: self.dj
        ]  # (300, 50, 100)
        return decimated_cube
    
    def forward_freq_to_freq(self, x_freq): # input and output in freq, costs 2
        assert x_freq.shape == self.ishape
        
        H_spec_x_freq = np.sum(
            self.H_spec_freq_full * x_freq[:, np.newaxis, ...], axis=0
        )  # (5, 300, 250, 251) * (5, 1, 250, 251) = (300, 250, 251))
        convoluted_cube = idft2(H_spec_x_freq)  # (300, 250, 500)

        # make decimated cube
        decimated_cube = convoluted_cube[
            :, :: self.di, :: self.dj
        ]  # (300, 50, 100)
        return dft2(decimated_cube)
    
    def forward_freq_to_real(self, x_freq): # input in freq, output in real, costs 1
        assert x_freq.shape == self.ishape
        
        H_spec_x_freq = np.sum(
            self.H_spec_freq_full * x_freq[:, np.newaxis, ...], axis=0
        )  # (5, 300, 250, 251) * (5, 1, 250, 251) = (300, 250, 251))
        convoluted_cube = idft2(H_spec_x_freq)  # (300, 250, 500)

        # make decimated cube
        decimated_cube = convoluted_cube[
            :, :: self.di, :: self.dj
        ]  # (300, 50, 100)
        return decimated_cube

    def adjoint(self, y):
        assert y.shape == self.oshape
        
        # bourrage de zéros
        original_cube = np.zeros(
            (self.n_lamb, self.shape_target[0], self.shape_target[1])
        )  # (300, 250, 500)
        original_cube[:, :: self.di, :: self.dj] = y
        

        # make convolution with conjugated weighted psfs
        original_cube_freq = rdft2(original_cube)[np.newaxis, ...]  # (1, 300, 250, 251)
        # H_spec_x_freq = np.sum(
        #     np.conj(self.H_spec_freq) * original_cube_freq, axis=1
        # )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        H_spec_x_freq = einsum(
            np.conj(self.H_spec_freq) * original_cube_freq, "t l i j -> t i j"
        )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        maps = irdftn(H_spec_x_freq, self.shape_target)  # (5, 250, 500)
        
        return maps  # shape = 5, 250, 500
    
    def adjoint_real_to_freq_full(self, y):
        assert y.shape == self.oshape
        
        # bourrage de zéros
        original_cube = np.zeros(
            (self.n_lamb, self.shape_target[0], self.shape_target[1])
        )  # (300, 250, 500)
        original_cube[:, :: self.di, :: self.dj] = y

        # make convolution with conjugated weighted psfs
        original_cube_freq = dft2(original_cube)[np.newaxis, ...]  # (1, 300, 250, 251)
        # H_spec_x_freq = np.sum(
        #     np.conj(self.H_spec_freq_full) * original_cube_freq, axis=1
        # )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        
        H_spec_x_freq = einsum(
            np.conj(self.H_spec_freq_full) * original_cube_freq, "t l i j -> t i j"
        )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        
        return H_spec_x_freq  # shape = 5, 250, 500
    
    def adjoint_real_to_freq(self, y):
        assert y.shape == self.oshape
        
        # bourrage de zéros
        original_cube = np.zeros(
            (self.n_lamb, self.shape_target[0], self.shape_target[1])
        )  # (300, 250, 500)
        original_cube[:, :: self.di, :: self.dj] = y

        # make convolution with conjugated weighted psfs
        original_cube_freq = rdft2(original_cube)[np.newaxis, ...]  # (1, 300, 250, 251)
        # H_spec_x_freq = np.sum(
        #     np.conj(self.H_spec_freq_full) * original_cube_freq, axis=1
        # )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        
        H_spec_x_freq = einsum(
            np.conj(self.H_spec_freq) * original_cube_freq, "t l i j -> t i j"
        )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        
        return H_spec_x_freq  # shape = 5, 250, 500
    
    # def adjoint_freq_full(self, y): # return adjoint in fourier, and real = False
    #     assert y.shape == self.oshape
        
    #     # bourrage de zéros
    #     original_cube = np.zeros(
    #         (self.n_lamb, self.shape_target[0], self.shape_target[1])
    #     )  # (300, 250, 500)
    #     original_cube[:, :: self.di, :: self.dj] = y
        

    #     # make convolution with conjugated weighted psfs
    #     original_cube_freq = dft2(original_cube)[np.newaxis, ...]  # (1, 300, 250, 251)
    #     H_spec_x_freq = np.sum(
    #         np.conj(self.H_spec_freq_full) * original_cube_freq, axis=1
    #     )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        
    #     return H_spec_x_freq  # shape = 5, 250, 500

    def fwadj(self, x):
        assert x.shape == self.ishape
        return apply_hessian2(self.hess_spec_freq, self.di, self.dj, self.shape_target, x)

class Regul_Spectro_Model_2(LinOp):
    def __init__(self, model:Spectro_Model_3, L_mu):
        # print("Use of class:", "Regul_Spectro_Model")
        self.model = model
        self.n_spec = model.n_spec
        self.shape_target = model.shape_target
        self.di = model.di
        self.dj = model.dj

        diff_kernel = laplacian(2)
        D_freq = ir2fr(diff_kernel, shape=model.shape_target, real=False)

        part_D_freq = partitioning_einops2(D_freq[np.newaxis, ...], model.di, model.dj)[0]

        HtH_freq_spectro_regul = np.copy(model.hess_spec_freq)
        # HtH_freq_spectro_regul = np.ascontiguousarray(HtH_freq_spectro)
        for k in range(model.n_spec):
            for i in range(model.di * model.dj):
                HtH_freq_spectro_regul[k, k, i, i, :, :] += (
                    L_mu[k] * (part_D_freq[i]) ** 2
                )

        self.regul_hess_spec_freq = HtH_freq_spectro_regul

        super().__init__(
            ishape=(model.n_spec, model.shape_target[0], model.shape_target[1]),
            oshape=(model.n_lamb, model.shape_target[0] // model.di, model.shape_target[1] // model.dj),
        )

    # def forward(self, x):
    #     return apply_hessian(
    #         self.regul_hess_spec_freq, self.model.decim, self.shape_target, x
    #     )

    # def adjoint(self, x):
    #     return self.forward(x)
    
    def forward(self, x):
        return self.model.forward(x)
    
    def adjoint(self, y):
        return self.model.adjoint(y)
    
    def fwadj(self, x):
        return apply_hessian2(self.regul_hess_spec_freq, self.di, self.dj, self.shape_target, x)

def concat_M(M):
    nb_blocks, _, nb_subblocks, _ = M.shape
    concat_width = nb_blocks * nb_subblocks
    concat = np.zeros((concat_width, concat_width), dtype=complex)  # always a square
    for l in range(nb_blocks):
        for c in range(nb_blocks):
            concat[
                l * nb_subblocks : (l + 1) * nb_subblocks,
                c * nb_subblocks : (c + 1) * nb_subblocks,
            ] += M[l, c, ...]
    return concat

def split_M(M, split_shape):
    split = np.zeros(split_shape, dtype=complex)
    nb_blocks, _, nb_subblocks, _ = split_shape
    for l in range(nb_blocks):
        for c in range(nb_blocks):
            split[l, c, ...] += M[
                l * nb_subblocks : (l + 1) * nb_subblocks,
                c * nb_subblocks : (c + 1) * nb_subblocks,
            ]
    return split

def make_iHtH_spectro(HtH_freq_spectro):
    inv_hess_freq = np.zeros_like(HtH_freq_spectro, dtype=complex)
    H, W = inv_hess_freq.shape[-2:]
    for h in range(H):
        for w in range(W):
            M = np.copy(HtH_freq_spectro[..., h, w])
            C = concat_M(M)
            iC = np.linalg.inv(C)
            S = split_M(iC, inv_hess_freq.shape[:4])
            inv_hess_freq[..., h, w] += S
    return inv_hess_freq

class Inv_Regul_Spectro_Model_2():#aljabr.LinOp):
    def __init__(self, reg_model: Regul_Spectro_Model_2):
        # print("Use of class:", "Inv_Regul_Spectro_Model")
        self.reg_model = reg_model
        self.inv_hess = make_iHtH_spectro(reg_model.regul_hess_spec_freq)

        # super().__init__(
        #     ishape=(
        #         reg_model.n_spec,
        #         reg_model.shape_target[0],
        #         reg_model.shape_target[1],
        #         # reg_model.n_spec,
        #         # reg_model.decim**2,
        #         # reg_model.shape_target[0] // reg_model.decim,
        #         # reg_model.shape_target[1] // reg_model.decim,
        #     ),
        #     oshape=(
        #         reg_model.n_spec,
        #         reg_model.shape_target[0],
        #         reg_model.shape_target[1],
        #     ),
        # )

    def forward(self, x):
        return apply_hessian2(
            self.inv_hess, self.reg_model.di, self.reg_model.dj, self.reg_model.shape_target, x
        )

    def forward_on_part_and_freq(self, x):
        return apply_hessian2(
            self.inv_hess,
            self.reg_model.di,
            self.reg_model.dj,
            self.reg_model.shape_target,
            x,
            x_is_freq_and_part=True,
        )

    def adjoint(self, x):
        return self.forward(x)
    
    def map_reconstruction(self, spectro_obs_data):
        adjoint_spectro_obs_data = self.reg_model.adjoint(spectro_obs_data)
        spectro_obs_data_freq = dft2(adjoint_spectro_obs_data)

        # partitionnement de spectro_obs_data_freq
        # part_Ht_y_freq, _, _ = partitioning(spectro_obs_data_freq, self.reg_model.di, self.reg_model.dj)  # (5, 25, 50, 100)
        part_Ht_y_freq = partitioning_einops2(spectro_obs_data_freq, self.reg_model.di, self.reg_model.dj)  # (5, 25, 50, 100)

        return self.forward_on_part_and_freq(part_Ht_y_freq)



# =============================================================================
# NEW INSTRUMENT FUNCTION FOR FASTER FORWARD
# =============================================================================

class Spectrometer_Model(LinOp):
    def __init__(
        self,
        psfs_monoch: np.ndarray,
        L_pce: np.ndarray,
        lamb_cube: np.ndarray,
        shape_target: tuple,
        n_spec: int, 
        di: int,
        dj: int,
        pixel_arcsec: float = 0.111
    ):
        # Validate geometry
        assert shape_target[0] % di == 0, "shape_target height must be divisible by di"
        assert shape_target[1] % dj == 0, "shape_target width must be divisible by dj"
        assert psfs_monoch.shape[1] <= shape_target[0]
        assert psfs_monoch.shape[2] <= shape_target[1]

        # Store static parameters
        self.psfs_monoch = psfs_monoch
        self.L_pce = L_pce
        self.di = di
        self.dj = dj
        self.shape_target = shape_target
        self.lamb_cube = lamb_cube
        self.pixel_arcsec = pixel_arcsec
        self.n_spec = n_spec

        # Precompute shift factor (decal) for Fourier alignment
        decal = np.zeros(shape_target)
        dsi, dsj = (di - 1) // 2, (dj - 1) // 2
        decal[-dsi, -dsj] = np.sqrt(np.prod(shape_target))
        self.decalf_rd = rdft2(decal)

        # Fixed input/output shapes
        ishape = (self.n_spec, shape_target[0], shape_target[1])
        oshape = (lamb_cube.shape[0], shape_target[0] // di, shape_target[1] // dj)
        super().__init__(ishape=ishape, oshape=oshape)

    def forward(self, abundances: np.ndarray, L_specs: np.ndarray) -> np.ndarray:
        """
        Compute instrument response for given abundances and endmember spectra.

        Parameters
        ----------
        abundances : array, shape (5, 40, 40)
        L_specs : array, shape (5, n_lambda)

        Returns
        -------
        response : array, shape (n_lambda, 10, 10)
        """
        # Validate shapes
        assert abundances.shape == self.ishape
        assert L_specs.shape[0] == abundances.shape[0]

        # Build the spectral forward operator H_spec_freq exactly as in the original model
        H_spec_freq = (
            make_H_spec_freq_sum2(
                self.psfs_monoch,
                self.L_pce,
                self.lamb_cube,
                L_specs,
                self.shape_target,
                self.di,
                self.dj,
            )
            * self.decalf_rd[np.newaxis, np.newaxis, ...]
        )

        # FFT of abundances
        x_freq = rdft2(abundances)[:, np.newaxis, ...]

        # Convolution in frequency
        out_freq = np.sum(H_spec_freq * x_freq, axis=0)

        # Back to real space
        conv_full = irdftn(out_freq, self.shape_target)

        # Decimate spatially
        response = conv_full[:, :: self.di, :: self.dj]
        return response

    def forward_freq_to_freq(self, x_freq): # input and output in freq, costs 2
        assert x_freq.shape == self.ishape
        
        H_spec_x_freq = np.sum(
            self.H_spec_freq_full * x_freq[:, np.newaxis, ...], axis=0
        )  # (5, 300, 250, 251) * (5, 1, 250, 251) = (300, 250, 251))
        convoluted_cube = idft2(H_spec_x_freq)  # (300, 250, 500)

        # make decimated cube
        decimated_cube = convoluted_cube[
            :, :: self.di, :: self.dj
        ]  # (300, 50, 100)
        return dft2(decimated_cube)
    
    def forward_freq_to_real(self, x_freq): # input in freq, output in real, costs 1
        assert x_freq.shape == self.ishape
        
        H_spec_x_freq = np.sum(
            self.H_spec_freq_full * x_freq[:, np.newaxis, ...], axis=0
        )  # (5, 300, 250, 251) * (5, 1, 250, 251) = (300, 250, 251))
        convoluted_cube = idft2(H_spec_x_freq)  # (300, 250, 500)

        # make decimated cube
        decimated_cube = convoluted_cube[
            :, :: self.di, :: self.dj
        ]  # (300, 50, 100)
        return decimated_cube

    def adjoint(self, y):
        assert y.shape == self.oshape
        
        # bourrage de zéros
        original_cube = np.zeros(
            (self.n_lamb, self.shape_target[0], self.shape_target[1])
        )  # (300, 250, 500)
        original_cube[:, :: self.di, :: self.dj] = y
        

        # make convolution with conjugated weighted psfs
        original_cube_freq = rdft2(original_cube)[np.newaxis, ...]  # (1, 300, 250, 251)
        # H_spec_x_freq = np.sum(
        #     np.conj(self.H_spec_freq) * original_cube_freq, axis=1
        # )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        H_spec_x_freq = einsum(
            np.conj(self.H_spec_freq) * original_cube_freq, "t l i j -> t i j"
        )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        maps = irdftn(H_spec_x_freq, self.shape_target)  # (5, 250, 500)
        
        return maps  # shape = 5, 250, 500
    
    def adjoint_real_to_freq_full(self, y):
        assert y.shape == self.oshape
        
        # bourrage de zéros
        original_cube = np.zeros(
            (self.n_lamb, self.shape_target[0], self.shape_target[1])
        )  # (300, 250, 500)
        original_cube[:, :: self.di, :: self.dj] = y

        # make convolution with conjugated weighted psfs
        original_cube_freq = dft2(original_cube)[np.newaxis, ...]  # (1, 300, 250, 251)
        # H_spec_x_freq = np.sum(
        #     np.conj(self.H_spec_freq_full) * original_cube_freq, axis=1
        # )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        
        H_spec_x_freq = einsum(
            np.conj(self.H_spec_freq_full) * original_cube_freq, "t l i j -> t i j"
        )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        
        return H_spec_x_freq  # shape = 5, 250, 500
    
    def adjoint_real_to_freq(self, y):
        assert y.shape == self.oshape
        
        # bourrage de zéros
        original_cube = np.zeros(
            (self.n_lamb, self.shape_target[0], self.shape_target[1])
        )  # (300, 250, 500)
        original_cube[:, :: self.di, :: self.dj] = y

        # make convolution with conjugated weighted psfs
        original_cube_freq = rdft2(original_cube)[np.newaxis, ...]  # (1, 300, 250, 251)
        # H_spec_x_freq = np.sum(
        #     np.conj(self.H_spec_freq_full) * original_cube_freq, axis=1
        # )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        
        H_spec_x_freq = einsum(
            np.conj(self.H_spec_freq) * original_cube_freq, "t l i j -> t i j"
        )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        
        return H_spec_x_freq  # shape = 5, 250, 500
    
    # def adjoint_freq_full(self, y): # return adjoint in fourier, and real = False
    #     assert y.shape == self.oshape
        
    #     # bourrage de zéros
    #     original_cube = np.zeros(
    #         (self.n_lamb, self.shape_target[0], self.shape_target[1])
    #     )  # (300, 250, 500)
    #     original_cube[:, :: self.di, :: self.dj] = y
        

    #     # make convolution with conjugated weighted psfs
    #     original_cube_freq = dft2(original_cube)[np.newaxis, ...]  # (1, 300, 250, 251)
    #     H_spec_x_freq = np.sum(
    #         np.conj(self.H_spec_freq_full) * original_cube_freq, axis=1
    #     )  # (5, 300, 250, 251) * (1, 300, 250, 251)
        
    #     return H_spec_x_freq  # shape = 5, 250, 500

    def fwadj(self, x):
        assert x.shape == self.ishape
        return apply_hessian2(self.hess_spec_freq, self.di, self.dj, self.shape_target, x)

class Imager_Model(LinOp):
    def __init__(
        self,
        psfs_monoch: np.ndarray,     # (L, H_psf, W_psf)
        L_pce: np.ndarray,           # (B, L)
        lamb_cube: np.ndarray,       # (L,)
        shape_target: tuple[int,int],# (H, W)
        n_spec: int,                 # number of spectral endmembers (S)
        di: int,
        dj: int,
        pixel_arcsec: float = 0.111
    ):
        # validate PSF fits in target
        assert psfs_monoch.shape[1] <= shape_target[0]
        assert psfs_monoch.shape[2] <= shape_target[1]

        self.shape_target = shape_target
        self.di = di
        self.dj = dj
        self.n_spec = n_spec
        self.lamb_cube = lamb_cube

        # number of spectral bands and wavelengths
        self.n_bands, n_lamb = L_pce.shape

        # 1) store PSFs with a spectral axis for broadcasting
        #    shape -> (1, 1, L, H_psf, W_psf)
        self.psfs = psfs_monoch[np.newaxis, np.newaxis, ...]

        # 2) unit-convert PCE, then reshape -> (B, 1, L, 1, 1)
        L_pce_conv = unit_conversion(L_pce, lamb_cube * 1e-6, pixel_arcsec)
        self.pce = L_pce_conv[:, np.newaxis, :, np.newaxis, np.newaxis]

        # 3) compute PCE normalization factors -> (B, 1, 1, 1)
        norms = trapezoid(
            L_pce_conv * lamb_cube[np.newaxis, :],
            x=lamb_cube,
            axis=1
        )
        self.pce_norms = norms[:, np.newaxis, np.newaxis, np.newaxis]

        # 4) lamb_cube in broadcast form -> (1, 1, L, 1, 1)
        self.lamb_cube = lamb_cube[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]

        # define input and output shapes for LinOp
        ishape = (self.n_spec, *self.shape_target)  # (S, H, W)
        oshape = (self.n_bands, *self.shape_target)  # (B, H, W)
        super().__init__( ishape=ishape, oshape=oshape )

    def _compute_H_freq_full(self, endmembers: np.ndarray) -> np.ndarray:
        """
        Compute the frequency-domain instrument response for given endmembers.

        Parameters
        ----------
        endmembers : (S, L) array of spectral signatures

        Returns
        -------
        H_freq_full : (B, S, H, W) complex frequency response
        """
        specs = endmembers[np.newaxis, :, :, np.newaxis, np.newaxis]  # (1, S, L, 1, 1)

        # multiply specs, PSFs, PCE, and wavelength term -> (B, S, L, H, W)
        stacked = specs * self.psfs * self.pce * self.lamb_cube

        # integrate along wavelength axis -> (B, S, H, W)
        H_int = trapezoid(
            stacked,
            x=self.lamb_cube,  # 1D array
            axis=2
        ) / self.pce_norms  # broadcast (B,1,1,1) -> (B,S,H,W)

        # convert to frequency domain
        return ir2fr(H_int, self.shape_target, real=False)

    def forward(self, abundances: np.ndarray, endmembers: np.ndarray) -> np.ndarray:
        """
        Real-space forward projection.

        Parameters
        ----------
        abundances : (S, H, W) abundance maps
        endmembers : (S, L) spectral signatures

        Returns
        -------
        y : (B, H, W) simulated measurements
        """
        Hf = self._compute_H_freq_full(endmembers)
        Xf = dft2(abundances)
        proj = np.sum(Hf * Xf[np.newaxis, ...], axis=1)
        return np.real(idft2(proj))

    def forward_freq_to_real(self, Xf: np.ndarray, endmembers: np.ndarray) -> np.ndarray:
        """
        From frequency-space abundances to real-space measurements.
        """
        Hf = self._compute_H_freq_full(endmembers)
        proj = np.sum(Hf * Xf[np.newaxis, ...], axis=1)
        return np.real(idft2(proj))

    def forward_freq_to_freq(self, Xf: np.ndarray, endmembers: np.ndarray) -> np.ndarray:
        """
        From frequency-space abundances to frequency-space measurements.
        """
        Hf = self._compute_H_freq_full(endmembers)
        return np.sum(Hf * Xf[np.newaxis, ...], axis=1)

    def adjoint(self, y: np.ndarray, endmembers: np.ndarray) -> np.ndarray:
        """
        Real-space adjoint projection.

        Parameters
        ----------
        y : (B, H, W) measurements
        endmembers : (S, L) spectral signatures

        Returns
        -------
        x_adjoint : (S, H, W) back-projected abundances
        """
        Hf = self._compute_H_freq_full(endmembers)
        Yf = dft2(y)
        back = np.sum(np.conj(Hf) * Yf[:, np.newaxis, ...], axis=0)
        return np.real(idft2(back))

    def adjoint_real_to_freq_full(self, y: np.ndarray, endmembers: np.ndarray) -> np.ndarray:
        """
        Real-space to full-frequency adjoint.
        """
        Hf = self._compute_H_freq_full(endmembers)
        return np.sum(np.conj(Hf) * dft2(y)[:, np.newaxis, ...], axis=0)

    def adjoint_real_to_freq(self, y: np.ndarray, endmembers: np.ndarray) -> np.ndarray:
        """
        Real-space to half-frequency adjoint.
        """
        Hf = self._compute_H_freq_full(endmembers)
        return np.sum(np.conj(Hf) * rdft2(y)[:, np.newaxis, ...], axis=0)

    def fwadj(self, abundances: np.ndarray, endmembers: np.ndarray) -> np.ndarray:
        """
        Hessian-like operation (forward+adjoint) via precomputed Hessian tensor.
        """
        Hf = self._compute_H_freq_full(endmembers)
        hess = compute_hess_mirim(Hf)
        part_hess = partitioning_hess_mirim_freq_full2(hess, self.di, self.dj)
        return apply_hessian2(part_hess, self.di, self.dj, self.shape_target, abundances)

