#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 21:54:28 2025

@author: dpineau
"""

import numpy as np
import os
import time
from qmm import QuadObjective, lcg, Huber, pcg
from udft import idft2, dft2
from einops import einsum

main_path = "/home/plcristille/dev/JWST/fusion_inverse/MS-HS-Fusion-Study-Case"
os.chdir(main_path)

from instrument_models import concatenating2, Spectro_Model_3, partitioning_einops2, Mirim_Model_For_Fusion
from fusion_tools import Regul_Fusion_Model2, Inv_Regul_Fusion_Model2, Regul_Fusion_Model3, Inv_Regul_Fusion_Model3
from diff_operators import Difference_Operator_Joint, Difference_Operator_Sep_Freq, NpDiff_r, NpDiff_c




#%%

# intérêt par rapport à QuadCriterion: modèles mis à jour pour décimation libre
class QuadCriterion2:
    # y_imager must be compatible with fusion model, i.e. use mirim_model_for_fusion here
    def __init__(
        self,
        mu_imager,
        y_imager,
        model_imager: Mirim_Model_For_Fusion,
        mu_spectro,
        y_spectro,
        model_spectro: Spectro_Model_3,
        mu_reg,
        printing=False,
        gradient="separated",
    ):
        self.mu_imager = mu_imager
        self.y_imager = y_imager
        self.model_imager = model_imager
        self.mu_spectro = mu_spectro
        self.y_spectro = y_spectro
        self.model_spectro = model_spectro

        n_spec = model_imager.part_hess_mirim_freq_full.shape[0]
        self.n_spec = n_spec

        assert (
            type(mu_reg) == float
            or type(mu_reg) == int
            or type(mu_reg) == list
            or type(mu_reg) == np.ndarray
        )
        self.mu_reg = mu_reg
        if type(mu_reg) == list or type(mu_reg) == np.ndarray:
            assert len(mu_reg) == n_spec

        shape_target = model_imager.shape_target
        shape_of_output = (n_spec, shape_target[0], shape_target[1])
        self.shape_of_output = shape_of_output

        if gradient == "joint":
            diff_op_joint = Difference_Operator_Joint(shape_target)
            self.diff_op_joint = diff_op_joint
        elif gradient == "separated":
            npdiff_r = NpDiff_r(shape_of_output)
            self.npdiff_r = npdiff_r
            npdiff_c = NpDiff_c(shape_of_output)
            self.npdiff_c = npdiff_c

        if type(self.mu_reg) == list or type(self.mu_reg) == np.ndarray:
            L_mu = np.copy(self.mu_reg)
        elif type(self.mu_reg) == int or type(self.mu_reg) == float:
            L_mu = np.ones(self.n_spec) * self.mu_reg  # same mu for all maps
        self.L_mu = L_mu

        self.printing = printing
        self.gradient = gradient

    def run_expsol(self):
        if self.printing:
            # print("Preprocessing starts...")
            t1 = time.time()
        
        if type(self.mu_reg) == list or type(self.mu_reg) == np.ndarray:
            L_mu = np.copy(self.mu_reg)
        elif type(self.mu_reg) == int or type(self.mu_reg) == float:
            L_mu = np.ones(self.n_spec) * self.mu_reg  # same mu for all maps
        
        regul_fusion_model = Regul_Fusion_Model2(
            self.model_imager,
            self.model_spectro,
            L_mu,
            self.mu_imager,
            self.mu_spectro,
            gradient=self.gradient,
        )

        inv_fusion_model = Inv_Regul_Fusion_Model2(regul_fusion_model)

        if self.printing:
            t2 = time.time()
            time_preprocess = round(t2 - t1, 3)
            # print("Preprocessing ended in {} sec.".format(time_preprocess))

        res_with_all_data = inv_fusion_model.map_reconstruction(
            self.y_imager, self.y_spectro
        )

        if self.printing:
            t3 = time.time()
            time_all = round(t3 - t1, 3)
            time_calc = round(time_all - time_preprocess, 3)
            print(
                "Total time needed for expsol = {} + {} = {} sec.".format(
                    time_preprocess, time_calc, time_all
                )
            )

        return res_with_all_data

    def run_lcg(self, maximum_iterations, tolerance=1e-12, calc_crit = False, perf_crit = None, value_init = 0.5):
        assert type(self.mu_reg) == int or type(self.mu_reg) == float
        # lcg codé que avec un hyper paramètre

        init = np.ones(self.shape_of_output) * value_init

        # t1 = time.time()

        imager_data_adeq = QuadObjective(
            self.model_imager.forward,
            self.model_imager.adjoint,
            hessp=self.model_imager.fwadj,
            data=self.y_imager,
            hyper=self.mu_imager,
            name="Imager",
        )

        spectro_data_adeq = QuadObjective(
            self.model_spectro.forward,
            self.model_spectro.adjoint,
            hessp=self.model_spectro.fwadj,
            data=self.y_spectro,
            hyper=self.mu_spectro,
            name="Spectro",
        )

        if self.gradient == "joint":  # regularization term with joint gradients
            prior = QuadObjective(
                self.diff_op_joint.D,
                self.diff_op_joint.D_t,
                self.diff_op_joint.DtD,
                hyper=self.mu_reg,
                name="Reg joint",
            )

        elif self.gradient == "separated":
            prior_r = QuadObjective(
                self.npdiff_r.forward,
                self.npdiff_r.adjoint,
                hyper=self.mu_reg,
            )
            prior_c = QuadObjective(
                self.npdiff_c.forward,
                self.npdiff_c.adjoint,
                hyper=self.mu_reg,
            )
            prior = prior_r + prior_c

        self.L_crit_val_lcg = []
        self.L_perf_crit = []
        self.L_x = []
        
        def perf_crit_for_lcg(res_lcg):
            x_hat = res_lcg.x.reshape(self.shape_of_output)
            self.L_perf_crit.append(perf_crit(x_hat))
            
            # self.L_x.append(x_hat)
        
        t1 = time.time()

        if calc_crit and perf_crit == None:
            print("LCG : Criterion calculated at each iteration!")
            # res_lcg = lcg(
            #     imager_data_adeq + spectro_data_adeq + prior,
            #     init,
            #     tol=tolerance,
            #     max_iter=maximum_iterations,
            #     calc_objv = True
            # )
            res_lcg = lcg(
                imager_data_adeq + spectro_data_adeq + prior,
                init,
                tol=tolerance,
                max_iter=maximum_iterations,
                callback = self.crit_val_for_lcg
            )
        elif calc_crit == False and perf_crit != None:
            print("LCG : perf_crit calculated at each iteration!")
            res_lcg = lcg(
                imager_data_adeq + spectro_data_adeq + prior,
                init,
                tol=tolerance,
                max_iter=maximum_iterations,
                callback = perf_crit_for_lcg
            )
        elif calc_crit and perf_crit != None:
            print("Criterion to calculate AND performance criterion to calculate ?")
            return None
        elif calc_crit == False and perf_crit == None:
            res_lcg = lcg(
                imager_data_adeq + spectro_data_adeq + prior,
                init,
                tol=tolerance,
                max_iter=maximum_iterations
            )
        if self.printing:
            print("Total time needed for LCG :", round(time.time() - t1, 3))
        
        # running_time = time.time() - t1

        return res_lcg  # , running_time
    
    def run_pcg(self, maximum_iterations, tolerance=1e-12, calc_crit = False, perf_crit = None, value_init = 0.5):
        assert type(self.mu_reg) == int or type(self.mu_reg) == float
        
        regul_fusion_model = Regul_Fusion_Model2(
            self.model_imager,
            self.model_spectro,
            self.L_mu,
            self.mu_imager,
            self.mu_spectro,
            gradient=self.gradient,
        )
        
        di, dj = self.model_spectro.di, self.model_spectro.dj
        shape_target = self.model_spectro.shape_target
        mat_Q = regul_fusion_model.regul_hess_fusion
        def apply_Q(q_freq):
            # partitionnement de x
            part_x_freq = partitioning_einops2(q_freq, di, dj)  # (5, 25, 50, 100)

            # produit de HtH avec x
            HtH_x_freq = mat_Q * part_x_freq[np.newaxis, :, np.newaxis, :, :, :]
            # (5, 5, 25, 25, 50, 100) * (1, 5, 1, 25, 50, 100) = (5, 5, 25, 25, 50, 100)

            HtH_x_freq_sum = einsum(HtH_x_freq, "ti tj di dj h w -> ti di h w")

            # reconstitution des cartes en freq
            concat_HtH_x_freq = concatenating2(HtH_x_freq_sum, shape_target, di, dj)
            # (5, 25, 50, 100) --> (5, 250, 500) --> (5, 250, 251)

            return concat_HtH_x_freq
            
        mirim_adjoint_data = self.model_imager.adjoint(self.y_imager)
        spectro_adjoint_data = self.model_spectro.adjoint(self.y_spectro)
        mat_q = dft2(self.mu_imager * mirim_adjoint_data + self.mu_spectro * spectro_adjoint_data)
        
        init_freq = dft2(np.ones(self.shape_of_output) * value_init)

        # t1 = time.time()

        self.L_crit_val_lcg = []
        self.L_perf_crit = []
        
        # def perf_crit_for_lcg(res_lcg):
        #     x_hat = res_lcg.x.reshape(self.shape_of_output)
        #     self.L_perf_crit.append(perf_crit(x_hat))
        
        if self.printing:
            t1 = time.time()
        
        res_pcg = pcg(
            normalp=apply_Q,
            secondm=mat_q,
            x0=init_freq,
            tol=tolerance,
            max_iter=maximum_iterations
            # callback = perf_crit_for_lcg
        )

        if self.printing:
            print("Total time needed for PCG :", round(time.time() - t1, 3))
        
        # running_time = time.time() - t1

        return res_pcg

    def crit_val(self, x_hat):
        data_term_imager = self.mu_imager * np.sum(
            (self.y_imager - self.model_imager.forward(x_hat)) ** 2
        )
        data_term_spectro = self.mu_spectro * np.sum(
            (self.y_spectro - self.model_spectro.forward(x_hat)) ** 2
        )

        if self.gradient == "joint":
            regul_term = self.mu_reg * np.sum((self.diff_op_joint.D(x_hat)) ** 2)
            
        elif self.gradient == "separated":
            regul_term = self.mu_reg * (
                np.sum(
                    self.npdiff_r.forward(x_hat) ** 2
                    + self.npdiff_c.forward(x_hat) ** 2
                )
            )

        J_val = (data_term_imager + data_term_spectro + regul_term) / 2
        # on divise par 2 par convention, afin de ne pas trouver un 1/2 dans le calcul de dérivée

        return J_val
    
    def crit_val_for_lcg(self, res_lcg):
        x_hat = res_lcg.x.reshape(self.shape_of_output)
        self.L_crit_val_lcg.append(self.crit_val(x_hat))


def mu_instru(std):
    return 1 / (2 * (std**2))



# comme optimal_res_dicho2, mais sans la recursivité, donc plus simple et optimisée
def optimal_res_dicho_quad2(
    criterion,
    quality_metrics_function,
    list_of_mu,
    printing=True,
    max_iter=30,
    tolerance=0.01,
    method_for_criterion = "exp",
    max_iter_lcg = 200
):
    assert len(list_of_mu) == 2

    threshold = list_of_mu[0] * tolerance

    def moy(a, b):
        return (a + b) / 2

    new_list_of_mu = [list_of_mu[0], moy(list_of_mu[0], list_of_mu[1]), list_of_mu[1]]

    def make_new_list_of_mu(list_of_mu):
        a, b, c = list_of_mu
        return [a, moy(a, b), b, moy(b, c), c]

    new_list_of_mu = make_new_list_of_mu(new_list_of_mu)

    if printing == True:
        print(new_list_of_mu)

    dico_mu_metric = {}
    n_iter = 0

    while (
        abs(new_list_of_mu[0] - new_list_of_mu[-1]) > threshold and n_iter <= max_iter
    ):
        if printing == True:
            print("Itération n°{}".format(n_iter + 1))
        L_metric = []
        for mu in new_list_of_mu:
            criterion.mu_reg = mu

            if mu in list(dico_mu_metric.keys()):
                L_metric.append(dico_mu_metric[mu])
                continue

            if method_for_criterion == "exp":
                res_maps = criterion.run_expsol()
            elif method_for_criterion == "lcg":
                res_maps = criterion.run_lcg(max_iter_lcg).x
            metric_value = quality_metrics_function(res_maps)
            L_metric.append(metric_value)
            dico_mu_metric[mu] = metric_value

        index_of_best_metric = L_metric.index(np.min(L_metric))

        if printing == True:
            print("L_metric", L_metric)
            print("index_of_best_metric", index_of_best_metric)

        # last_b = new_list_of_mu[2]
        # print(last_b)

        if index_of_best_metric == 0:
            del new_list_of_mu[2]
            del new_list_of_mu[2]
            del new_list_of_mu[2]
            new_list_of_mu = [new_list_of_mu[0] / 10] + new_list_of_mu
        elif index_of_best_metric == 1:
            del new_list_of_mu[3]  # before last
            del new_list_of_mu[3]  # last
        elif index_of_best_metric == 2:
            del new_list_of_mu[0]  # first
            del new_list_of_mu[3]  # last
        elif index_of_best_metric == 3:
            del new_list_of_mu[0]  # first
            del new_list_of_mu[0]  # first
        elif index_of_best_metric == 4:
            del new_list_of_mu[0]
            del new_list_of_mu[0]
            del new_list_of_mu[0]
            new_list_of_mu = new_list_of_mu + [new_list_of_mu[-1] * 10]

        new_list_of_mu = make_new_list_of_mu(new_list_of_mu)
        threshold = new_list_of_mu[2] * tolerance
        if printing == True:
            print("new_list_of_mu :", new_list_of_mu)
            print("threshold", threshold)
            print(
                "new_list_of_mu[0] - new_list_of_mu[-1]",
                abs(new_list_of_mu[0] - new_list_of_mu[-1]),
            )
            print("\n")
        n_iter += 1

    index_of_best_metric = L_metric.index(np.min(L_metric))
    best_metric = L_metric[index_of_best_metric]
    best_mu = new_list_of_mu[index_of_best_metric]

    return best_metric, best_mu

#%%


class SemiQuad_Criterion_Fusion:
    # y_imager must be compatible with fusion model, i.e. use mirim_model_for_fusion here
    def __init__(
        self,
        mu_imager,
        y_imager,
        model_imager: Mirim_Model_For_Fusion,
        mu_spectro,
        y_spectro,
        model_spectro: Spectro_Model_3,
        mu_reg,
        thresh,
        scale = 0.2,
        printing=False,
    ):
        self.mu_imager = mu_imager
        self.y_imager = y_imager
        self.model_imager = model_imager

        self.mu_spectro = mu_spectro
        self.y_spectro = y_spectro
        self.model_spectro = model_spectro
        
        n_spec = model_imager.part_hess_mirim_freq_full.shape[0]
        self.n_spec = n_spec

        assert type(mu_reg) == float or type(mu_reg) == int or type(mu_reg) == list or type(mu_reg) == np.ndarray
        if type(mu_reg) == list or type(mu_reg) == np.ndarray:
            assert len(mu_reg) == n_spec
        self.mu_reg = mu_reg

        shape_target = model_imager.shape_target
        shape_of_output = (n_spec, shape_target[0], shape_target[1])
        self.shape_of_output = shape_of_output

        self.dosf = Difference_Operator_Sep_Freq(shape_target) # input and output in freq: faster
        # self.dos = Difference_Operator_Sep(shape_target) # input and output in space: slower, use of rdft2 and irdftn
        self.npdiff_r = NpDiff_r(shape_of_output)
        self.npdiff_c = NpDiff_c(shape_of_output)

        self.thresh = thresh
        print("Huber threshold =", thresh)

        # fonction huber de base qui rajoute (x0.5) par rapport aux définitions d'Amine
        self.huber = lambda x: 2 * Huber(thresh).value(x)
        self.dhuber = lambda x: 2 * Huber(thresh).gradient(x)

        # scale parameter, 0 <= scale <= 0.5
        self.scale = scale
        print("Parameter scale =", scale)
        print("\n")

        self.printing = printing
        
    # def J_expsol(self, x_hat_freq, br_freq, bc_freq): # FAUX, ICI ON CALCULE CRITERE DU PREMIER PB QUADRATIQUE
    #     # si dft2 ou idft2 comptent pour 1 temps de calcul, et rdft2 ou irdftn comptent pour 0.5 temps de calcul
    #     # alors faire tous les calculs dans Fourier ici coûte 4. Si on veut faire tous les calculs dans le domaine spatial, ça coûte 7.
    #     # donc on passe tout dans Fourier (besoin de rajouter forward freq pour model imageur et modèle spectro)
    #     data_term_imager_freq = self.mu_imager * np.sum(
    #         (dft2(self.y_imager) - self.model_imager.forward_freq(x_hat_freq)) ** 2
    #     )
    #     data_term_spectro_freq = self.mu_spectro * np.sum(
    #         (dft2(self.y_spectro) - self.model_spectro.forward_freq(x_hat_freq)) ** 2
    #     )
    #     coeff = self.mu_reg / (2 * self.scale)
    #     regul_term_freq =  coeff * (np.sum((self.dosf.D1(x_hat_freq) - br_freq) ** 2) + np.sum((self.dosf.D2(x_hat_freq) - bc_freq) ** 2))

    #     J_exp = (data_term_imager_freq + data_term_spectro_freq + regul_term_freq) / 2
    #     # on divise par 2 par convention, afin de ne pas trouver un 1/2 dans le calcul de dérivée

    #     return np.abs(J_exp)
    
    def crit_val(self, x_hat_freq): # besoin de tout passer dans le domaine spatial pour que les Huber s'appliquent
        data_term_imager = self.mu_imager * np.sum(
            (self.y_imager - self.model_imager.forward_freq_to_real(x_hat_freq)) ** 2
        )
        data_term_spectro = self.mu_spectro * np.sum(
            (self.y_spectro - self.model_spectro.forward_freq_to_real(x_hat_freq)) ** 2
        )
        
        x_hat = np.real(idft2(x_hat_freq))
        
        Dr_x_hat = self.npdiff_r.forward(x_hat)
        Dc_x_hat = self.npdiff_c.forward(x_hat)
        
        regul_term = self.mu_reg * (np.sum(self.huber(Dr_x_hat) + self.huber(Dc_x_hat)))
        
        J_val = (data_term_imager + data_term_spectro + regul_term) / 2
        # on divise par 2 par convention, afin de ne pas trouver un 1/2 dans le calcul de dérivée

        return np.abs(J_val)

    # def calc_br_bc_freq(self, x_freq): # pénalisation de Huber dans fourier, marche moins bien mais +rapide
    #     Drx_freq = self.dosf.D1(x_freq)
    #     Dcx_freq = self.dosf.D2(x_freq)

    #     br_freq = Drx_freq - self.scale * self.dhuber(Drx_freq)
    #     bc_freq = Dcx_freq - self.scale * self.dhuber(Dcx_freq)

    #     return br_freq, bc_freq
    
    # def calc_br_bc_freq(self, x_freq): # input and output in freq
    #     x = np.real(idft2(x_freq))
        
    #     Drx = self.npdiff_r.forward(x)
    #     Dcx = self.npdiff_c.forward(x)

    #     br = Drx - self.scale * self.dhuber(Drx)
    #     bc = Dcx - self.scale * self.dhuber(Dcx)

    #     return dft2(br), dft2(bc)
    
    # input and output in positions, plus rapide que calc_br_bc_freq
    # gros gain de temps à ne jamais faire passage ou sortie de Fourier sur les br ou bc, go le faire sur x uniquement
    def calc_br_bc(self, x):
        Drx = self.npdiff_r.forward(x)
        Dcx = self.npdiff_c.forward(x)

        br = Drx - self.scale * self.dhuber(Drx)
        bc = Dcx - self.scale * self.dhuber(Dcx)

        return br, bc

    def run_expsol(self, n_iter_max=100, diff_min=1e-7, calc_crit = False, perf_crit = None):  # Geman & Yang
    
        # if self.printing:
        #     print("Preprocessing starts...")
        t1 = time.time()
        
        # NE PAS BOUGER LA DÉFINITION DE L_MU D'ICI. L_MU DOIT ETRE REDÉFINI ICI, SINON INCOMPATIBLE AVEC OPTIMAL_RES_DICHO
        if type(self.mu_reg) == list:
            L_mu = np.array(self.mu_reg)
        elif type(self.mu_reg) == np.ndarray:
            L_mu = np.copy(self.mu_reg)
        elif type(self.mu_reg) == int or type(self.mu_reg) == float:
            L_mu = np.ones(self.n_spec) * self.mu_reg  # same mu for all maps
        
        # coefficient multiplying (D^T br + D^T bc) in q, one coeff for each abundance maps
        L_coeff = L_mu[:, np.newaxis, np.newaxis] / (2 * self.scale)

        # instanciate regularised fusion model
        regul_fusion_model = Regul_Fusion_Model3(
            self.model_imager,
            self.model_spectro,
            L_mu,
            self.mu_imager,
            self.mu_spectro,
            self.scale,
        )

        # preprocess calculations for q (sum of adjoint data)
        # (pourrait être optimisé en évitant que adjoint ne sorte de Fourier,
        # ce qui force à rerentrer dans Fourier avec dft2: un aller retour à gagner
        # En fait pas de gain de temps, car on fait du fft avec real = False)
        # mirim_adjoint_data = regul_fusion_model.mirim_model_for_fusion.adjoint_freq_full(
        #     self.y_imager
        # )
        # spectro_adjoint_data = regul_fusion_model.spectro_model.adjoint_freq_full(self.y_spectro)
        # sum_adjoint_data_freq = (
        #     self.mu_imager * mirim_adjoint_data + self.mu_spectro * spectro_adjoint_data
        # )
        # inv_fusion_model = Inv_Regul_Fusion_Model3(regul_fusion_model)
        
        mirim_adjoint_data = regul_fusion_model.mirim_model_for_fusion.adjoint(
            self.y_imager
        )
        spectro_adjoint_data = regul_fusion_model.spectro_model.adjoint(self.y_spectro)
        sum_adjoint_data = (
            self.mu_imager * mirim_adjoint_data + self.mu_spectro * spectro_adjoint_data
        )
        # sum_adjoint_data_freq = dft2(sum_adjoint_data)
        inv_fusion_model = Inv_Regul_Fusion_Model3(regul_fusion_model)

        t2 = time.time()
        time_preprocess = round(t2 - t1, 3)
        
        if self.printing:
            # print("Preprocessing ended in {} sec.".format(time_preprocess))
            # print("Start of iterations...")
            t1 = time.time()
        self.time_preprocess = time_preprocess

        # instanciating abundance maps
        rec_maps_freq = dft2(np.ones(self.shape_of_output, dtype=np.complex128) * 0.5)
        previous_rec_maps_freq = np.zeros(self.shape_of_output, dtype=np.complex128)
        diff = 1e30
        # previous_diff = 2e30
        L_diff = []
        
        # self.L_x = []
        
        if calc_crit:
            print("semiquad_gy_criterion : value of criterion calculated at each iteration!")
            L_crit = []
            
        if perf_crit != None:
            print("semiquad_gy_criterion : perf_crit calculated at each iteration!")
            L_perf_crit = []

        convergence = False
        n_iter = 0
        while n_iter < n_iter_max and diff > diff_min: #and previous_diff > diff:
            # VERSION DANS FOURIER
            # calculate auxiliary variables
            # br_freq, bc_freq = self.calc_br_bc_freq(rec_maps_freq)

            # calculate q
            # q_freq = sum_adjoint_data_freq + L_coeff * (
            #     self.dosf.D1_t(br_freq) + self.dosf.D2_t(bc_freq)
            # )
            
            # calculate a
            # rec_maps_freq = inv_fusion_model.apply_Qinv_freq(q_freq)
            
            
            
            # VERSION DANS POSITIONS (plus besoin de calculer sum_adjoint_data_freq)
            # calculate auxiliary variables
            br, bc = self.calc_br_bc(np.real(idft2(rec_maps_freq)))
            
            # calculate q
            q = sum_adjoint_data + L_coeff * (
                self.npdiff_r.adjoint(br) + self.npdiff_c.adjoint(bc)
            )

            # calculate a
            rec_maps_freq = inv_fusion_model.apply_Qinv_freq(dft2(q))
            
            
            # previous_diff = diff

            diff = np.abs(np.sum((rec_maps_freq - previous_rec_maps_freq) ** 2))
            # print("diff", diff)
            L_diff.append(diff)

            previous_rec_maps_freq = rec_maps_freq
            
            # print("iteration {}, diff = {}".format(n_iter, diff))
            
            if calc_crit:
                # if n_iter % 2 == 0:
                L_crit.append(self.crit_val(rec_maps_freq))
            
            if perf_crit != None:
                L_perf_crit.append(perf_crit(np.real(idft2(rec_maps_freq))))
                
            # self.L_x.append(np.real(idft2(rec_maps_freq)))

            n_iter += 1
            # print("n_iter", n_iter)
            if n_iter % (n_iter_max // 10) == 0:
                # n_iter_div_10 = n_iter // 10
                # if np.mean(L_diff[-2*n_iter_div_10 : -n_iter_div_10]) <= np.mean(L_diff[-n_iter_div_10:]):
                #     convergence = True
                #     break
                
                if self.printing:
                    print("iteration {}, diff = {}".format(n_iter, diff))
                
            

        if self.printing:
            t2 = time.time()
            time_iterations = round(t2 - t1, 3)
            print("Reconstruction ended")
            print(
                "Number of executed iterations : {} ({} sec)".format(
                    n_iter, time_iterations
                )
            )
            print(
                "Total time needed = {} + {} = {} sec.".format(
                    time_preprocess, time_iterations, time_preprocess + time_iterations
                )
            )
            print("Iterations ended due to :")
            if n_iter == n_iter_max:
                print("Maximum number of iterations reached. Convergence not ensured.")
            if diff < diff_min:
                print("Difference between iterated solutions < threshold ({}).".format(diff_min))
            if convergence:
                print("Difference between iterated solutions converged. Convergence ensured, and optimal solution found.")
            # if previous_diff < diff:
            #     print("Difference between iterated solutions increased. Convergence not ensured.")
            # else:
            #     print("Problem occured in SemiQuad_Criterion_Fusion.run_expsol.")
            print("\n")

        rec_maps = np.real(idft2(rec_maps_freq))
        self.L_diff = L_diff
        
        if np.mean(rec_maps) < 1:
            print("Careful: np.mean(rec_maps) < 1. Huber probably didn't work properly.")
            
        if calc_crit:
            self.L_crit = L_crit
        
        if perf_crit != None:
            self.L_perf_crit = L_perf_crit

        return rec_maps



def optimal_res_dicho_semiquad2(
    semiquad_criterion,
    quality_metrics_function,
    list_of_mu,
    printing=True,
    max_iter=30,
    tolerance=0.01,
    n_iter_max_crit=100,
    diff_min=1e-7,
):
    assert len(list_of_mu) == 2

    threshold = list_of_mu[0] * tolerance

    def moy(a, b):
        return (a + b) / 2

    new_list_of_mu = [list_of_mu[0], moy(list_of_mu[0], list_of_mu[1]), list_of_mu[1]]

    def make_new_list_of_mu(list_of_mu):
        a, b, c = list_of_mu
        return [a, moy(a, b), b, moy(b, c), c]

    new_list_of_mu = make_new_list_of_mu(new_list_of_mu)

    if printing == True:
        print(new_list_of_mu)

    dico_mu_metric = {}
    n_iter = 0

    while (
        abs(new_list_of_mu[0] - new_list_of_mu[-1]) > threshold and n_iter <= max_iter
    ):
        if printing == True:
            print("Itération n°{}".format(n_iter + 1))
        L_metric = []
        for mu in new_list_of_mu:
            semiquad_criterion.mu_reg = mu

            if mu in list(dico_mu_metric.keys()):
                L_metric.append(dico_mu_metric[mu])
                continue

            res_maps = semiquad_criterion.run_expsol(
                n_iter_max=n_iter_max_crit, diff_min=diff_min
            )
            metric_value = quality_metrics_function(res_maps)
            L_metric.append(metric_value)
            dico_mu_metric[mu] = metric_value

        index_of_best_metric = L_metric.index(np.min(L_metric))

        if printing == True:
            print("L_metric", L_metric)
            print("index_of_best_metric", index_of_best_metric)

        # last_b = new_list_of_mu[2]
        # print(last_b)

        if index_of_best_metric == 0:
            del new_list_of_mu[2]
            del new_list_of_mu[2]
            del new_list_of_mu[2]
            new_list_of_mu = [new_list_of_mu[0] / 10] + new_list_of_mu
        elif index_of_best_metric == 1:
            del new_list_of_mu[3]  # before last
            del new_list_of_mu[3]  # last
        elif index_of_best_metric == 2:
            del new_list_of_mu[0]  # first
            del new_list_of_mu[3]  # last
        elif index_of_best_metric == 3:
            del new_list_of_mu[0]  # first
            del new_list_of_mu[0]  # first
        elif index_of_best_metric == 4:
            del new_list_of_mu[0]
            del new_list_of_mu[0]
            del new_list_of_mu[0]
            new_list_of_mu = new_list_of_mu + [new_list_of_mu[-1] * 10]

        new_list_of_mu = make_new_list_of_mu(new_list_of_mu)
        threshold = new_list_of_mu[2] * tolerance
        if printing == True:
            print("new_list_of_mu :", new_list_of_mu)
            print("threshold", threshold)
            print(
                "new_list_of_mu[0] - new_list_of_mu[-1]",
                abs(new_list_of_mu[0] - new_list_of_mu[-1]),
            )
            print("\n")
        n_iter += 1

    index_of_best_metric = L_metric.index(np.min(L_metric))
    best_metric = L_metric[index_of_best_metric]
    best_mu = new_list_of_mu[index_of_best_metric]

    return best_metric, best_mu
