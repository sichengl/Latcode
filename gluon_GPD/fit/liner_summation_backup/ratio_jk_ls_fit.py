import os
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("/ccs/home/sicheng/Correlation_Analysis")
sys.path.append(os.path.join(os.path.dirname(__file__), "Correlation_Analysis-main"))
from module.resampling import *
import gvar as gv
import lsqfit as lsf


operator_list = ["O3","TXTX","TYTY","XYXY"]
tsep_list = [4, 5, 6, 7, 8,9]
w_list = [0, 1, 2, 3, 4,5,6,7,8,9]
tgf_list = [20,25,30,35,40]
cfg_list = list(range(204, 1404, 6))
tsrc_list = list(range(0, 96, 12))
psink_list = [0,1,2, 3, 4, 5]
psink_list = [x + 5 for x in psink_list]
Gt = 96
tins_skip=1

base_dir = os.path.dirname(os.path.abspath(__file__))
fit_output_dir = os.path.join(base_dir, "data_summation_jk_fit")
os.makedirs(fit_output_dir, exist_ok=True)
fit_output_path = os.path.join(fit_output_dir, "summation_jk_fit_dic.p")
n_cfg = len(cfg_list)
n_tsep = len(tsep_list)
n_psink = len(psink_list)
n_tgf = len(tgf_list)
n_w = len(w_list)
input_dir = os.path.join(base_dir, "data_ratio_jk_list")
input_path = os.path.join(input_dir, "ratio_jk_dic.p")
ratio_jk_ls_dic=gv.load(input_path)

#Now take each (operator,psink,tgf) do fit for (w,tsep)
#We use the summation method so tins index disappears before fitting

#Need to define two functions
#make_prior & fit_func
def make_prior(operator_name, psink, tgf):
    prior = gv.BufferDict()

    for w in w_list:
        prior[f"C{w}"] = gv.gvar(0.0, 10.0)
        prior[f"M{w}"] = gv.gvar(0.0, 10.0)

    return prior


def summation_fcn(x_values, p):
    y_model = []

    for w, tsep in x_values:
        y_model.append(p[f"C{w}"] + p[f"M{w}"] * tsep)

    return y_model

summation_fit_dic = {}

for operator_name in operator_list:
    ratio = ratio_jk_ls_dic[operator_name]
    for ipsink, psink in enumerate(psink_list):
        for itgf, tgf in enumerate(tgf_list):
            x_list = []
            y_list = []
            for w in w_list:
                for tsep in tsep_list:
                    key = f"tsep{tsep}_tgf{tgf}_psink{psink-5}_w{w}"
                    print(f"processing {key}")
                    ratio_tins_summed = np.sum(ratio[key][:,tins_skip:tsep+1-tins_skip],axis=1 )
                    x_list.append((w,tsep))
                    y_list.append(ratio_tins_summed)
            
            y_array = np.array(y_list)  # [n_data_point, n_jk]
            n_samples = y_array.shape[1]
            y_mean = np.mean(y_array, axis=1)
            delta = y_array - y_mean[:, None]
            cov = (n_samples - 1) / n_samples * (delta @ delta.T)
            
            M_jk_list = []
            C_jk_list = []
            Q_jk_list = []
            chi2_dof_jk_list = []
            
            for i_jk in range(n_samples):

                y_jk = y_array[:,i_jk]
                y_gvar = gv.gvar(y_jk,cov)
                fit = lsf.nonlinear_fit(
                    data=(x_list, y_gvar),
                    prior=make_prior(operator_name, psink, tgf),
                    fcn=summation_fcn,
                    maxit=10000,)

                C_jk_list.append([fit.p[f"C{w}"].mean for w in w_list])
                M_jk_list.append([fit.p[f"M{w}"].mean for w in w_list])
                Q_jk_list.append(float(fit.Q))
                chi2_dof_jk_list.append(float(fit.chi2 / fit.dof))
            
            result_key = f"{operator_name}_tgf{tgf}_psink{psink-5}"

            summation_fit_dic[result_key] = {
                "operator_name": operator_name,
                "psink": psink - 5,
                "tgf": tgf,
                "w_list": list(w_list),
                "tsep_list": list(tsep_list),
                "tins_skip": tins_skip,
                "x_list": list(x_list),
                "M_jk": np.array(M_jk_list),
                "C_jk": np.array(C_jk_list),
                "Q_jk": np.array(Q_jk_list),
                "chi2_dof_jk": np.array(chi2_dof_jk_list),
                "covariance_matrix": cov
                }

            print("saved in dictionary:", result_key)

gv.dump(summation_fit_dic, fit_output_path)
print("saved summation jk fit results to", fit_output_path)
