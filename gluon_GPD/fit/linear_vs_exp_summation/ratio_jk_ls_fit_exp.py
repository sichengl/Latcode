import os
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import lsqfit as lsf

import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("/ccs/home/sicheng/Correlation_Analysis")
sys.path.append(os.path.join(os.path.dirname(__file__), "Correlation_Analysis-main"))
from module.resampling import *
import gvar as gv
import lsqfit as lsf
import multiprocessing as mp


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
        prior[f"C0{w}"] = gv.gvar(0.0, 10.0)
        prior[f"M{w}"] = gv.gvar(0.0, 10.0)
        prior[f"C1{w}"] = gv.gvar(0.0,10.0)
        prior[f"C2{w}"] = gv.gvar(0.0,10.0)
    prior[f"logdE"]   = gv.gvar(-0.5,1)
    return prior


def summation_fcn(x_values, p):
    dE = gv.exp(p["logdE"])
    y_model = []

    for w, tsep in x_values:
        exp_term = np.exp(-dE * tsep) #gv or np ?
        y_model.append(
            p[f"C0{w}"]
            + p[f"M{w}"] * tsep
            + (p[f"C1{w}"] * tsep + p[f"C2{w}"]) * exp_term
        )

    return y_model

def fit_func(task):
    operator_name, psink, tgf = task
    ratio = ratio_jk_ls_dic[operator_name]
            
    x_list = []
    y_list = []
    for w in w_list:
        for tsep in tsep_list:
            key = f"tsep{tsep}_tgf{tgf}_psink{psink-5}_w{w}"
            print(f"processing {key}")
            ratio_tins_summed = np.sum(ratio[key][:,tins_skip:tsep+1-tins_skip],axis=1 ).real #take real part of summed ratio
            x_list.append((w,tsep))
            y_list.append(ratio_tins_summed)
    
    y_array = np.array(y_list)  # [n_data_point, n_jk]
    n_samples = y_array.shape[1]
    y_mean = np.mean(y_array, axis=1)
    delta = y_array - y_mean[:, None]
    cov = (n_samples - 1) / n_samples * (delta @ delta.T)
    
    M_jk_list = []
    C0_jk_list = []
    C1_jk_list = []
    C2_jk_list = []
    Q_jk_list = []
    chi2_dof_jk_list = []
    logdE_jk_list = []
    dE_jk_list = [] 
    for i_jk in range(n_samples):

        y_jk = y_array[:,i_jk]
        y_gvar = gv.gvar(y_jk,cov)
        fit = lsf.nonlinear_fit(
            data=(x_list, y_gvar),
            prior=make_prior(operator_name, psink, tgf),
            fcn=summation_fcn,
            maxit=10000,)
        print(f"In sample #{i_jk}, operator={operator_name}, psink={psink-5}, tgf={tgf}")
        print(fit)
        
        C0_jk_list.append([fit.p[f"C0{w}"].mean for w in w_list])
        C1_jk_list.append([fit.p[f"C1{w}"].mean for w in w_list])
        C2_jk_list.append([fit.p[f"C2{w}"].mean for w in w_list])
        M_jk_list.append([fit.p[f"M{w}"].mean for w in w_list])
        Q_jk_list.append(float(fit.Q))
        chi2_dof_jk_list.append(float(fit.chi2 / fit.dof))
        logdE_jk_list.append(fit.p["logdE"].mean)
        dE_jk_list.append(gv.exp(fit.p["logdE"]).mean)
    result_key = f"{operator_name}_tgf{tgf}_psink{psink-5}"

    result_dic = {
        "operator_name": operator_name,
        "psink": psink - 5,
        "tgf": tgf,
        "w_list": list(w_list),
        "tsep_list": list(tsep_list),
        "tins_skip": tins_skip,
        "x_list": list(x_list),
        "M_jk": np.array(M_jk_list),
        "C0_jk": np.array(C0_jk_list),
        "C1_jk": np.array(C1_jk_list),
        "C2_jk": np.array(C2_jk_list),
        "logdE_jk": np.array(logdE_jk_list),
        "dE_jk": np.array(dE_jk_list),
        "Q_jk": np.array(Q_jk_list),
        "chi2_dof_jk": np.array(chi2_dof_jk_list),
        "covariance_matrix": cov
        }

    return result_key,result_dic


if __name__ == "__main__":
    tasks = []

    for operator_name in operator_list:
        for psink in psink_list:
            for tgf in tgf_list:
                tasks.append((operator_name, psink, tgf))

    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    summation_fit_dic = {}

    with mp.Pool(processes=nproc) as pool:
        for result_key, result_dic in pool.imap_unordered(fit_func, tasks):
            summation_fit_dic[result_key] = result_dic
            print("finished:", result_key, flush=True)

    gv.dump(summation_fit_dic, fit_output_path)
    print("saved summation jk fit results to", fit_output_path)
