import os
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("/ccs/home/sicheng/Correlation_Analysis")
sys.path.append(os.path.join(os.path.dirname(__file__), "Correlation_Analysis-main"))
from module.resampling import *


#This file reads 3pt, 2pt and FF of each cfg and make a jackknife sample list for them
#On the level of each jackknife sample, we do the fitting to get O00, and get a list of O00
#Then we do jackknife average for the list of O00, which can tackle the correlation between each measurement correctly

tsep_list = [4, 5, 6, 7, 8,9]
w_list = [0, 1, 2, 3, 4,5,6,7,8,9]
tgf_list = [20,25,30,35,40]
cfg_list = list(range(204, 1404, 6))
tsrc_list = list(range(0, 96, 12))
psink_list = [0,1,2, 3, 4, 5]
psink_list = [x + 5 for x in psink_list]
Gt = 96
tins_max = max(tsep_list) + 1

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "data_ratio_jk_list")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "ratio_jk_dic.p")
n_cfg = len(cfg_list)
n_tsep = len(tsep_list)
n_psink = len(psink_list)
n_tgf = len(tgf_list)
n_w = len(w_list)

# Shapes:
#   pt2_plot[cfg, psink, tsep_index]
#   pt2O_O3_plot[cfg, tsep_index, tins<=max_tsep, psink, tgf_index, w_index]
#   O_O3_plot[cfg, tins<=max_tsep, tgf_index, w_index]
#   no need to roll, origion is already tsep=0
pt2 = np.zeros((n_cfg, n_psink, n_tsep), dtype="<c16")
pt2O3 = np.zeros((n_cfg, n_tsep, tins_max, n_psink, n_tgf, n_w), dtype="<c16")
pt2TXTX = np.zeros((n_cfg, n_tsep, tins_max, n_psink, n_tgf, n_w),"<c16")
pt2TYTY = np.zeros((n_cfg, n_tsep, tins_max, n_psink, n_tgf, n_w),"<c16")
pt2XYXY = np.zeros((n_cfg, n_tsep, tins_max, n_psink, n_tgf, n_w),"<c16")
FF_O3 = np.zeros((n_cfg, tins_max, n_tgf, n_w), dtype="<c16")
FF_TX = np.zeros((n_cfg, tins_max, n_tgf, n_w), dtype="<c16")
FF_TY = np.zeros((n_cfg, tins_max, n_tgf, n_w), dtype="<c16")
FF_XY = np.zeros((n_cfg, tins_max, n_tgf, n_w), dtype="<c16")
for i_cfg, cfg in enumerate(cfg_list):
    print(f"reading <2pt*O> , <2pt> and <O> cfg={cfg}")
    pt3_dir=f"{base_dir}/pdf_data/3pt_conn_inputs_man_cfg{cfg}.h5"
    with h5py.File(pt3_dir, "r") as f:
        
        #All three are rolled by -tsrc and  averaged over tsrc
        pt2FF_read = f["pt2FF"]
        FF_read = f["FF"]
        pt2_read = f["pt2"]
        
        pt2[i_cfg] = pt2_read[psink_list,:][:,tsep_list]
        #BUILD 3PT
        for itsep, tsep in enumerate(tsep_list):
            # CO slice shape: [tins, psink, mu_diag, tgf, z_WL]
            pt2FF_slice = pt2FF_read[tsep, :tins_max, psink_list, :, :, :][...,tgf_list,:][...,w_list]
            # Shape after selection: [tins, psink, mu_diag, tgf_index, w_index]
            pt2TXTX[i_cfg,itsep] = pt2FF_slice[:,:,3,:,:]
            pt2TYTY[i_cfg,itsep] = pt2FF_slice[:,:,4,:,:]
            pt2XYXY[i_cfg,itsep] = pt2FF_slice[:,:,0,:,:]
        # FF shape: [tins, mu_diag, tgf, z_WL].
        FF_slice = FF_read[:tins_max, :, tgf_list, :][..., w_list]
        FF_TX[i_cfg] = FF_slice[:,3,:,:]
        FF_TY[i_cfg] = FF_slice[:,4,:,:]
        FF_XY[i_cfg] = FF_slice[:,0,:,:]
pt2O3 = -2*pt2XYXY + pt2TXTX + pt2TYTY
FF_O3 = -2*FF_XY + FF_TX + FF_TY
print("building jackknife lists")
jk_ls_pt2 = jackknife(pt2)
jk_ls_pt2O3 = jackknife(pt2O3)
jk_ls_pt2TXTX = jackknife(pt2TXTX)
jk_ls_pt2TYTY = jackknife(pt2TYTY)
jk_ls_pt2XYXY = jackknife(pt2XYXY)
jk_ls_FF_O3 = jackknife(FF_O3)
jk_ls_FF_TX = jackknife(FF_TX)
jk_ls_FF_TY = jackknife(FF_TY)
jk_ls_FF_XY = jackknife(FF_XY)


pt3_list = [jk_ls_pt2O3,jk_ls_pt2TXTX,jk_ls_pt2TYTY,jk_ls_pt2XYXY]
FF_list = [jk_ls_FF_O3,jk_ls_FF_TX,jk_ls_FF_TY,jk_ls_FF_XY]
operator_list = ["O3", "TXTX", "TYTY", "XYXY"]

ratio_jk_dic = {}
for operator_name in operator_list:
    ratio_jk_dic[operator_name] = {}
for i_tgf, tgf in enumerate(tgf_list):
    for ipsink, psink in enumerate(psink_list):
        for iw, w in enumerate(w_list):
            print(f"processing tgf={tgf}, psink={psink}, w={w}")

            for itsep, tsep in enumerate(tsep_list):
                for operator_name, pt3, FF in zip(operator_list, pt3_list, FF_list):
                    key = f"tsep{tsep}_tgf{tgf}_psink{psink-5}_w{w}"

                    jk_2pt = jk_ls_pt2[:, ipsink, itsep]
                    jk_3pt = pt3[:, itsep, : tsep + 1, ipsink, i_tgf, iw]
                    jk_FF  = FF[:, : tsep + 1, i_tgf, iw]

                    jk_numerator = jk_3pt - jk_2pt[:,None] * jk_FF
                    jk_ratio = ( jk_numerator / jk_2pt[:,None] ).real

                    ratio_jk_dic[operator_name][key] = jk_ratio   




gv.dump(ratio_jk_dic, output_path)

print("saved ratio jackknife dictionary to", output_path)
