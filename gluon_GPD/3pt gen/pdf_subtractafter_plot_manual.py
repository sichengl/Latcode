import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("/ccs/home/sicheng/Correlation_Analysis")
sys.path.append(os.path.join(os.path.dirname(__file__), "Correlation_Analysis-main"))

from module.resampling import *


tsep_list = [4, 5, 6, 7, 8]
w_list = [0, 1, 2, 3, 4]
tgf_list = [30]
cfg_list = list(range(204, 1404, 6))
tsrc_list = list(range(0, 96, 12))
psink_list = [2, 3, 4, 5]
psink_list = [x + 5 for x in psink_list]
Gt = 96
tins_max = max(tsep_list) + 1

base_path = "/lustre/orion/lgt132/scratch/sicheng/GPD_calc"
conn_template = f"{base_path}/pdf_data/3pt_conn_inputs_man_cfg{{cfg}}.h5"

os.makedirs("./plots", exist_ok=True)

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
pt2_plot = np.zeros((n_cfg, n_psink, n_tsep), dtype="<c16")
pt2FF_O3_plot = np.zeros((n_cfg, n_tsep, tins_max, n_psink, n_tgf, n_w), dtype="<c16")
FF_O3_plot = np.zeros((n_cfg, tins_max, n_tgf, n_w), dtype="<c16")


for i_cfg, cfg in enumerate(cfg_list):
    print(f"reading <2pt*O> , <2pt> and <O> cfg={cfg}")
    with h5py.File(conn_template.format(cfg=cfg), "r") as f:
        
        #All three are rolled by -tsrc and  averaged over tsrc
        pt2FF = f["pt2FF"]
        FF = f["FF"]
        pt2 = f["pt2"]
        
        pt2_plot[i_cfg] = pt2[psink_list,:][:,tsep_list]
        for itsep, tsep in enumerate(tsep_list):
            # CO slice shape: [tins, psink, mu_diag, tgf, z_WL]
            pt2FF_slice = pt2FF[tsep, :tins_max, psink_list, :, :, :][...,tgf_list,:]
            
            #build O3 before jackknife
            # Shape after selection: [tins, psink, mu_diag, tgf_index, w_index]
            pt2FF_slice = pt2FF_slice[..., w_list]
            pt2FF_O3_plot[i_cfg, itsep] = (
                -2 * pt2FF_slice[:, :, 0, :, :]
                + pt2FF_slice[:, :, 3, :, :]
                + pt2FF_slice[:, :, 4, :, :]
            )

        # FF shape: [tins, mu_diag, tgf, z_WL].
        FF_slice = FF[:tins_max, :, tgf_list, :][..., w_list]
        FF_O3_plot[i_cfg] = -2 * FF_slice[:, 0, :, :] + FF_slice[:, 3, :, :] + FF_slice[:, 4, :, :]


print("building jackknife lists")
jk_ls_pt2 = jackknife(pt2_plot)
jk_ls_pt2FF_O3 = jackknife(pt2FF_O3_plot)
jk_ls_FF_O3 = jackknife(FF_O3_plot)


for i_tgf, tgf in enumerate(tgf_list):
    for ipsink, psink in enumerate(psink_list):
        for iw, w in enumerate(w_list):
            print(f"processing tgf={tgf}, psink={psink}, w={w}")

            plt.figure(figsize=(14, 8))

            for itsep, tsep in enumerate(tsep_list):
                x_centered = np.arange(tsep + 1) - 0.5 * tsep

                jk_2pt = jk_ls_pt2[:, ipsink, itsep]
                jk_2ptFF = jk_ls_pt2FF_O3[:, itsep, : tsep + 1, ipsink, i_tgf, iw]
                jk_FF = jk_ls_FF_O3[:, : tsep + 1, i_tgf, iw]

                jk_numerator = jk_2ptFF.real - jk_2pt.real[:, None] * jk_FF.real
                jk_ratio = jk_numerator / jk_2pt.real[:, None]
                ratio_avg = jk_ls_avg(jk_ratio)

                print(
                    f"  tsep={tsep}: mean real={np.mean(gv.mean(ratio_avg))}, "
                    f"mean err={np.mean(gv.sdev(ratio_avg))}"
                )

                plt.errorbar(
                    x_centered,
                    gv.mean(ratio_avg),
                    yerr=gv.sdev(ratio_avg),
                    capsize=3,
                    marker="o",
                    linestyle="-",
                    label=f"tsep={tsep}",
                )

            plt.axvline(0, color="k", linestyle="--", linewidth=1, alpha=0.4)
            plt.xlabel(r"Insertion Time ($t_{ins} - t_{sep}/2$)", fontsize=12)
            plt.ylabel(r"$(<C_2 O> - <C_2><O>) / <C_2>$", fontsize=12)
            plt.title(
                f"Connected-subtracted ratio; q=0; psink={psink-5}; "
                f"wilsonlen={w}; flow={tgf * 0.1:.2f}",
                fontsize=14,
            )

            plt.legend(
                loc="upper center",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=11,
                frameon=True,
            )

            max_tsep = max(tsep_list)
            plt.xlim(-0.5 * max_tsep - 0.5, 0.5 * max_tsep + 0.5)

            plt.savefig(
                f"./plots/ratio_O3_conn_jklevel_man_p{psink-5}_w{w}_flow{tgf}_"
                f"tsep{tsep_list}_cfgs{len(cfg_list)}_centered.png",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.05,
            )
            plt.close()
