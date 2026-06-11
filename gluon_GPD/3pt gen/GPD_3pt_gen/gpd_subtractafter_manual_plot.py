import os
import sys
import gvar as gv
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("/ccs/home/sicheng/Correlation_Analysis")
sys.path.append(os.path.join(os.path.dirname(__file__), "Correlation_Analysis-main"))
from module.resampling import *

# p = p' + q (my convention)
tsep_list = [4, 5, 6, 7, 8]
w_list = [2]
tgf_list = [30]
cfg_list = list(range(204, 1404, 6))
tsrc_list = list(range(0, 96, 12))
psink_list = [3]
psink_list = [x + 5 for x in psink_list]
Gt = 96
tins_max = max(tsep_list) + 1

base_path = "/lustre/orion/lgt132/scratch/sicheng/GPD_calc"
conn_template = "./gpd_data/3pt_conn_inputs_man_cfg{cfg}.h5"
q_list = np.array(
    [[qx, qy, qz] for qx in [0, -1, 1] for qy in [0, -1, 1] for qz in [0, -1, 1]],
    dtype=np.int64,
)

os.makedirs("./plots", exist_ok=True)

n_cfg = len(cfg_list)
n_tsep = len(tsep_list)
n_psink = len(psink_list)
n_tgf = len(tgf_list)
n_w = len(w_list)
n_q = len(q_list)

# Shapes:
#   pt2_plot[cfg, psink, tsep_index]
#   pt2FF_O3_plot[cfg,tsep_index, tins<=max_tsep, psink, tgf_index, w_index,q]
#   FF_O3_plot[cfg, tins<=max_tsep, tgf_index, w_index,q]
#   no need to roll, origin is already tsep=0
pt2_plot = np.zeros((n_cfg, n_psink, n_tsep), dtype="<c16")
pt2FF_O3_plot = np.zeros(
    (n_cfg, n_tsep, tins_max, n_psink, n_tgf, n_w,n_q), dtype="<c16"
)
FF_O3_plot = np.zeros((n_cfg,  tins_max, n_tgf, n_w,n_q), dtype="<c16")


for i_cfg, cfg in enumerate(cfg_list):
    print(f"reading <2pt*O> , <2pt> and <O> cfg={cfg}")
    with h5py.File(conn_template.format(cfg=cfg), "r") as f:
        
        #All three are rolled by -tsrc and  averaged over tsrc
        pt2FF = f["pt2FF"]
        FF = f["FF"]
        pt2 = f["pt2"]
        pt2FF_slice = np.zeros((n_tsep,tins_max,n_psink,n_tgf,n_w,n_q))
        pt2_plot[i_cfg] = pt2[psink_list, :][:, tsep_list]
        pt2FF_slice[i_cfg] = pt2FF[tsep_list,...][:,:tins_max,...][:,:,:,psink_list,...][...,tgf_list,:,:][...,w_list,:]
            
        pt2FF_O3 = (
                -2 * pt2FF_slice[:, :, 0, :, :,:,:]
                + pt2FF_slice[:, :, 3, :, :,:,:]
                + pt2FF_slice[:, :, 4, :, :,:,:]
        )

        FF_slice = np.asarray(FF[:tins_max, :, :, :, :])
        FF_slice = FF_slice[:, :, tgf_list, :, :]
        FF_slice = FF_slice[:, :, :, w_list, :]
        FF_O3 = -2 * FF_slice[:, 0, :, :, :] + FF_slice[:, 3, :, :, :] + FF_slice[:, 4, :, :, :]
        FF_O3_plot[i_cfg] = np.moveaxis(FF_O3, -1, 0)


print("building jackknife lists")
jk_ls_pt2 = jackknife(pt2_plot)
jk_ls_pt2FF_O3 = jackknife(pt2FF_O3_plot)
jk_ls_FF_O3 = jackknife(FF_O3_plot)


for iq, qvec in enumerate(q_list):
    q_tuple = tuple(int(qi) for qi in qvec)
    q_label = f"q{q_tuple[0]}_{q_tuple[1]}_{q_tuple[2]}"
    for i_tgf, tgf in enumerate(tgf_list):
        for ipsink, psink in enumerate(psink_list):
            for iw, w in enumerate(w_list):
                print(f"processing q={q_tuple}, tgf={tgf}, psink={psink}, w={w}")

                plt.figure(figsize=(14, 8))

                for itsep, tsep in enumerate(tsep_list):
                    x_centered = np.arange(tsep + 1) - 0.5 * tsep

                    jk_2pt = jk_ls_pt2[:, ipsink, itsep]
                    jk_2ptFF = jk_ls_pt2FF_O3[:, iq, itsep, : tsep + 1, ipsink, i_tgf, iw]
                    jk_FF = jk_ls_FF_O3[:, iq, : tsep + 1, i_tgf, iw]

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
                    f"Connected-subtracted ratio; q={q_tuple}; psink={psink-5}; "
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
                    f"./plots/ratio_O3_conn_jklevel_man_{q_label}_p{psink-5}_"
                    f"w{w}_flow{tgf}_tsep{tsep_list}_cfgs{len(cfg_list)}_centered.png",
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=0.05,
                )
                plt.close()

