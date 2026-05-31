import os
import sys

import gvar as gv
import h5py
import numpy as np

sys.path.append("/ccs/home/sicheng/Correlation_Analysis")
sys.path.append(os.path.join(os.path.dirname(__file__), "Correlation_Analysis-main"))

from module.resampling import *


pt2_tsep_list = list(range(0, 15))
tsep_list = [4, 5, 6, 7, 8]
w_list = [0, 1, 2, 3, 4]
tgf_list = [30]
cfg_list = list(range(204, 1404, 6))
tsrc_list = list(range(0, 96, 12))
psink_list = [0,1,2, 3, 4, 5]
psink_list = [x + 5 for x in psink_list]
Gt = 96
tins_max = max(tsep_list) + 1

base_path = "/lustre/orion/lgt132/scratch/sicheng/GPD_calc"
conn_template = f"{base_path}/pdf_data/3pt_conn_inputs_man_cfg{{cfg}}.h5"

output_dir = "./gvar_data"
os.makedirs(output_dir, exist_ok=True)

n_cfg = len(cfg_list)
n_pt2_tsep = len(pt2_tsep_list)
n_tsep = len(tsep_list)
n_psink = len(psink_list)
n_tgf = len(tgf_list)
n_w = len(w_list)

# Shapes:
#   pt2_fit_plot[cfg, psink, pt2_tsep_index]
#   pt2_ratio_plot[cfg, psink, ratio_tsep_index]
#   pt2FF_O3_plot[cfg, tsep_index, tins<=max_tsep, psink, tgf_index, w_index]
#   FF_O3_plot[cfg, tins<=max_tsep, tgf_index, w_index]
#   no need to roll, origin is already tsep=0
pt2_fit_plot = np.zeros((n_cfg, n_psink, n_pt2_tsep), dtype="<c16")
pt2_ratio_plot = np.zeros((n_cfg, n_psink, n_tsep), dtype="<c16")
pt2FF_O3_plot = np.zeros((n_cfg, n_tsep, tins_max, n_psink, n_tgf, n_w), dtype="<c16")
FF_O3_plot = np.zeros((n_cfg, tins_max, n_tgf, n_w), dtype="<c16")


for i_cfg, cfg in enumerate(cfg_list):
    print(f"reading <2pt*O> , <2pt> and <O> cfg={cfg}")
    with h5py.File(conn_template.format(cfg=cfg), "r") as f:
        # All three are rolled by -tsrc and averaged over tsrc.
        pt2FF = f["pt2FF"]
        FF = f["FF"]
        pt2 = f["pt2"]

        pt2_fit_plot[i_cfg] = pt2[psink_list, :][:, pt2_tsep_list]
        pt2_ratio_plot[i_cfg] = pt2[psink_list, :][:, tsep_list]
        for itsep, tsep in enumerate(tsep_list):
            # CO slice shape: [tins, psink, mu_diag, tgf, z_WL]
            pt2FF_slice = pt2FF[tsep, :tins_max, psink_list, :, :, :][..., tgf_list, :]

            # Build O3 before jackknife.
            # Shape after selection: [tins, psink, mu_diag, tgf_index, w_index]
            pt2FF_slice = pt2FF_slice[..., w_list]
            pt2FF_O3_plot[i_cfg, itsep] = (
                -2 * pt2FF_slice[:, :, 0, :, :]
                + pt2FF_slice[:, :, 3, :, :]
                + pt2FF_slice[:, :, 4, :, :]
            )

        # FF shape: [tins, mu_diag, tgf, z_WL].
        FF_slice = FF[:tins_max, :, tgf_list, :][..., w_list]
        FF_O3_plot[i_cfg] = (
            -2 * FF_slice[:, 0, :, :] + FF_slice[:, 3, :, :] + FF_slice[:, 4, :, :]
        )


print("building jackknife lists")
jk_pt2_fit = jackknife(pt2_fit_plot)
jk_pt2_ratio = jackknife(pt2_ratio_plot)
jk_3pt_O3 = jackknife(pt2FF_O3_plot)
jk_FF_O3 = jackknife(FF_O3_plot)

pt2_by_psink_gv = {}
for ipsink, psink in enumerate(psink_list):
    key = f"p{psink - 5}"
    pt2_by_psink_gv[key] = {
        "real": jk_ls_avg(jk_pt2_fit[:, ipsink, :].real),
        "imag": jk_ls_avg(jk_pt2_fit[:, ipsink, :].imag),
    }

pt2_gv = {}
pt3_O3_gv = {}
ratio_gv = {}
x_centered_dic = {}

for i_tgf, tgf in enumerate(tgf_list):
    for ipsink, psink in enumerate(psink_list):
        for iw, w in enumerate(w_list):
            print(f"processing tgf={tgf}, psink={psink}, w={w}")

            for itsep, tsep in enumerate(tsep_list):
                key = f"tsep{tsep}_p{psink - 5}_flow{tgf}_w{w}"
                x_centered = np.arange(tsep + 1) - 0.5 * tsep

                jk_2pt = jk_pt2_ratio[:, ipsink, itsep]
                jk_2ptFF = jk_3pt_O3[:, itsep, : tsep + 1, ipsink, i_tgf, iw]
                jk_FF = jk_FF_O3[:, : tsep + 1, i_tgf, iw]

                jk_numerator = jk_2ptFF.real - jk_2pt.real[:, None] * jk_FF.real
                jk_ratio = jk_numerator / jk_2pt.real[:, None]
                ratio_avg = jk_ls_avg(jk_ratio)

                pt2_gv[key] = {
                    "real": jk_ls_avg(jk_2pt.real),
                    "imag": jk_ls_avg(jk_2pt.imag),
                }
                pt3_O3_gv[key] = {
                    "real": jk_ls_avg(jk_2ptFF.real),
                    "imag": jk_ls_avg(jk_2ptFF.imag),
                }
                ratio_gv[key] = ratio_avg
                x_centered_dic[key] = x_centered

                print(
                    f"  tsep={tsep}: mean real={np.mean(gv.mean(ratio_avg))}, "
                    f"mean err={np.mean(gv.sdev(ratio_avg))}"
                )


output_path = os.path.join(output_dir, "pdf_O3_conn_gvar_jk_manual.p")
gv.dump(
    {
        "pt2": pt2_gv,
        "pt2_by_psink": pt2_by_psink_gv,
        "pt3_O3": pt3_O3_gv,
        "ratio": ratio_gv,
        "x_centered": x_centered_dic,
        "Lt": Gt,
        "Gt": Gt,
        "cfg_list": cfg_list,
        "tsrc_list": tsrc_list,
        "pt2_tsep_list": pt2_tsep_list,
        "pt2_tmin": 2,
        "pt2_tmax": 10,
        "tsep_list": tsep_list,
        "psink_list": psink_list,
        "psink_phys_list": [psink - 5 for psink in psink_list],
        "tgf_list": tgf_list,
        "w_list": w_list,
    },
    output_path,
)

print(f"saved gvar data to {output_path}")
print(f"  pt2_by_psink entries = {len(pt2_by_psink_gv)}")
print(f"  pt2 entries    = {len(pt2_gv)}")
print(f"  pt3_O3 entries = {len(pt3_O3_gv)}")
print(f"  ratio entries  = {len(ratio_gv)}")
