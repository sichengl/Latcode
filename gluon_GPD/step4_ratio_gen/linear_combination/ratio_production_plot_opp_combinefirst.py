import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gvar as gv
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_DIR = SCRIPT_DIR.parent
BASE_DIR = PYTHON_DIR.parent

sys.path.append("/ccs/home/sicheng/Correlation_Analysis")
sys.path.append(str(PYTHON_DIR / "Correlation_Analysis-main"))
from module.resampling import *


tsep_list = [4, 5, 6, 7, 8,9,10]
w_list = [0, 1, 2, 3,4,5,6,7,8,9]
tgf_list = [20, 25, 30, 35, 40]
cfg_list = list(range(204, 204+400*6, 6))
pf_list = [(0, 0, pz) for pz in [0, 1, 2, 3, 4, 5, 6]]
#00=XYXY, 33=TXTX, 44=TYTY, 30=XTXY, 03=XYXT
lorentz_list = [(0, 0), (3, 3), (4, 4), (3, 0), (0, 3)]
Gt = 96
Ls = 32
tins_max = max(tsep_list) + 1
conn_template = str(BASE_DIR / "3pt_production" / "gpd_data_opp" / "3pt_opp_conn_inputs_man_cfg{cfg}.h5")

q_list = np.array([[0, 0, 0]], dtype=np.int64)
operator_plot_list = [
    (
        "TXTXpTYTY",
        [(lorentz_list.index((3, 3)), 1.0), (lorentz_list.index((4, 4)), 1.0)],
        "TXTX + TYTY",
    ),
    (
        "TXTXpTYTYm2XYXY",
        [
            (lorentz_list.index((3, 3)), 1.0),
            (lorentz_list.index((4, 4)), 1.0),
            (lorentz_list.index((0, 0)), -2.0),
        ],
        "TXTX + TYTY - 2 * XYXY",
    ),
    ]

ratio_plot_dir = SCRIPT_DIR / "ratio_plots_manual_real2pt_opp"
ratio_jk_dir = SCRIPT_DIR / "ratio_jk_manual_real2pt_opp"

n_cfg = len(cfg_list)
n_tsep = len(tsep_list)
n_lorentz = len(lorentz_list)

def process_one(task):
    iq, q_tuple, i_tgf, tgf, ipf, pf_tuple, iw, w = task

    q_label = f"q{q_tuple[0]}_{q_tuple[1]}_{q_tuple[2]}"
    pf_label = f"pf{pf_tuple[0]}_{pf_tuple[1]}_{pf_tuple[2]}"

    print(f"processing q={q_tuple}, tgf={tgf}, pf={pf_tuple}, w={w}", flush=True)

    pt2_f_read = np.zeros((n_cfg, tins_max), dtype="<c16")
    pt2_i_read = np.zeros((n_cfg, tins_max), dtype="<c16")
    pt2FF_read = np.zeros((n_cfg, n_tsep, tins_max, n_lorentz), dtype="<c16")
    FF_read = np.zeros((n_cfg, tins_max, n_lorentz), dtype="<c16")

    for i_cfg, cfg in enumerate(cfg_list):
        print(f"reading cfg={cfg}")
        with h5py.File(conn_template.format(cfg=cfg), "r") as f:
            pt2FF = f["pt2FF"]
            FF = f["FF"]
            pt2_f = f["pt2_f"]
            pt2_i = f["pt2_i"]

            pt2_f_read[i_cfg] = pt2_f[ipf, :tins_max]
            pt2_i_read[i_cfg] = pt2_i[ipf, iq, :tins_max]

            #tins,lorentz,tgf,z_WL,q
            FF_read[i_cfg] = FF[:tins_max, :, i_tgf, iw, iq]
            pt2FF_read[i_cfg] = pt2FF[tsep_list, :tins_max, ipf, :, i_tgf, iw, iq]

    jk_ls_pt2_f = jackknife(pt2_f_read)
    jk_ls_pt2_i = jackknife(pt2_i_read)

    outs = []
    for op_name, components, op_expression in operator_plot_list:
        fig_real_out = (
            ratio_plot_dir / f"{op_name}_realratio_opp_{q_label}_{pf_label}_"
            f"w{w}_flow{tgf}_tsep{tsep_list}_cfgs{len(cfg_list)}.png"
        )
        fig_imag_out = (
            ratio_plot_dir / f"{op_name}_imagratio_opp_{q_label}_{pf_label}_"
            f"w{w}_flow{tgf}_tsep{tsep_list}_cfgs{len(cfg_list)}.png"
        )

        #ratio is a complex jackknife list
        ratio_out = (
            ratio_jk_dir / f"{op_name}_ratio_jk_opp_{q_label}_{pf_label}_"
            f"w{w}_flow{tgf}_tsep{tsep_list}_cfgs{len(cfg_list)}.h5"
        )

        op_3pt_read = np.zeros((n_cfg, n_tsep, tins_max), dtype="<c16")
        op_FF_read = np.zeros((n_cfg, tins_max), dtype="<c16")
        for op_idx, coeff in components:
            op_3pt_read += coeff * pt2FF_read[:, :, :, op_idx]
            op_FF_read += coeff * FF_read[:, :, op_idx]

        op_then_jk_3pt = jackknife(op_3pt_read)
        op_then_jk_FF = jackknife(op_FF_read)
        ratio_jk = np.full((n_cfg, n_tsep, tins_max), np.nan + 1j * np.nan, dtype="<c16")

        fig_real, ax_real = plt.subplots(1, 1, figsize=(9, 6))
        fig_imag, ax_imag = plt.subplots(1, 1, figsize=(9, 6))
        for itsep, tsep in enumerate(tsep_list):
            x_centered = np.arange(tsep + 1) - 0.5 * tsep
            tau = np.arange(tsep + 1)
            t_minus_tau = tsep - tau

            symmetric_phase = np.exp(1j * np.pi * q_tuple[2] * w / Ls)
            jk_2pt_f_T_c = jk_ls_pt2_f[:, tsep]
            jk_2pt_f_T = jk_ls_pt2_f[:, tsep].real
            jk_2pt_i_T = jk_ls_pt2_i[:, tsep].real
            jk_2pt_f_tau = jk_ls_pt2_f[:, tau].real
            jk_2pt_i_tau = jk_ls_pt2_i[:, tau].real
            jk_2pt_f_Tmtau = jk_ls_pt2_f[:, t_minus_tau].real
            jk_2pt_i_Tmtau = jk_ls_pt2_i[:, t_minus_tau].real
            jk_2ptFF = op_then_jk_3pt[:, itsep, :tsep + 1]
            jk_FF = op_then_jk_FF[:, :tsep + 1]
            jk_numerator = jk_2ptFF - jk_2pt_f_T_c[:, None] * jk_FF
            jk_numerator = jk_numerator * symmetric_phase
            jk_raw_ratio = jk_numerator / jk_2pt_f_T[:, None]
            factor = np.sqrt(
                (jk_2pt_f_T[:, None] * jk_2pt_i_Tmtau * jk_2pt_f_tau)
                / (jk_2pt_i_T[:, None] * jk_2pt_f_Tmtau * jk_2pt_i_tau)
            )
            jk_ratio_complex = jk_raw_ratio * factor
            ratio_jk[:, itsep, :tsep + 1] = jk_ratio_complex
            phase_mean = np.angle(np.nanmean(jk_ratio_complex, axis=0))
            print(f"mean phase is {phase_mean}")
            ratio_avg_real = jk_ls_avg(jk_ratio_complex.real)
            ratio_avg_imag = jk_ls_avg(jk_ratio_complex.imag)
            ax_real.errorbar(
                x_centered,
                gv.mean(ratio_avg_real),
                yerr=gv.sdev(ratio_avg_real),
                capsize=3,
                marker="o",
                linestyle="-",
                label=f"tsep={tsep}",
            )
            ax_imag.errorbar(
                x_centered,
                gv.mean(ratio_avg_imag),
                yerr=gv.sdev(ratio_avg_imag),
                capsize=3,
                marker="o",
                linestyle="-",
                label=f"tsep={tsep}",
            )

        max_tsep = max(tsep_list)
        ax_real.axvline(0, color="k", linestyle="--", linewidth=1, alpha=0.4)
        ax_real.set_xlim(-0.5 * max_tsep - 0.5, 0.5 * max_tsep + 0.5)
        ax_real.set_xlabel(r"Insertion Time ($t_{ins} - t_{sep}/2$)", fontsize=12)
        ax_real.set_ylabel(r"Re $R_{\rm sym}$", fontsize=12)
        ax_real.set_title(f"{op_name}: real part ratio", fontsize=13)
        ax_real.legend(fontsize=9, frameon=True)

        fig_real.suptitle(
            f"Off-forward ratio; q={q_tuple}; pf={pf_tuple}; "
            f"wilsonlen={w}; flow={tgf * 0.1:.2f}",
            fontsize=14,
        )

        fig_real.tight_layout()
        fig_real.savefig(fig_real_out, dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig_real)

        ax_imag.axvline(0, color="k", linestyle="--", linewidth=1, alpha=0.4)
        ax_imag.set_xlim(-0.5 * max_tsep - 0.5, 0.5 * max_tsep + 0.5)
        ax_imag.set_xlabel(r"Insertion Time ($t_{ins} - t_{sep}/2$)", fontsize=12)
        ax_imag.set_ylabel(r"Im $R_{\rm sym}$", fontsize=12)
        ax_imag.set_title(f"{op_name}: imag part ratio", fontsize=13)
        ax_imag.legend(fontsize=9, frameon=True)

        fig_imag.suptitle(
            f"Off-forward ratio; q={q_tuple}; pf={pf_tuple}; "
            f"wilsonlen={w}; flow={tgf * 0.1:.2f}",
            fontsize=14,
        )

        fig_imag.tight_layout()
        fig_imag.savefig(fig_imag_out, dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig_imag)

        with h5py.File(ratio_out, "w") as f:
            f.create_dataset("ratio_jk", data=ratio_jk)  #here the jackknife list saved is a complex list
            f.create_dataset("tsep_list", data=np.array(tsep_list, dtype=np.int64))
            f.create_dataset("q_code", data=np.array(q_tuple, dtype=np.int64))
            f.create_dataset("pf_code", data=np.array(pf_tuple, dtype=np.int64))
            f.create_dataset(
                "component_lorentz",
                data=np.array([lorentz_list[op_idx] for op_idx, coeff in components], dtype=np.int64),
            )
            f.create_dataset(
                "component_coefficients",
                data=np.array([coeff for op_idx, coeff in components], dtype=np.float64),
            )
            f.attrs["operator"] = op_name
            f.attrs["operator_expression"] = op_expression
            f.attrs["dim_ratio_jk"] = "jackknife,tsep_index,tau"
            f.attrs["q_convention"] = "FF q uses exp(+i q x); code-momentum pi = pf + q"
            f.attrs["vacuum_subtraction"] = "C3_sub_jk = C3_jk - C2f_jk(T) * FF_jk(tau)"
            f.attrs["ratio_definition"] = "symmetric off-forward ratio"
            f.attrs["combine_order"] = "combine C3 and FF by cfg first, then jackknife and form ratio"
        outs.append(str(fig_real_out))
        outs.append(str(fig_imag_out))
        outs.append(str(ratio_out))
    return outs


if __name__ == "__main__":
    ratio_plot_dir.mkdir(parents=True, exist_ok=True)
    ratio_jk_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for iq, qvec in enumerate(q_list):
        q_tuple = tuple(int(qi) for qi in qvec)
        for i_tgf, tgf in enumerate(tgf_list):
            for ipf, pf_tuple in enumerate(pf_list):
                for iw, w in enumerate(w_list):
                    tasks.append((iq, q_tuple, i_tgf, tgf, ipf, pf_tuple, iw, w))

    n_workers = int(os.environ.get("N_WORKERS", "30"))
    print(f"total tasks = {len(tasks)}, N_WORKERS = {n_workers}", flush=True)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_one, task) for task in tasks]

        for future in as_completed(futures):
            outs = future.result()
            for out in outs:
                print(f"saved {out}", flush=True)
