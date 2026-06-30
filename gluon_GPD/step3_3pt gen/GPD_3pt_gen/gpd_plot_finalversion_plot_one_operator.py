import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import gvar as gv
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("/ccs/home/sicheng/Correlation_Analysis")
sys.path.append(os.path.join(os.path.dirname(__file__), "Correlation_Analysis-main"))
from module.resampling import *


tsep_list = [4, 5, 6, 7, 8,9,10]
w_list = [0, 1, 2, 3]
tgf_list = [20, 25, 30, 35, 40]
cfg_list = list(range(204, 1404, 6))
psink_list = [x + 5 for x in [0, 1, 2, 3, 4, 5]]
lorentz_list = [(0, 0), (3, 3), (4, 4), (3, 0), (0, 3)]
Gt = 96
tins_max = max(tsep_list) + 1
conn_template = "./gpd_data/3pt_conn_inputs_man_cfg{cfg}.h5"

q_list = np.array(
    [[qx, qy, qz] for qx in [0, -1, 1] for qy in [0, -1, 1] for qz in [0, -1, 1]],
    dtype=np.int64,
)
q_plot_list = {
    (1, 0, 0), (0,1,0)
}
operator_plot_list = [
    ("TXTX", lorentz_list.index((3, 3))),
    ("TYTY", lorentz_list.index((4, 4))),
]

n_cfg = len(cfg_list)
n_tsep = len(tsep_list)
n_lorentz = len(lorentz_list)

def process_one(task):
    iq, q_tuple, i_tgf, tgf, psink, iw, w = task

    q_label = f"q{q_tuple[0]}_{q_tuple[1]}_{q_tuple[2]}"

    print(f"processing q={q_tuple}, tgf={tgf}, psink={psink-5}, w={w}", flush=True)

    pt2_read = np.zeros((n_cfg, n_tsep), dtype="<c16")
    pt2FF_read = np.zeros((n_cfg, n_tsep, tins_max, n_lorentz), dtype="<c16")
    FF_read = np.zeros((n_cfg, tins_max, n_lorentz), dtype="<c16")

    for i_cfg, cfg in enumerate(cfg_list):
        with h5py.File(conn_template.format(cfg=cfg), "r") as f:
            pt2FF = f["pt2FF"]
            FF = f["FF"]
            pt2 = f["pt2"]

            pt2_read[i_cfg] = pt2[psink, tsep_list]
            FF_read[i_cfg] = FF[:tins_max, :, i_tgf, iw, iq]
            pt2FF_read[i_cfg] = pt2FF[tsep_list, :tins_max, psink, :, i_tgf, iw, iq]

    jk_ls_pt2 = jackknife(pt2_read)

    outs = []
    for op_name, op_idx in operator_plot_list:
        out = (
            f"./plots_manual/{op_name}_methodB_{q_label}_p{psink-5}_"
            f"w{w}_flow{tgf}_tsep{tsep_list}_cfgs{len(cfg_list)}_centered.png"
        )

        # Method B: construct O3 first, then jackknife.
        op_3pt_read = pt2FF_read[:, :, :, op_idx]
        op_FF_read = FF_read[:, :, op_idx]

        op_then_jk_3pt = jackknife(op_3pt_read)
        op_then_jk_FF = jackknife(op_FF_read)

        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        for itsep, tsep in enumerate(tsep_list):
            x_centered = np.arange(tsep + 1) - 0.5 * tsep

            jk_2pt = jk_ls_pt2[:, itsep]
            jk_2ptFF = op_then_jk_3pt[:, itsep, :tsep + 1]
            jk_FF = op_then_jk_FF[:, :tsep + 1]

            jk_numerator = jk_2ptFF - jk_2pt[:, None] * jk_FF
            jk_ratio = (jk_numerator / jk_2pt[:, None]).real
            ratio_avg = jk_ls_avg(jk_ratio)

            ax.errorbar(
                x_centered,
                gv.mean(ratio_avg),
                yerr=gv.sdev(ratio_avg),
                capsize=3,
                marker="o",
                linestyle="-",
                label=f"tsep={tsep}",
            )

        max_tsep = max(tsep_list)
        ax.axvline(0, color="k", linestyle="--", linewidth=1, alpha=0.4)
        ax.set_xlim(-0.5 * max_tsep - 0.5, 0.5 * max_tsep + 0.5)
        ax.set_xlabel(r"Insertion Time ($t_{ins} - t_{sep}/2$)", fontsize=12)
        ax.set_ylabel(r"$(<C_2 O> - <C_2><O>) / <C_2>$", fontsize=12)
        ax.set_title(f"{op_name}: method B", fontsize=13)
        ax.legend(fontsize=9, frameon=True)

        fig.suptitle(
            f"Connected-subtracted ratio; q={q_tuple}; psink={psink-5}; "
            f"wilsonlen={w}; flow={tgf * 0.1:.2f}",
            fontsize=14,
        )

        fig.tight_layout()
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        outs.append(out)

    return outs


if __name__ == "__main__":
    os.makedirs("./plots_manual", exist_ok=True)

    tasks = []
    for iq, qvec in enumerate(q_list):
        q_tuple = tuple(int(qi) for qi in qvec)
        if q_tuple not in q_plot_list:
            continue 
        for i_tgf, tgf in enumerate(tgf_list):
            for psink in psink_list:
                for iw, w in enumerate(w_list):
                    tasks.append((iq, q_tuple, i_tgf, tgf, psink, iw, w))

    n_workers = int(os.environ.get("N_WORKERS", "4"))
    print(f"total tasks = {len(tasks)}, N_WORKERS = {n_workers}", flush=True)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_one, task) for task in tasks]

        for future in as_completed(futures):
            outs = future.result()
            for out in outs:
                print(f"saved {out}", flush=True)
