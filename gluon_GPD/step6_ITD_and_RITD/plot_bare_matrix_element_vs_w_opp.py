import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


INPUT_DIR = Path(
    r"/lustre/orion/lgt132/scratch/sicheng/GPD_calc_v3/ratio_production/bare_matrix_element_jk_opp_400cfgs"
)
PLOT_DIR = Path(
    r"/lustre/orion/lgt132/scratch/sicheng/GPD_calc_v3/ratio_production/bare_matrix_element_w_plots_opp"
)

CFG_NUM = 400
FIT_TAG = "two_state"

OPERATOR_LIST = ["TXTX", "TYTY"]
Q_LIST = [(0, 0, 0)]
PF_LIST = [(0, 0, pz) for pz in range(7)]
FLOW_LIST = [20, 25, 30, 35, 40]
W_LIST = list(range(10))


def jackknife_average_error(jk_samples):
    jk_samples = np.asarray(jk_samples, dtype=np.float64)
    n_jk = jk_samples.shape[0]

    average = np.mean(jk_samples, axis=0)
    diff = jk_samples - average[None, :]
    covariance = (n_jk - 1.0) / n_jk * diff.T @ diff
    covariance = 0.5 * (covariance + covariance.T)
    error = np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    return average, error


PLOT_DIR.mkdir(parents=True, exist_ok=True)

n_saved = 0
n_missing = 0
n_skipped = 0

for operator in OPERATOR_LIST:
    for q_tuple in Q_LIST:
        q_label = f"q{q_tuple[0]}_{q_tuple[1]}_{q_tuple[2]}"

        for pf_tuple in PF_LIST:
            pf_label = f"pf{pf_tuple[0]}_{pf_tuple[1]}_{pf_tuple[2]}"

            for flow in FLOW_LIST:
                file_name = (
                    f"{operator}_bareM_jk_opp_{q_label}_{pf_label}_"
                    f"flow{flow}_fit{FIT_TAG}_cfgs{CFG_NUM}.h5"
                )
                input_path = INPUT_DIR / file_name

                if not input_path.exists():
                    print(f"missing {input_path}", flush=True)
                    n_missing += 1
                    continue

                print(f"reading {input_path}", flush=True)
                with h5py.File(input_path, "r") as f:
                    bare_matrix_element_jk = f["bare_matrix_element_jk"][:]

                if bare_matrix_element_jk.ndim != 2:
                    print(
                        f"skip {input_path.name}: expected 2D data, "
                        f"got shape {bare_matrix_element_jk.shape}",
                        flush=True,
                    )
                    n_skipped += 1
                    continue

                if bare_matrix_element_jk.shape[1] != len(W_LIST):
                    print(
                        f"skip {input_path.name}: expected {len(W_LIST)} w values, "
                        f"got {bare_matrix_element_jk.shape[1]}",
                        flush=True,
                    )
                    n_skipped += 1
                    continue

                real_average, real_error = jackknife_average_error(
                    bare_matrix_element_jk.real
                )
                imag_average, imag_error = jackknife_average_error(
                    bare_matrix_element_jk.imag
                )

                fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True)

                axes[0].errorbar(
                    W_LIST,
                    real_average,
                    yerr=real_error,
                    marker="o",
                    linestyle="-",
                    capsize=3,
                )
                axes[0].axhline(0.0, color="0.3", linestyle=":", linewidth=1)
                axes[0].set_ylabel("Re M00")
                axes[0].grid(alpha=0.25)

                axes[1].errorbar(
                    W_LIST,
                    imag_average,
                    yerr=imag_error,
                    marker="o",
                    linestyle="-",
                    capsize=3,
                )
                axes[1].axhline(0.0, color="0.3", linestyle=":", linewidth=1)
                axes[1].set_xlabel("w")
                axes[1].set_ylabel("Im M00")
                axes[1].set_xticks(W_LIST)
                axes[1].grid(alpha=0.25)

                fig.suptitle(f"{operator}; q={q_tuple}; pf={pf_tuple}; flow={flow}")
                fig.tight_layout()

                plot_name = (
                    f"{operator}_bareM_vs_w_{q_label}_{pf_label}_flow{flow}.png"
                )
                plot_path = PLOT_DIR / plot_name
                fig.savefig(plot_path, dpi=300)
                plt.close(fig)

                print(f"saved {plot_path}", flush=True)
                n_saved += 1

print(
    f"done: saved {n_saved} plots, missing {n_missing} files, "
    f"skipped {n_skipped} files",
    flush=True,
)

