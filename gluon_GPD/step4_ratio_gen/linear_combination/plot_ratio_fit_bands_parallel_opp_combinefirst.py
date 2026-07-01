import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RATIO_DIR = SCRIPT_DIR / "ratio_jk_manual_real2pt_opp"
FIT_DIR = SCRIPT_DIR / "bare_matrix_element_jk_opp_400cfgs"
PLOT_DIR = SCRIPT_DIR / "ratio_fit_band_plots_opp"

CFG_LIST = list(range(204, 204 + 400 * 6, 6))
TSEP_LIST = [4, 5, 6, 7, 8,9,10]
Q_LIST = [(0, 0, 0)]
PF_LIST = [(0, 0, pz) for pz in range(7)]
W_LIST = [0, 1, 2, 3,4,5,6,7,8,9]
TGF_LIST = [20, 25, 30, 35, 40]
OPERATOR_LIST = ["TXTXpTYTY", "TXTXpTYTYm2XYXY"]
FIT_TAG = "two_state"
N_WORKERS = int(os.environ.get("N_WORKERS", "30"))


def jk_mean_err(jk_array):
    n_jk = jk_array.shape[0]
    avg = np.mean(jk_array, axis=0)
    diff = jk_array - avg[None, ...]
    err = np.sqrt((n_jk - 1.0) / n_jk * np.sum(diff**2, axis=0))
    return avg, err


def model_from_params(params, iw, tsep, tau_grid):
    M00 = params["M00"][:, iw, None]
    Ai = params["Ai"][:, iw, None]
    Af = params["Af"][:, iw, None]
    Afi = params["Afi"][:, iw, None]
    dEi = params["dEi"][:, None]
    dEf = params["dEf"][:, None]
    tau = tau_grid[None, :]
    t_minus_tau = tsep - tau
    return (
        M00
        + Ai * np.exp(-dEi * tau)
        + Af * np.exp(-dEf * t_minus_tau)
        + Afi * np.exp(-dEi * tau - dEf * t_minus_tau)
    )


def plot_one(operator, q_tuple, pf_tuple, flow, w):
    q_label = f"q{q_tuple[0]}_{q_tuple[1]}_{q_tuple[2]}"
    pf_label = f"pf{pf_tuple[0]}_{pf_tuple[1]}_{pf_tuple[2]}"
    ratio_path = RATIO_DIR / (
        f"{operator}_ratio_jk_opp_{q_label}_{pf_label}_"
        f"w{w}_flow{flow}_tsep{TSEP_LIST}_cfgs{len(CFG_LIST)}.h5"
    )
    fit_path = FIT_DIR / (
        f"{operator}_bareM_jk_opp_{q_label}_{pf_label}_"
        f"flow{flow}_fit{FIT_TAG}_cfgs{len(CFG_LIST)}.h5"
    )

    if not ratio_path.exists():
        print(f"missing {ratio_path}", flush=True)
        return
    if not fit_path.exists():
        print(f"missing {fit_path}", flush=True)
        return

    with h5py.File(ratio_path, "r") as f:
        ratio_jk = f["ratio_jk"][:]
        tsep_list = f["tsep_list"][:].astype(np.int64)
        operator_expression = f.attrs.get("operator_expression", operator)

    with h5py.File(fit_path, "r") as f:
        w_list = f["w_list"][:].astype(np.int64)
        if w not in w_list:
            print(f"w={w} not in {fit_path}", flush=True)
            return
        iw = int(np.where(w_list == w)[0][0])
        bare_jk = f["bare_matrix_element_jk"][:, iw]
        params = {}
        fit_quality = {}
        for part in ["real", "imag"]:
            params[part] = {
                name: f[f"fit_params_jk/{part}/{name}"][:]
                for name in ["M00", "Ai", "Af", "Afi", "dEi", "dEf"]
            }
            chi2 = f[f"fit_quality/{part}/chi2"][:]
            dof = f[f"fit_quality/{part}/dof"][:]
            Q = f[f"fit_quality/{part}/Q"][:]
            status = f[f"fit_quality/{part}/status"][:]
            good = (status == 0) & np.isfinite(chi2) & np.isfinite(Q) & (dof > 0)
            if np.any(good):
                fit_quality[part] = (
                    np.mean(chi2[good] / dof[good]),
                    np.mean(Q[good]),
                )
            else:
                fit_quality[part] = (np.nan, np.nan)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(tsep_list)))

    for part in ["real", "imag"]:
        fig, ax = plt.subplots(figsize=(9, 6))
        data = ratio_jk.real if part == "real" else ratio_jk.imag
        M_jk = bare_jk.real if part == "real" else bare_jk.imag
        M_avg, M_err = jk_mean_err(M_jk)
        ax.axhspan(M_avg - M_err, M_avg + M_err, color="0.7", alpha=0.25, label="M00 error")
        ax.axhline(M_avg, color="k", linestyle="--", linewidth=1.2, label="M00 mean")

        for itsep, tsep in enumerate(tsep_list):
            tau = np.arange(0, int(tsep) + 1)
            x = tau - 0.5 * int(tsep)
            data_jk = data[:, itsep, tau]
            data_avg, data_err = jk_mean_err(data_jk)

            fit_tau = np.linspace(0, int(tsep), 200)
            fit_x = fit_tau - 0.5 * int(tsep)
            fit_jk = model_from_params(params[part], iw, int(tsep), fit_tau)
            fit_avg, fit_err = jk_mean_err(fit_jk)

            color = colors[itsep]
            ax.errorbar(
                x,
                data_avg,
                yerr=data_err,
                color=color,
                marker="o",
                linestyle="none",
                capsize=3,
                label=f"data T={int(tsep)}",
            )
            ax.fill_between(fit_x, fit_avg - fit_err, fit_avg + fit_err, color=color, alpha=0.22)
            ax.plot(fit_x, fit_avg, color=color, linewidth=1.8, label=f"fit T={int(tsep)}")

        ax.axvline(0, color="0.2", linestyle=":", linewidth=1)
        ax.set_xlabel(r"$\tau - T/2$")
        ax.set_ylabel(f"{part} ratio")
        chi2_dof, Q_mean = fit_quality[part]
        ax.set_title(
            f"{operator_expression} {part}; q={q_tuple}; pf={pf_tuple}; w={w}; flow={flow}; "
            f"chi2/dof={chi2_dof:.3g}, Q={Q_mean:.3g}"
        )
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        out_path = PLOT_DIR / f"{operator}_{part}_{q_label}_{pf_label}_w{w}_flow{flow}.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"saved {out_path}", flush=True)


def main():
    tasks = []
    for operator in OPERATOR_LIST:
        for q_tuple in Q_LIST:
            for pf_tuple in PF_LIST:
                for flow in TGF_LIST:
                    for w in W_LIST:
                        tasks.append((operator, q_tuple, pf_tuple, flow, w))

    print(f"total plot tasks = {len(tasks)}, N_WORKERS = {N_WORKERS}", flush=True)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(plot_one, *task) for task in tasks]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
