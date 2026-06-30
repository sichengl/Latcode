from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "forward_itd_ratios_opp"
PLOT_DIR = SCRIPT_DIR / "forward_itd_ratio_plots_opp"

COMBO_NAME = "TXTX_plus_TYTY_minus_2XYXY"
CFG_LIST = list(range(204, 204 + 400 * 6, 6))
Q_TUPLE = (0, 0, 0)
FLOW_LIST = [20, 25, 30, 35, 40]
W_REF = 0

SKIP_W_REF_IN_PLOTS = True
CONNECT_SAME_W_POINTS = True

PLOT_DIR.mkdir(parents=True, exist_ok=True)

q_label = f"q{Q_TUPLE[0]}_{Q_TUPLE[1]}_{Q_TUPLE[2]}"
markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
colors = plt.cm.tab10(np.linspace(0, 1, 10))

for flow in FLOW_LIST:
    input_name = (
        f"{COMBO_NAME}_forward_itd_ratios_"
        f"{q_label}_flow{flow}_zref{W_REF}_cfgs{len(CFG_LIST)}.h5"
    )
    input_path = INPUT_DIR / input_name

    if not input_path.exists():
        print(f"missing {input_path}", flush=True)
        continue

    with h5py.File(input_path, "r") as f:
        nu = f["nu"][:]
        pz_list = f["pz_list"][:].astype(np.int64)
        w_list = f["w_list"][:].astype(np.int64)

        single_real_mean = f["ratios/single_real_mean"][:]
        single_real_sdev = f["ratios/single_real_sdev"][:]
        single_imag_mean = f["ratios/single_imag_mean"][:]
        single_imag_sdev = f["ratios/single_imag_sdev"][:]

        double_real_mean = f["ratios/double_real_mean"][:]
        double_real_sdev = f["ratios/double_real_sdev"][:]
        double_imag_mean = f["ratios/double_imag_mean"][:]
        double_imag_sdev = f["ratios/double_imag_sdev"][:]

    for ratio_name in ["single", "double"]:
        for part in ["real", "imag"]:
            if ratio_name == "single" and part == "real":
                y_mean = single_real_mean
                y_sdev = single_real_sdev
                ylabel = r"Re $M(p,z)/M(0,z)$"
                title = "single ratio real"
            elif ratio_name == "single" and part == "imag":
                y_mean = single_imag_mean
                y_sdev = single_imag_sdev
                ylabel = r"Im $M(p,z)/M(0,z)$"
                title = "single ratio imag"
            elif ratio_name == "double" and part == "real":
                y_mean = double_real_mean
                y_sdev = double_real_sdev
                ylabel = r"Re reduced ITD"
                title = "double ratio real"
            else:
                y_mean = double_imag_mean
                y_sdev = double_imag_sdev
                ylabel = r"Im reduced ITD"
                title = "double ratio imag"

            fig, ax = plt.subplots(figsize=(8.0, 5.8))

            for iw, w in enumerate(w_list):
                if SKIP_W_REF_IN_PLOTS and int(w) == W_REF:
                    continue

                x = nu[:, iw]
                y = y_mean[:, iw]
                yerr = y_sdev[:, iw]
                order = np.argsort(x)
                linestyle = "-" if CONNECT_SAME_W_POINTS else "none"

                ax.errorbar(
                    x[order],
                    y[order],
                    yerr=yerr[order],
                    marker=markers[iw % len(markers)],
                    color=colors[iw % len(colors)],
                    linestyle=linestyle,
                    linewidth=1.0,
                    markersize=5,
                    capsize=3,
                    label=rf"$w={int(w)}$",
                )

            if ratio_name == "single" and part == "real":
                ax.axhline(1.0, color="0.45", linestyle="--", linewidth=0.9)
            if ratio_name == "double" and part == "real":
                ax.axhline(1.0, color="0.45", linestyle="--", linewidth=0.9)
            if part == "imag":
                ax.axhline(0.0, color="0.45", linestyle="--", linewidth=0.9)

            ax.set_xlabel(r"$\nu=s_\nu(2\pi/L_s)\,p_z\,w$")
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 2)
            ax.set_title(
                f"{COMBO_NAME};OPP WL; {title}; q={Q_TUPLE}; flow={flow}; zref={W_REF}"
            )
            ax.tick_params(direction="in")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False, fontsize=8, ncol=2)
            fig.tight_layout()

            plot_name = (
                f"{COMBO_NAME}_opp_{ratio_name}_{part}_"
                f"{q_label}_flow{flow}_zref{W_REF}_cfgs{len(CFG_LIST)}.png"
            )
            plot_path = PLOT_DIR / plot_name
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"saved {plot_path}", flush=True)

