import os
import numpy as np
import gvar as gv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


#operator_list = ["O3", "TXTX", "TYTY", "XYXY"]
operator_list = ["XYXY"]
tgf_list = [20, 25, 30, 35, 40]
p_list = [0, 1, 2, 3, 4, 5]
w_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
z_plot_list = [1, 2, 3, 4, 5, 6]

Ls = 32

double_ratio_reference_w = 0

energy_by_p = {
    0: gv.gvar("0.1414(14)"),
    1: gv.gvar("0.2482(28)"),
    2: gv.gvar("0.4281(49)"),
    3: gv.gvar("0.6225(70)"),
    4: gv.gvar("0.796(14)"),
    5: gv.gvar("0.961(24)"),
}
m0 = energy_by_p[0]

base_dir = os.path.dirname(os.path.abspath(__file__))

input_dir = os.path.join(base_dir, "data_summation_jk_fit")
input_path = os.path.join(input_dir, "summation_jk_fit_dic.p")

ratio_output_dir = os.path.join(base_dir, "data_single_double_ratio_from_summation_jk")
os.makedirs(ratio_output_dir, exist_ok=True)
ratio_output_path = os.path.join(
    ratio_output_dir,
    f"single_double_ratio_zref{double_ratio_reference_w}_single_energy_corrected.p",
)

plot_dir = os.path.join(base_dir, "plots_single_double_ratio_from_summation_jk")
os.makedirs(plot_dir, exist_ok=True)

summation_fit_dic = gv.load(input_path)

ratio_dic = {}

iw_ref = w_list.index(double_ratio_reference_w)

for operator_name in operator_list:
    for tgf in tgf_list:
        key_p0 = f"{operator_name}_tgf{tgf}_psink0"
        M_0_jk = summation_fit_dic[key_p0]["M_jk"]

        for p in p_list:
            key_p = f"{operator_name}_tgf{tgf}_psink{p}"
            M_p_jk = summation_fit_dic[key_p]["M_jk"]

            single_jk = M_p_jk / M_0_jk

            M_p_ref_jk = M_p_jk[:, iw_ref:iw_ref + 1]
            M_0_ref_jk = M_0_jk[:, iw_ref:iw_ref + 1]
            double_jk = (M_p_jk / M_p_ref_jk) / (M_0_jk / M_0_ref_jk)

            n_jk = single_jk.shape[0]

            single_mean = np.mean(single_jk, axis=0)
            single_delta = single_jk - single_mean[None, :]
            single_cov = (n_jk - 1) / n_jk * (single_delta.T @ single_delta)
            single_gv = gv.gvar(single_mean, single_cov)

            # Energy correction for the single ratio:
            # single -> (m0 / E_p) * single
            single_gv = (m0 / energy_by_p[p]) * single_gv

            double_mean = np.mean(double_jk, axis=0)
            double_delta = double_jk - double_mean[None, :]
            double_cov = (n_jk - 1) / n_jk * (double_delta.T @ double_delta)
            double_gv = gv.gvar(double_mean, double_cov)

            result_key = f"{operator_name}_tgf{tgf}_psink{p}_zref{double_ratio_reference_w}"

            ratio_dic[result_key] = {
                "operator_name": operator_name,
                "tgf": tgf,
                "psink": p,
                "z_ref": double_ratio_reference_w,
                "w_list": list(w_list),
                "single_energy_factor": m0 / energy_by_p[p],
                "single_jk_raw": single_jk,
                "double_jk": double_jk,
                "single": single_gv,
                "double": double_gv,
                "single_cov_raw": single_cov,
                "double_cov": double_cov,
            }

gv.dump(ratio_dic, ratio_output_path)
print("saved ratio data to", ratio_output_path)


markers = ["o", "*", "D", "^", "s", "v", "P", "X", "<", ">"]
colors = ["crimson", "blue", "green", "magenta", "gray", "c", "orange", "purple", "brown", "black"]

for operator_name in operator_list:
    for ratio_name in ["single", "double"]:
        ncol = 2
        nrow = int(np.ceil(len(tgf_list) / ncol))

        fig, axes = plt.subplots(
            nrow,
            ncol,
            figsize=(11, 4 * nrow),
            sharex=True,
            squeeze=False,
        )
        axes = axes.ravel()

        for ax, tgf in zip(axes, tgf_list):
            for iz, z in enumerate(z_plot_list):
                x_values = []
                y_values = []
                y_errors = []

                for p in p_list:
                    key = f"{operator_name}_tgf{tgf}_psink{p}_zref{double_ratio_reference_w}"
                    ratio_gv = ratio_dic[key][ratio_name]

                    iw = w_list.index(z)
                    nu = 2.0 * np.pi * p * z / Ls

                    x_values.append(nu)
                    y_values.append(gv.mean(ratio_gv[iw]))
                    y_errors.append(gv.sdev(ratio_gv[iw]))

                ax.errorbar(
                    x_values,
                    y_values,
                    yerr=y_errors,
                    fmt=markers[iz % len(markers)],
                    color=colors[iz % len(colors)],
                    linestyle="none",
                    capsize=3,
                    markersize=4,
                    label=rf"$z={z}a$",
                )

            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
            ax.set_title(rf"$\tau={tgf / 10:.1f}$")
            ax.set_xlabel(r"$\nu = 2\pi pz/L_s$")
            ax.set_ylim(0, 2)
            if ratio_name == "single":
                ax.set_ylabel(r"$(m_0/E_p)M(p,z)/M(0,z)$")
            else:
                ax.set_ylabel(r"$\mathfrak{M}(\nu,z^2)$")

            ax.grid(linestyle=":")
            ax.legend(fontsize=8)

        for ax in axes[len(tgf_list):]:
            ax.axis("off")

        if ratio_name == "single":
            fig.suptitle(f"{operator_name} energy-corrected single ratio from summation-jk")
            plot_name = (
                f"{operator_name}_single_ratio_energy_corrected_ioffe_time_"
                f"zref{double_ratio_reference_w}.png"
            )
        else:
            fig.suptitle(
                f"{operator_name} double ratio from summation-jk, "
                f"z_ref={double_ratio_reference_w}"
            )
            plot_name = f"{operator_name}_double_ratio_ioffe_time_zref{double_ratio_reference_w}.png"

        fig.tight_layout()
        plot_path = os.path.join(plot_dir, plot_name)
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)

        print("saved plot to", plot_path)
