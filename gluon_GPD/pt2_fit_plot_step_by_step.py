import os

import gvar as gv
import lsqfit as lsf
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def _validate_nstate(nstate):
    if isinstance(nstate, bool) or not isinstance(nstate, int) or nstate < 1:
        raise ValueError("nstate must be a positive integer")


def pt2_re_fcn(pt2_t, p, Lt, nstate=2):
    """Real part of the n-state two-point correlator.

    This is the only model formula used in the 2pt fit:

        C2(t) = sum_n z_n^2 / (2 E_n) * [exp(-E_n t) + exp(-E_n (Lt - t))]

    For nstate=2:
        E_0 = E0
        E_1 = E0 + dE1

    The fit prior uses the key "log(dE1)"; lsqfit/gvar then exposes the
    positive transformed parameter as p["dE1"] inside this function.
    """
    _validate_nstate(nstate)

    value = 0.0
    energy = p["E0"]
    for state in range(nstate):
        if state > 0:
            energy = energy + p[f"dE{state}"]

        z = p[f"z{state}"]
        value = value + z**2 / (2.0 * energy) * (
            np.exp(-energy * pt2_t) + np.exp(-energy * (Lt - pt2_t))
        )

    return value


# This is the all-in-one, step-by-step version of pt2_fit_plot.py.
# It keeps the 2pt model formula and the fitting flow in one file, so you can
# see what is being fitted without jumping between project files.

input_path = "./gvar_data/pdf_O3_conn_gvar_jk_manual.p"
output_dir = "./plots"
os.makedirs(output_dir, exist_ok=True)

summary_path = os.path.join(output_dir, "step_by_step_pt2_fit_summary.txt")
debug_path = os.path.join(output_dir, "step_by_step_pt2_fit_debug.txt")

boundary = "periodic"
nstate = 2
default_Lt = 96
pt2_real_sign = -1.0

# The right edge in the dictionary is Python-style exclusive. For example,
# (4, 14) means t = 4, 5, ..., 13.
fit_window_by_psink = {
    "p0": (3, 15),
    "p1": (4, 14),
    "p2": (4, 14),
    "p3": (5, 15),
    "p4": (3, 13),
    "p5": (3, 15),
}


print(f"loading {input_path}")
data = gv.load(input_path)

if "Lt" in data:
    Lt = int(data["Lt"])
elif "Gt" in data:
    Lt = int(data["Gt"])
else:
    Lt = default_Lt
    print(f"input data does not contain Lt/Gt; using Lt={Lt}")

if "pt2_tsep_list" not in data:
    raise KeyError("input data must contain pt2_tsep_list")
if "pt2_by_psink" not in data:
    raise KeyError("input data must contain pt2_by_psink")

t_all = np.asarray(data["pt2_tsep_list"], dtype=float)
psink_keys = sorted(data["pt2_by_psink"], key=lambda key: int(key[1:]))

if data.get("pt2_by_psink_sign_fixed", False):
    input_sign = 1.0
else:
    input_sign = pt2_real_sign

print(f"Lt = {Lt}")
print(f"pt2 input sign = {input_sign}")
print(f"momenta = {psink_keys}")

with open(summary_path, "w", encoding="utf-8") as summary_file, open(
    debug_path, "w", encoding="utf-8"
) as debug_file:
    summary_file.write("Step-by-step 2pt two-state fit summary\n")
    summary_file.write(f"input_path = {input_path}\n")
    summary_file.write(f"Lt = {Lt}\n")
    summary_file.write(f"pt2 input sign = {input_sign}\n\n")

    debug_file.write("Step-by-step 2pt two-state fit debug\n")
    debug_file.write(f"input_path = {input_path}\n")
    debug_file.write(f"Lt = {Lt}\n")
    debug_file.write(f"pt2 input sign = {input_sign}\n\n")

    for psink_key in psink_keys:
        print("")
        print(f"=== {psink_key} ===")

        pt2_for_this_momentum = data["pt2_by_psink"][psink_key]
        c2_real_all = input_sign * np.asarray(pt2_for_this_momentum["real"], dtype=object)
        c2_imag_all = np.asarray(pt2_for_this_momentum["imag"], dtype=object)

        if psink_key not in fit_window_by_psink:
            raise KeyError(f"missing fit window for {psink_key}")

        tmin, tmax_exclusive = fit_window_by_psink[psink_key]
        fit_mask = (t_all >= tmin) & (t_all < tmax_exclusive)
        if not np.any(fit_mask):
            raise ValueError(f"empty fit window for {psink_key}")

        t_fit = t_all[fit_mask]
        c2_real_fit = c2_real_all[fit_mask]
        c2_imag_fit = c2_imag_all[fit_mask]
        fit_t_label = f"{int(t_fit[0])}-{int(t_fit[-1])}"

        # Estimate a loose prior directly from the data in the chosen fit window.
        # This is the same logic as the function-based script, but kept inline.
        c2_fit_mean = np.asarray(gv.mean(c2_real_fit), dtype=float)
        positive_mask = c2_fit_mean > 0
        if np.count_nonzero(positive_mask) >= 2:
            t_positive = t_fit[positive_mask]
            c2_positive = c2_fit_mean[positive_mask]
            e0_guess = float(
                np.median(np.log(c2_positive[:-1] / c2_positive[1:]) / np.diff(t_positive))
            )
        else:
            e0_guess = 0.5

        if not np.isfinite(e0_guess) or e0_guess <= 0:
            e0_guess = 0.5
        e0_guess = float(np.clip(e0_guess, 0.05, 2.0))

        amplitude_guess = float(np.nanmax(np.abs(c2_fit_mean)))
        if not np.isfinite(amplitude_guess) or amplitude_guess <= 0:
            amplitude_guess = 1.0

        first_fit_t = float(np.min(t_fit))
        z0_guess = np.sqrt(
            max(2.0 * e0_guess * amplitude_guess * np.exp(e0_guess * first_fit_t), 1e-30)
        )
        z_width = max(10.0 * abs(z0_guess), 10.0)

        prior = gv.BufferDict()
        prior["E0"] = gv.gvar(e0_guess, max(0.5 * e0_guess, 0.5))
        prior["log(dE1)"] = gv.gvar(np.log(0.5), 1.0)
        prior["z0"] = gv.gvar(z0_guess, z_width)
        prior["z1"] = gv.gvar(0.0, z_width)

        print(f"fit t = {fit_t_label}")
        print("prior:")
        print(f"  E0 = {float(gv.mean(prior['E0'])):.6g} +/- {float(gv.sdev(prior['E0'])):.6g}")
        print(
            "  log(dE1) = "
            f"{float(gv.mean(prior['log(dE1)'])):.6g} +/- "
            f"{float(gv.sdev(prior['log(dE1)'])):.6g}"
        )
        print(f"  z0 = {float(gv.mean(prior['z0'])):.6g} +/- {float(gv.sdev(prior['z0'])):.6g}")
        print(f"  z1 = {float(gv.mean(prior['z1'])):.6g} +/- {float(gv.sdev(prior['z1'])):.6g}")

        fit = lsf.nonlinear_fit(
            data=(t_fit, c2_real_fit),
            prior=prior,
            fcn=lambda t, p: pt2_re_fcn(t, p, Lt, nstate=nstate),
            maxit=10000,
        )

        fit_curve_at_data = pt2_re_fcn(t_fit, fit.p, Lt, nstate=nstate)
        data_pull = (
            np.asarray(gv.mean(c2_real_fit), dtype=float)
            - np.asarray(gv.mean(fit_curve_at_data), dtype=float)
        ) / np.asarray(gv.sdev(c2_real_fit), dtype=float)

        c2_rel_err = np.asarray(gv.sdev(c2_real_fit), dtype=float) / np.abs(
            np.asarray(gv.mean(c2_real_fit), dtype=float)
        )

        print(f"Q = {fit.Q:.6g}")
        print(f"chi2/dof = {fit.chi2 / fit.dof:.6g}")
        print(f"E0 = {fit.p['E0']}")
        print(f"dE1 = {fit.p['dE1']}")
        print(f"max |data pull| = {np.nanmax(np.abs(data_pull)):.3g}")
        print(f"C2 rel err range = {np.nanmin(c2_rel_err):.3g} .. {np.nanmax(c2_rel_err):.3g}")

        summary_file.write(f"[{psink_key}]\n")
        summary_file.write(f"fit t = {fit_t_label}\n")
        summary_file.write(f"Q = {fit.Q:.8g}\n")
        summary_file.write(f"chi2/dof = {fit.chi2 / fit.dof:.8g}\n")
        summary_file.write(f"E0 = {fit.p['E0']}\n")
        summary_file.write(f"dE1 = {fit.p['dE1']}\n")
        summary_file.write(f"max |data pull| = {np.nanmax(np.abs(data_pull)):.8g}\n\n")

        debug_file.write(f"[{psink_key}]\n")
        debug_file.write(f"fit t = {fit_t_label}\n")
        debug_file.write(f"prior E0 = {float(gv.mean(prior['E0'])):.16e} +/- {float(gv.sdev(prior['E0'])):.16e}\n")
        debug_file.write(
            "prior log(dE1) = "
            f"{float(gv.mean(prior['log(dE1)'])):.16e} +/- "
            f"{float(gv.sdev(prior['log(dE1)'])):.16e}\n"
        )
        debug_file.write(f"prior z0 = {float(gv.mean(prior['z0'])):.16e} +/- {float(gv.sdev(prior['z0'])):.16e}\n")
        debug_file.write(f"prior z1 = {float(gv.mean(prior['z1'])):.16e} +/- {float(gv.sdev(prior['z1'])):.16e}\n")
        debug_file.write(f"Q = {fit.Q:.16e}\n")
        debug_file.write(f"chi2/dof = {fit.chi2 / fit.dof:.16e}\n")
        debug_file.write(f"E0 = {fit.p['E0']}\n")
        debug_file.write(f"dE1 = {fit.p['dE1']}\n")
        debug_file.write("t data_mean data_sdev fit_mean pull\n")
        for t_value, y_value, dy_value, fit_value, pull_value in zip(
            t_fit,
            gv.mean(c2_real_fit),
            gv.sdev(c2_real_fit),
            gv.mean(fit_curve_at_data),
            data_pull,
        ):
            debug_file.write(
                f"{int(t_value)} {y_value:.16e} {dy_value:.16e} "
                f"{fit_value:.16e} {pull_value:.8g}\n"
            )
        debug_file.write("\n")

        # Plot the original 2pt data together with the fitted two-state curve.
        curve_t = np.linspace(float(np.min(t_all)), float(np.max(t_all)), 300)
        curve_c2 = pt2_re_fcn(curve_t, fit.p, Lt, nstate=nstate)
        curve_mean = np.asarray(gv.mean(curve_c2), dtype=float)
        curve_sdev = np.asarray(gv.sdev(curve_c2), dtype=float)

        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.add_patch(
            Rectangle(
                (float(t_fit[0]), 0.0),
                float(t_fit[-1] - t_fit[0]),
                1.0,
                transform=ax.get_xaxis_transform(),
                facecolor="grey",
                alpha=0.28,
                edgecolor="none",
                zorder=0,
                label="fit region",
            )
        )
        ax.errorbar(
            t_all,
            gv.mean(c2_real_all),
            yerr=gv.sdev(c2_real_all),
            fmt="o",
            mfc="white",
            capsize=4,
            label="data",
        )
        ax.plot(t_fit, gv.mean(c2_real_fit), "o", color="tab:blue", label="fit points")
        ax.plot(curve_t, curve_mean, color="tab:orange", label="two-state fit")
        ax.fill_between(
            curve_t,
            curve_mean - curve_sdev,
            curve_mean + curve_sdev,
            color="tab:orange",
            alpha=0.35,
            label="fit band",
        )
        ax.set_xlabel("t_sep / a")
        ax.set_ylabel("-C2 real")
        ax.set_title(
            f"{psink_key}, fit t={fit_t_label}, "
            f"chi2/dof={fit.chi2 / fit.dof:.3g}, Q={fit.Q:.3g}"
        )
        ax.grid(linestyle=":")
        ax.legend(fontsize=9)
        fig.savefig(
            os.path.join(output_dir, f"step_by_step_pt2_fit_{psink_key}_c2pt.png"),
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)

        # Plot the effective mass using the same fit result.
        if boundary == "periodic" and len(t_all) >= 3:
            data_meff = np.arccosh((c2_real_all[2:] + c2_real_all[:-2]) / (2.0 * c2_real_all[1:-1]))
            data_meff_t = t_all[1:-1]

            fit_t_grid = np.arange(int(np.min(t_all)), int(np.max(t_all)) + 1, dtype=float)
            fit_c2_grid = pt2_re_fcn(fit_t_grid, fit.p, Lt, nstate=nstate)
            fit_meff = np.arccosh((fit_c2_grid[2:] + fit_c2_grid[:-2]) / (2.0 * fit_c2_grid[1:-1]))
            fit_meff_t = fit_t_grid[1:-1]

            meff_fit_mask = (data_meff_t >= tmin) & (data_meff_t < tmax_exclusive)

            fig, ax = plt.subplots(figsize=(7.0, 4.5))
            if np.any(meff_fit_mask):
                ax.add_patch(
                    Rectangle(
                        (float(data_meff_t[meff_fit_mask][0]), 0.0),
                        float(data_meff_t[meff_fit_mask][-1] - data_meff_t[meff_fit_mask][0]),
                        1.0,
                        transform=ax.get_xaxis_transform(),
                        facecolor="grey",
                        alpha=0.28,
                        edgecolor="none",
                        zorder=0,
                        label="fit region",
                    )
                )
            ax.errorbar(
                data_meff_t,
                gv.mean(data_meff),
                yerr=gv.sdev(data_meff),
                fmt="o",
                mfc="white",
                capsize=4,
                label="data",
            )
            ax.plot(fit_meff_t, gv.mean(fit_meff), color="tab:orange", label="two-state fit")
            ax.fill_between(
                fit_meff_t,
                gv.mean(fit_meff) - gv.sdev(fit_meff),
                gv.mean(fit_meff) + gv.sdev(fit_meff),
                color="tab:orange",
                alpha=0.35,
                label="fit band",
            )
            ax.set_xlabel("t_sep / a")
            ax.set_ylabel("m_eff")
            ax.set_title(
                f"{psink_key}, fit t={fit_t_label}, "
                f"chi2/dof={fit.chi2 / fit.dof:.3g}, Q={fit.Q:.3g}"
            )
            ax.grid(linestyle=":")
            ax.legend(fontsize=9)
            fig.savefig(
                os.path.join(output_dir, f"step_by_step_pt2_fit_{psink_key}_meff.png"),
                dpi=220,
                bbox_inches="tight",
            )
            plt.close(fig)

print("")
print(f"saved summary to {summary_path}")
print(f"saved debug to {debug_path}")
print(f"saved step-by-step plots to {output_dir}")
