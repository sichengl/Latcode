import os
import sys

import gvar as gv
import lsqfit as lsf
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "LaMETLat-main", "src"))

from lametlat.ground_state.fit_funcs import pt2_re_fcn


input_path = "./gvar_data/pdf_O3_conn_gvar_jk_manual.p"
output_dir = "./plots"
output_path = os.path.join(output_dir, "dispersion_relation_check.png")
summary_path = os.path.join(output_dir, "dispersion_relation_check.txt")

Ls = 32
Lt = 96
nstate = 2
pt2_real_sign = -1.0
fit_window_by_psink = {
    "p0": (3, 15),
    "p1": (4, 14),
    "p2": (4, 14),
    "p3": (5, 15),
    "p4": (3, 13),
    "p5": (3, 15),
}


def psink_to_n(psink_key):
    return int(psink_key[1:])


def pt2_input_sign(data):
    return 1.0 if data.get("pt2_by_psink_sign_fixed", False) else pt2_real_sign


def estimate_prior(t_data, y_data):
    mean = np.asarray(gv.mean(y_data), dtype=float)
    t_data = np.asarray(t_data, dtype=float)
    positive = mean > 0

    if np.count_nonzero(positive) >= 2:
        t_pos = t_data[positive]
        y_pos = mean[positive]
        e0_guess = np.median(np.log(y_pos[:-1] / y_pos[1:]) / np.diff(t_pos))
    else:
        e0_guess = 0.5
    if not np.isfinite(e0_guess) or e0_guess <= 0:
        e0_guess = 0.5
    e0_guess = float(np.clip(e0_guess, 0.05, 2.0))

    amp = float(np.nanmax(np.abs(mean)))
    z_guess = np.sqrt(max(2.0 * e0_guess * amp * np.exp(e0_guess * np.min(t_data)), 1e-30))
    z_width = max(10.0 * z_guess, 10.0)

    prior = gv.BufferDict()
    prior["E0"] = gv.gvar(e0_guess, max(0.5 * e0_guess, 0.5))
    prior["log(dE1)"] = gv.gvar(np.log(0.5), 1.0)
    prior["z0"] = gv.gvar(z_guess, z_width)
    prior["z1"] = gv.gvar(0.0, z_width)
    return prior


def fit_pt2(t_fit, y_fit, Lt_value):
    prior = estimate_prior(t_fit, y_fit)

    def fcn(t, p):
        return pt2_re_fcn(t, p, Lt_value, nstate=nstate)

    return lsf.nonlinear_fit(data=(t_fit, y_fit), prior=prior, fcn=fcn, maxit=10000)


def fit_dispersion(p2, e2):
    prior = gv.BufferDict()
    prior["m2"] = gv.gvar(gv.mean(e2[0]), max(10.0 * gv.sdev(e2[0]), 1.0))
    prior["c2"] = gv.gvar(1.0, 2.0)

    def fcn(x, p):
        return p["m2"] + p["c2"] * x

    return lsf.nonlinear_fit(data=(p2, e2), prior=prior, fcn=fcn, maxit=10000)


def main():
    os.makedirs(output_dir, exist_ok=True)
    data = gv.load(input_path)
    Lt_value = int(data.get("Lt", data.get("Gt", Lt)))
    sign = pt2_input_sign(data)
    t = np.asarray(data["pt2_tsep_list"], dtype=float)

    fit_rows = []
    for psink_key in sorted(data["pt2_by_psink"], key=psink_to_n):
        n = psink_to_n(psink_key)
        tmin, tmax = fit_window_by_psink[psink_key]
        y = sign * np.asarray(data["pt2_by_psink"][psink_key]["real"], dtype=object)
        mask = (t >= tmin) & (t < tmax)
        fit = fit_pt2(t[mask], y[mask], Lt_value)
        fit_rows.append(
            {
                "psink_key": psink_key,
                "n": n,
                "p": 2.0 * np.pi * n / Ls,
                "fit": fit,
                "E0": fit.p["E0"],
                "Q": fit.Q,
                "chi2_dof": fit.chi2 / fit.dof,
                "tmin": tmin,
                "tmax": tmax,
            }
        )

    n_vals = np.asarray([row["n"] for row in fit_rows], dtype=int)
    p_vals = np.asarray([row["p"] for row in fit_rows], dtype=float)
    p2 = p_vals**2
    E = np.asarray([row["E0"] for row in fit_rows], dtype=object)
    E2 = E**2

    dispersion_fit = fit_dispersion(p2, E2)
    m2_fit = dispersion_fit.p["m2"]
    c2_fit = dispersion_fit.p["c2"]

    m0 = E[0]
    rel_E_fixed = np.sqrt(m0**2 + p2)
    rel_E2_fixed = m0**2 + p2
    fixed_diff = E2 - rel_E2_fixed
    fixed_pull = gv.mean(fixed_diff) / gv.sdev(fixed_diff)

    curve_p = np.linspace(0.0, max(p_vals) * 1.05, 300)
    curve_p2 = curve_p**2
    curve_E2_free = m2_fit + c2_fit * curve_p2
    curve_E_free = np.sqrt(curve_E2_free)
    curve_E_fixed = np.sqrt(m0**2 + curve_p2)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Ls = {Ls}\n")
        f.write("momentum p_z a = 2*pi*n/Ls\n")
        f.write("2pt fits use sign-fixed -C2_real and selected windows\n\n")
        for row in fit_rows:
            f.write(
                f"{row['psink_key']} n={row['n']} p={row['p']:.8g} "
                f"window=[{row['tmin']},{row['tmax']}) "
                f"E0={row['E0']} Q={row['Q']:.6g} "
                f"chi2/dof={row['chi2_dof']:.6g}\n"
            )
        f.write("\nFree dispersion fit: E^2 = m^2 + c^2 p^2\n")
        f.write(f"m2 = {m2_fit}\n")
        f.write(f"m = {np.sqrt(m2_fit)}\n")
        f.write(f"c2 = {c2_fit}\n")
        f.write(f"Q = {dispersion_fit.Q:.8g}\n")
        f.write(f"chi2/dof = {dispersion_fit.chi2 / dispersion_fit.dof:.8g}\n\n")
        f.write("Fixed relativistic check using m=E(p=0), c2=1\n")
        for row, pull in zip(fit_rows, fixed_pull):
            f.write(f"{row['psink_key']} pull in E^2 = {pull:.6g}\n")

    print(f"Free dispersion fit c2 = {c2_fit}")
    print(f"Free dispersion fit Q = {dispersion_fit.Q:.6g}")
    print(f"Free dispersion fit chi2/dof = {dispersion_fit.chi2 / dispersion_fit.dof:.6g}")
    print("Fixed c2=1 pulls in E^2:")
    for row, pull in zip(fit_rows, fixed_pull):
        print(f"  {row['psink_key']}: {pull:.6g}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    ax = axes[0]
    ax.errorbar(
        p_vals,
        gv.mean(E),
        yerr=gv.sdev(E),
        fmt="o",
        capsize=4,
        mfc="white",
        label="2pt fit E0",
    )
    ax.plot(curve_p, gv.mean(curve_E_fixed), "--", label=r"$\sqrt{m_0^2+p^2}$")
    ax.plot(curve_p, gv.mean(curve_E_free), "-", label=rf"fit $c^2={c2_fit}$")
    ax.fill_between(
        curve_p,
        gv.mean(curve_E_free) - gv.sdev(curve_E_free),
        gv.mean(curve_E_free) + gv.sdev(curve_E_free),
        alpha=0.25,
        label="free fit band",
    )
    for n, p, e_mean in zip(n_vals, p_vals, gv.mean(E)):
        ax.annotate(f"p{n}", (p, e_mean), textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel(r"$p_z a = 2\pi n/L_s$")
    ax.set_ylabel(r"$E a$")
    ax.set_title("Energy dispersion")
    ax.grid(linestyle=":")
    ax.legend()

    ax = axes[1]
    ax.errorbar(
        p2,
        gv.mean(E2),
        yerr=gv.sdev(E2),
        fmt="o",
        capsize=4,
        mfc="white",
        label=r"$E^2$ from 2pt",
    )
    ax.plot(curve_p2, gv.mean(m0**2 + curve_p2), "--", label=r"$m_0^2+p^2$")
    ax.plot(curve_p2, gv.mean(curve_E2_free), "-", label=rf"$m^2+c^2p^2$, $c^2={c2_fit}$")
    ax.fill_between(
        curve_p2,
        gv.mean(curve_E2_free) - gv.sdev(curve_E2_free),
        gv.mean(curve_E2_free) + gv.sdev(curve_E2_free),
        alpha=0.25,
        label="free fit band",
    )
    for n, x, y in zip(n_vals, p2, gv.mean(E2)):
        ax.annotate(f"p{n}", (x, y), textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel(r"$(p_z a)^2$")
    ax.set_ylabel(r"$(E a)^2$")
    ax.set_title("Relativistic dispersion check")
    ax.grid(linestyle=":")
    ax.legend()

    fig.savefig(output_path, dpi=180)
    print(f"saved {output_path}")
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
