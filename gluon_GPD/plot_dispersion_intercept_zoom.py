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
output_path = os.path.join(output_dir, "dispersion_intercept_zoom.png")

Ls = 32
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


def fit_pt2(t_fit, y_fit, Lt):
    prior = estimate_prior(t_fit, y_fit)

    def fcn(t, p):
        return pt2_re_fcn(t, p, Lt, nstate=nstate)

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
    Lt = int(data.get("Lt", data.get("Gt", 96)))
    sign = pt2_input_sign(data)
    t = np.asarray(data["pt2_tsep_list"], dtype=float)

    rows = []
    for psink_key in sorted(data["pt2_by_psink"], key=psink_to_n):
        n = psink_to_n(psink_key)
        tmin, tmax = fit_window_by_psink[psink_key]
        y = sign * np.asarray(data["pt2_by_psink"][psink_key]["real"], dtype=object)
        mask = (t >= tmin) & (t < tmax)
        fit = fit_pt2(t[mask], y[mask], Lt)
        rows.append((psink_key, n, 2.0 * np.pi * n / Ls, fit.p["E0"]))

    labels = [row[0] for row in rows]
    p = np.asarray([row[2] for row in rows], dtype=float)
    p2 = p**2
    E = np.asarray([row[3] for row in rows], dtype=object)
    E2 = E**2
    m0 = E[0]
    m02 = m0**2
    disp_fit = fit_dispersion(p2, E2)

    curve_p2 = np.linspace(0.0, max(p2) * 1.05, 300)
    curve_fixed = m02 + curve_p2
    curve_free = disp_fit.p["m2"] + disp_fit.p["c2"] * curve_p2

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    ax = axes[0]
    ax.errorbar(p2, gv.mean(E2), yerr=gv.sdev(E2), fmt="o", capsize=4, mfc="white")
    ax.plot(curve_p2, gv.mean(curve_fixed), "--", label=rf"$m_0^2+p^2$, $m_0^2={m02}$")
    ax.plot(curve_p2, gv.mean(curve_free), "-", label=rf"free intercept $m^2={disp_fit.p['m2']}$")
    ax.axhline(gv.mean(m02), color="grey", linestyle=":", label=r"$m_0^2$")
    ax.fill_between(
        curve_p2,
        gv.mean(curve_free) - gv.sdev(curve_free),
        gv.mean(curve_free) + gv.sdev(curve_free),
        alpha=0.25,
    )
    for label, x, y in zip(labels, p2, gv.mean(E2)):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4))
    ax.set_xlim(-0.01, 0.20)
    ax.set_ylim(0.0, 0.22)
    ax.set_xlabel(r"$(p_z a)^2$")
    ax.set_ylabel(r"$(E a)^2$")
    ax.set_title("Intercept zoom")
    ax.grid(linestyle=":")
    ax.legend(fontsize=9)

    ax = axes[1]
    shifted = E2 - m02
    ax.errorbar(p2, gv.mean(shifted), yerr=gv.sdev(shifted), fmt="o", capsize=4, mfc="white")
    ax.plot(curve_p2, curve_p2, "--", label=r"$p^2$")
    ax.plot(curve_p2, gv.mean(curve_free - disp_fit.p["m2"]), "-", label=rf"free slope $c^2={disp_fit.p['c2']}$")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    for label, x, y in zip(labels, p2, gv.mean(shifted)):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel(r"$(p_z a)^2$")
    ax.set_ylabel(r"$(E a)^2 - (m_0 a)^2$")
    ax.set_title("Subtract intercept")
    ax.grid(linestyle=":")
    ax.legend(fontsize=9)

    fig.savefig(output_path, dpi=180)
    print(f"m0 = {m0}")
    print(f"m0^2 = {m02}")
    print(f"free m^2 = {disp_fit.p['m2']}")
    print(f"free c^2 = {disp_fit.p['c2']}")
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
