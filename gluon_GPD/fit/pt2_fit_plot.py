import os
import re
import sys

import gvar as gv
import lsqfit as lsf
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.append(os.path.join(os.path.dirname(__file__), "LaMETLat-main", "src"))

from lametlat.ground_state.fit_funcs import pt2_re_fcn
from lametlat.plotting.plot_settings import (
    ERRORBAR_STYLE,
    FONT_SIZE,
    LEGEND_SIZE,
    TSEP_LABEL,
    default_plot,
)


input_path = "./gvar_data/pdf_O3_conn_gvar_jk_manual.p"
output_dir = "./plots"
DATA_ERRORBAR_STYLE = {
    **ERRORBAR_STYLE,
    "marker": "o",
    "markersize": 5,
    "mfc": "none",
    "elinewidth": 1.4,
    "capsize": 4,
}

boundary = "periodic"
nstate = 2
default_Lt = 96
default_tmin = 2
default_tmax = 10
default_tgf = 30
default_w = 0
pt2_real_sign = -1.0
fit_window_by_psink = {
    "p0": (3, 15),
    "p1": (4, 14),
    "p2": (4, 14),
    "p3": (5, 15),
    "p4": (3, 13),
    "p5": (3, 15),
}


def pt2_input_sign(data):
    if data.get("pt2_by_psink_sign_fixed", False):
        return 1.0
    return pt2_real_sign


def estimate_pt2_prior_scale(t_data, y_data):
    mean = np.asarray(gv.mean(y_data), dtype=float)
    t_data = np.asarray(t_data, dtype=float)

    positive = mean > 0
    if np.count_nonzero(positive) >= 2:
        t_pos = t_data[positive]
        y_pos = mean[positive]
        e0_guess = float(np.median(np.log(y_pos[:-1] / y_pos[1:]) / np.diff(t_pos)))
    else:
        e0_guess = 0.5

    if not np.isfinite(e0_guess) or e0_guess <= 0:
        e0_guess = 0.5
    e0_guess = float(np.clip(e0_guess, 0.05, 2.0))

    amp = float(np.nanmax(np.abs(mean)))
    if not np.isfinite(amp) or amp <= 0:
        amp = 1.0

    t0 = float(np.min(t_data))
    z_guess = np.sqrt(max(2.0 * e0_guess * amp * np.exp(e0_guess * t0), 1e-30))
    return e0_guess, float(z_guess)


def pt2_prior(t_data, y_data):
    e0_guess, z_guess = estimate_pt2_prior_scale(t_data, y_data)
    z_width = max(10.0 * abs(z_guess), 10.0)

    prior = gv.BufferDict()
    prior["E0"] = gv.gvar(e0_guess, max(0.5 * e0_guess, 0.5))
    prior["log(dE1)"] = gv.gvar(np.log(0.5), 1.0)
    prior["z0"] = gv.gvar(z_guess, z_width)
    prior["z1"] = gv.gvar(0.0, z_width)
    return prior


def format_prior_value(value):
    return f"{float(gv.mean(value)):.6g} +/- {float(gv.sdev(value)):.6g}"


def fit_pt2_twostate(t_data, y_data, Lt):
    def fcn(t, p):
        return pt2_re_fcn(t, p, Lt, nstate=nstate)

    prior = pt2_prior(t_data, y_data)
    fit = lsf.nonlinear_fit(
        data=(t_data, y_data),
        prior=prior,
        fcn=fcn,
        maxit=10000,
    )
    return fit, prior


def mark_fit_region(ax, t_values, *, label=None):
    t_values = np.asarray(t_values, dtype=float)
    left = float(np.min(t_values))
    right = float(np.max(t_values))
    ax.add_patch(
        Rectangle(
            (left, 0.0),
            right - left,
            1.0,
            transform=ax.get_xaxis_transform(),
            facecolor="grey",
            alpha=0.28,
            edgecolor="none",
            zorder=0,
            label=label,
        )
    )


def pt2_to_meff_local(pt2_array, boundary="periodic"):
    data = np.asarray(pt2_array, dtype=object)

    if boundary in {"periodic", "anti-periodic"}:
        return np.arccosh((data[2:] + data[:-2]) / (2 * data[1:-1]))
    if boundary == "none":
        return np.log(data[:-1] / data[1:])
    raise ValueError(f"unsupported boundary mode: {boundary!r}")


def meff_t_values(t_values, boundary="periodic"):
    t_values = np.asarray(t_values, dtype=float)

    if boundary in {"periodic", "anti-periodic"}:
        return t_values[1:-1]
    if boundary == "none":
        return t_values[:-1]
    raise ValueError(f"unsupported boundary mode: {boundary!r}")


def positive_for_log(mean, sdev):
    mean = np.asarray(mean, dtype=float)
    sdev = np.asarray(sdev, dtype=float)
    return np.all(mean > 0) and np.all(mean - sdev > 0)


def load_full_pt2(source_data, psink_key, fit_info):
    if source_data is not None and "pt2_by_psink" in source_data:
        pt2_by_psink = source_data["pt2_by_psink"]
        if psink_key in pt2_by_psink:
            sign = pt2_input_sign(source_data)
            return (
                np.asarray(
                    source_data.get("pt2_tsep_list", source_data["tsep_list"]),
                    dtype=float,
                ),
                sign * np.asarray(pt2_by_psink[psink_key]["real"], dtype=object),
            )

    return (
        np.asarray(fit_info["t_data"], dtype=float),
        np.asarray(fit_info["pt2_real"], dtype=object),
    )


def infer_lattice_time(data, source_data=None):
    for item in (data, source_data):
        if item is None:
            continue

        for key in ("Lt", "Gt"):
            if key in item:
                return int(item[key])

        if "tsrc_list" in item:
            tsrc = np.asarray(item["tsrc_list"], dtype=int)
            if len(tsrc) > 1:
                step = int(np.median(np.diff(np.sort(tsrc))))
                return int(np.max(tsrc) + step)

    print(f"input data does not contain Lt/Gt; using default Lt={default_Lt}")
    return default_Lt


def infer_fit_window(data):
    tmin = data.get("pt2_tmin", data.get("tmin", default_tmin))
    tmax = data.get("pt2_tmax", data.get("tmax", default_tmax))
    return int(tmin), int(tmax)


def psink_keys_from_raw_data(data):
    if "pt2_by_psink" in data:
        return sorted(data["pt2_by_psink"])

    if "psink_phys_list" in data:
        return [f"p{psink}" for psink in data["psink_phys_list"]]

    keys = set()
    for key in data.get("pt2", {}):
        match = re.search(r"_p([^_]+)_", key)
        if match:
            keys.add(f"p{match.group(1)}")
    return sorted(keys)


def load_raw_pt2_for_psink(data, psink_key):
    if "pt2_tsep_list" not in data:
        raise KeyError(
            "input gvar file does not contain pt2_tsep_list. "
            "Rerun gvar_gen.py so the 2pt fit data include t=2..9."
        )

    tsep_list = np.asarray(data["pt2_tsep_list"], dtype=float)
    sign = pt2_input_sign(data)

    if "pt2_by_psink" in data and psink_key in data["pt2_by_psink"]:
        pt2 = data["pt2_by_psink"][psink_key]
        return (
            tsep_list,
            sign * np.asarray(pt2["real"], dtype=object),
            np.asarray(pt2["imag"], dtype=object),
        )

    tgf = data.get("tgf_list", [default_tgf])[0]
    w = data.get("w_list", [default_w])[0]

    pt2_real = []
    pt2_imag = []
    for tsep in tsep_list:
        tsep_key = int(tsep)
        key = f"tsep{tsep_key}_{psink_key}_flow{tgf}_w{w}"
        if key not in data["pt2"]:
            prefix = f"tsep{tsep_key}_{psink_key}_"
            matches = [candidate for candidate in data["pt2"] if candidate.startswith(prefix)]
            if not matches:
                raise KeyError(f"missing pt2 data for {prefix}")
            key = matches[0]

        pt2_real.append(data["pt2"][key]["real"])
        pt2_imag.append(data["pt2"][key]["imag"])

    return tsep_list, sign * gv.gvar(pt2_real), gv.gvar(pt2_imag)


def fit_raw_pt2_for_psink(data, psink_key, Lt, tmin, tmax):
    t_data, pt2_real, pt2_imag = load_raw_pt2_for_psink(data, psink_key)
    mask = (t_data >= tmin) & (t_data < tmax)
    if not np.any(mask):
        raise ValueError(f"empty fit window [{tmin}, {tmax}) for {psink_key}")

    fit_t = t_data[mask]
    expected_t = np.arange(tmin, tmax, dtype=float)
    if not np.array_equal(fit_t, expected_t):
        raise ValueError(
            f"fit window [{tmin}, {tmax}) requires t={expected_t.astype(int).tolist()}, "
            f"but {input_path} contains t={fit_t.astype(int).tolist()} for {psink_key}. "
            "Rerun gvar_gen.py before pt2_fit_plot.py."
        )

    fit, prior = fit_pt2_twostate(t_data[mask], pt2_real[mask], Lt)
    fit_info = {
        "t_data": t_data[mask],
        "pt2_real": pt2_real[mask],
        "pt2_imag": pt2_imag[mask],
        "pt2_real_sign": pt2_input_sign(data),
        "prior": prior,
        "fit_p": fit.p,
        "Q": fit.Q,
        "chi2": fit.chi2,
        "dof": fit.dof,
        "chi2_dof": fit.chi2 / fit.dof,
        "logGBF": fit.logGBF,
        "format": fit.format(True),
    }
    return fit_info, t_data, pt2_real


def print_fit_diagnostics(psink_key, fit_info):
    t_fit = np.asarray(fit_info["t_data"], dtype=int)
    y_fit = np.asarray(fit_info["pt2_real"], dtype=object)
    prior = fit_info.get("prior", pt2_prior(t_fit, y_fit))
    mean = gv.mean(y_fit)
    sdev = gv.sdev(y_fit)
    rel_sdev = np.divide(sdev, np.abs(mean), out=np.full_like(sdev, np.nan), where=mean != 0)

    print(f"  fit t = {t_fit.tolist()}")
    print("  prior:")
    print(f"    E0 = {format_prior_value(prior['E0'])}")
    print(
        f"    log(dE1) = {format_prior_value(prior['log(dE1)'])} "
        f"(dE1 ~= {format_prior_value(gv.exp(prior['log(dE1)']))})"
    )
    print(f"    z0 = {format_prior_value(prior['z0'])}")
    print(f"    z1 = {format_prior_value(prior['z1'])}")
    print(
        "  C2 rel err range = "
        f"{np.nanmin(rel_sdev):.3g} .. {np.nanmax(rel_sdev):.3g}"
    )


def append_fit_debug(psink_key, fit_info, Lt):
    debug_path = os.path.join(output_dir, "pt2_fit_debug.txt")
    t_fit = np.asarray(fit_info["t_data"], dtype=float)
    y_fit = np.asarray(fit_info["pt2_real"], dtype=object)
    fit_y = pt2_re_fcn(t_fit, fit_info["fit_p"], Lt, nstate=nstate)

    data_mean = np.asarray(gv.mean(y_fit), dtype=float)
    data_sdev = np.asarray(gv.sdev(y_fit), dtype=float)
    fit_mean = np.asarray(gv.mean(fit_y), dtype=float)
    pull = np.divide(
        data_mean - fit_mean,
        data_sdev,
        out=np.full_like(data_mean, np.nan),
        where=data_sdev != 0,
    )

    prior = fit_info.get("prior", pt2_prior(t_fit, y_fit))

    with open(debug_path, "a", encoding="utf-8") as f:
        f.write(f"[{psink_key}]\n")
        f.write(f"Lt = {Lt}\n")
        f.write(f"pt2_real_sign = {fit_info.get('pt2_real_sign', pt2_real_sign)}\n")
        f.write(f"fit_t = {t_fit.astype(int).tolist()}\n")
        f.write(f"Q = {fit_info['Q']:.8g}\n")
        f.write(f"chi2/dof = {fit_info['chi2_dof']:.8g}\n")
        f.write(f"prior E0 = {format_prior_value(prior['E0'])}\n")
        f.write(f"prior log(dE1) = {format_prior_value(prior['log(dE1)'])}\n")
        f.write(f"prior dE1 ~= {format_prior_value(gv.exp(prior['log(dE1)']))}\n")
        f.write(f"prior z0 = {format_prior_value(prior['z0'])}\n")
        f.write(f"prior z1 = {format_prior_value(prior['z1'])}\n")
        f.write(f"fit E0 = {fit_info['fit_p']['E0']}\n")
        f.write(f"fit dE1 = {fit_info['fit_p']['dE1']}\n")
        f.write("t sign_fixed_data_mean data_sdev fit_mean pull\n")
        for t, y, dy, fy, r in zip(t_fit, data_mean, data_sdev, fit_mean, pull):
            f.write(f"{int(t)} {y:.16e} {dy:.16e} {fy:.16e} {r:.8g}\n")
        f.write("\n")


def save_c2pt_plot(psink_key, fit_info, Lt, tmin, tmax, t_data, pt2_real):
    fit_p = fit_info["fit_p"]

    fit_t = np.linspace(float(np.min(t_data)), float(np.max(t_data)), 200)
    fit_y = pt2_re_fcn(fit_t, fit_p, Lt, nstate=nstate)
    fit_mask = (t_data >= tmin) & (t_data < tmax)

    data_mean = gv.mean(pt2_real)
    data_sdev = gv.sdev(pt2_real)
    fit_mean = gv.mean(fit_y)
    fit_sdev = gv.sdev(fit_y)
    fit_t_label = f"{int(t_data[fit_mask][0])}-{int(t_data[fit_mask][-1])}"

    fig, ax = default_plot()
    mark_fit_region(
        ax,
        t_data[fit_mask],
        label="fit region",
    )
    ax.errorbar(
        t_data,
        data_mean,
        yerr=data_sdev,
        label="Data",
        **DATA_ERRORBAR_STYLE,
    )
    ax.plot(
        t_data[fit_mask],
        data_mean[fit_mask],
        "o",
        color="tab:blue",
        markersize=4,
        label="fit points",
        zorder=3,
    )
    ax.plot(fit_t, fit_mean, label="Two-state fit")
    ax.fill_between(
        fit_t,
        fit_mean - fit_sdev,
        fit_mean + fit_sdev,
        alpha=0.45,
        label="fit band",
    )
    ax.plot(fit_t, fit_mean - fit_sdev, linewidth=0.8, alpha=0.8)
    ax.plot(fit_t, fit_mean + fit_sdev, linewidth=0.8, alpha=0.8)

    ax.set_xlabel(TSEP_LABEL, **FONT_SIZE)
    ax.set_ylabel(r"$-C_{2\mathrm{pt}}^{\mathrm{real}}(t_{\mathrm{sep}})$", **FONT_SIZE)
    ax.set_title(
        rf"{psink_key}, fit $t={fit_t_label}$, "
        rf"$\chi^2/\mathrm{{dof}}={fit_info['chi2_dof']:.3g}$, "
        rf"$Q={fit_info['Q']:.3g}$",
        **FONT_SIZE,
    )
    ax.legend(**LEGEND_SIZE)

    stem = os.path.join(output_dir, f"pt2_twostate_fit_{psink_key}_c2pt")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", transparent=True)
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_meff_plot(psink_key, fit_info, Lt, tmin, tmax, t_data, pt2_real):
    fit_p = fit_info["fit_p"]

    if len(t_data) > 1 and not np.allclose(np.diff(t_data), 1):
        print(f"skip meff plot for {psink_key}: tsep values are not contiguous")
        return
    if len(t_data) < 3 and boundary in {"periodic", "anti-periodic"}:
        return
    if len(t_data) < 2 and boundary == "none":
        return

    fit_t = np.arange(int(np.min(t_data)), int(np.max(t_data)) + 1, dtype=float)
    fit_y = pt2_re_fcn(fit_t, fit_p, Lt, nstate=nstate)

    data_meff = pt2_to_meff_local(pt2_real, boundary=boundary)
    fit_meff = pt2_to_meff_local(fit_y, boundary=boundary)

    data_t = meff_t_values(t_data, boundary=boundary)
    fit_t_meff = meff_t_values(fit_t, boundary=boundary)
    data_meff_mean = gv.mean(data_meff)

    fig, ax = default_plot()
    fit_data_meff_mask = (data_t >= tmin) & (data_t < tmax)
    mark_fit_region(
        ax,
        data_t[fit_data_meff_mask],
        label="fit region",
    )
    ax.errorbar(
        data_t,
        data_meff_mean,
        yerr=gv.sdev(data_meff),
        label="Data",
        **ERRORBAR_STYLE,
    )
    ax.plot(fit_t_meff, gv.mean(fit_meff), label="Two-state fit")
    ax.fill_between(
        fit_t_meff,
        gv.mean(fit_meff) - gv.sdev(fit_meff),
        gv.mean(fit_meff) + gv.sdev(fit_meff),
        alpha=0.35,
    )

    ax.set_xlabel(TSEP_LABEL, **FONT_SIZE)
    ax.set_ylabel(r"${m}_{\mathrm{eff}}$", **FONT_SIZE)
    ax.set_title(
        rf"{psink_key}, fit $t={tmin}-{tmax - 1}$, "
        rf"$\chi^2/\mathrm{{dof}}={fit_info['chi2_dof']:.3g}$, "
        rf"$Q={fit_info['Q']:.3g}$",
        **FONT_SIZE,
    )
    ax.legend(**LEGEND_SIZE)

    stem = os.path.join(output_dir, f"pt2_twostate_fit_{psink_key}_meff")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", transparent=True)
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "pt2_fit_debug.txt"), "w", encoding="utf-8") as f:
        f.write(f"input_path = {input_path}\n")
        f.write(f"fit window = [{default_tmin}, {default_tmax}) unless overridden by input\n\n")

    fit_data = gv.load(input_path)

    if "fit_results" not in fit_data:
        Lt = infer_lattice_time(fit_data)
        tmin, tmax = infer_fit_window(fit_data)

        for psink_key in psink_keys_from_raw_data(fit_data):
            fit_tmin, fit_tmax = fit_window_by_psink.get(psink_key, (tmin, tmax))
            fit_info, t_data, pt2_real = fit_raw_pt2_for_psink(
                fit_data,
                psink_key,
                Lt,
                fit_tmin,
                fit_tmax,
            )
            save_c2pt_plot(psink_key, fit_info, Lt, fit_tmin, fit_tmax, t_data, pt2_real)
            save_meff_plot(psink_key, fit_info, Lt, fit_tmin, fit_tmax, t_data, pt2_real)
            print(f"saved pt2 fit plots for {psink_key}")
            print(f"  Q = {fit_info['Q']:.6g}")
            print(f"  chi2/dof = {fit_info['chi2_dof']:.6g}")
            print(f"  E0 = {fit_info['fit_p']['E0']}")
            print(f"  dE1 = {fit_info['fit_p']['dE1']}")
            print_fit_diagnostics(psink_key, fit_info)
            append_fit_debug(psink_key, fit_info, Lt)

        print(f"saved plots to {output_dir}")
        return

    source_data = None
    source_path = fit_data.get("input_path")
    if source_path is not None and os.path.exists(source_path):
        source_data = gv.load(source_path)

    Lt = infer_lattice_time(fit_data, source_data)
    tmin, tmax = infer_fit_window(fit_data)

    for psink_key, fit_info in fit_data["fit_results"].items():
        t_data, pt2_real = load_full_pt2(source_data, psink_key, fit_info)
        save_c2pt_plot(psink_key, fit_info, Lt, tmin, tmax, t_data, pt2_real)
        save_meff_plot(psink_key, fit_info, Lt, tmin, tmax, t_data, pt2_real)
        print(f"saved pt2 fit plots for {psink_key}")
        print_fit_diagnostics(psink_key, fit_info)
        append_fit_debug(psink_key, fit_info, Lt)

    print(f"saved plots to {output_dir}")


if __name__ == "__main__":
    main()
