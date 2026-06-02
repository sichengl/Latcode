#use fited pt2 as prior and fit for all at once
import os
import gvar as gv
import lsqfit as lsf
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


input_path = "./gvar_data/pdf_O3_conn_gvar_jk_manual.p"
output_dir = "./plots_joint_ratio"

psink_key_list = ["p0","p1","p2","p3","p4","p5"]
flow_list = [25,30,35,40]
w_list = [0,1,2,3,4,5,6,7,8,9]
operator_list = ["O3", "txtx", "tyty", "xyxy"]
tsep_fit_windows = [[4, 5, 6, 7, 8, 9]]
tins_skip_list = [1]
ratio_sign = 1.0
plot_y_pad_fraction = 0.15


# Fill these by hand from the two-state C2 fit for each psink_key.
# The fit parameters are E0, log(dE1), z0, z1, O00, O01, O11.
# This helper lets you enter E0 and dE1 from the C2 fit.
def c2_prior(e0, de1, z0, z1):
    if None in (e0, de1, z0, z1):
        return None

    e0 = gv.gvar(e0)
    de1 = gv.gvar(de1)
    return {
        "E0": e0,
        "dE1": de1,
        "z0": gv.gvar(z0),
        "z1": gv.gvar(z1),
    }


C2_PRIOR_GV_BY_PSINK = {
    "p0": c2_prior("0.1414(14)","0.770(73)","0.0001975(22)","-0.00067(10)"),
    "p1": c2_prior("0.2482(28)", "0.579(47)", "0.000900(17)", "0.00206(16)"),
    "p2": c2_prior("0.4281(49)", "0.444(37)", "0.00327(10)", "-0.00566(27)"),
    "p3": c2_prior("0.6225(70)", "0.401(45)", "0.00825(36)", "-0.01197(48)"),
    "p4": c2_prior("0.796(14)", "0.321(47)", "0.0135(12)", "0.01840(25)"),
    "p5": c2_prior("0.961(24)", "0.293(57)", "0.0166(26)", "0.02271(73)"),
}


def get_c2_prior(psink_key):
    if psink_key not in C2_PRIOR_GV_BY_PSINK:
        raise KeyError(f"missing C2 prior row for {psink_key}")

    c2_prior_gv = C2_PRIOR_GV_BY_PSINK[psink_key]
    if c2_prior_gv is None:
        raise ValueError(
            f"C2 priors for {psink_key} are not filled. "
            "Fill C2_PRIOR_GV_BY_PSINK at the top of this file first."
        )

    return c2_prior_gv


def make_prior(psink_key):
    c2_prior_gv = get_c2_prior(psink_key)

    prior = gv.BufferDict()
    prior["E0"] = c2_prior_gv["E0"]
    prior["log(dE1)"] = gv.log(c2_prior_gv["dE1"])
    prior["z0"] = c2_prior_gv["z0"]
    prior["z1"] = c2_prior_gv["z1"]
    prior["O00"] = gv.gvar(0.0, 10.0)
    prior["O01"] = gv.gvar(0.0, 10.0)  # here assume O01=O10
    prior["O11"] = gv.gvar(0.0, 10.0)
    return prior


def energy_levels(p):
    e0 = p["E0"]
    e1 = p["E0"] + p["dE1"]
    return e0, e1


def resolve_input_path(path):
    if os.path.exists(path):
        return path

    fallback_path = os.path.join(".", os.path.basename(path))
    if os.path.exists(fallback_path):
        return fallback_path

    raise FileNotFoundError(f"could not find input data at {path} or {fallback_path}")


def ratio_key(tsep, psink_key, flow, w):
    return f"tsep{tsep}_{psink_key}_flow{flow}_w{w}"


def ratio_data_dict(data, operator_name):
    if operator_name == "O3":
        return data["ratio"]

    if "ratio_components" not in data:
        raise KeyError(
            f"missing ratio_components in input data; rerun gvar_gen.py for {operator_name}"
        )
    if operator_name not in data["ratio_components"]:
        raise KeyError(f"missing ratio component {operator_name}")

    return data["ratio_components"][operator_name]


def state_amplitudes(p):
    e0, e1 = energy_levels(p)
    a0 = p["z0"] / gv.sqrt(2.0 * e0)
    a1 = p["z1"] / gv.sqrt(2.0 * e1)
    return a0, a1


def pt2_re_fcn(tsep, p, lt):
    e0, e1 = energy_levels(p)
    a0, a1 = state_amplitudes(p)
    return (
        a0 * a0 * (gv.exp(-e0 * tsep) + gv.exp(-e0 * (lt - tsep)))
        + a1 * a1 * (gv.exp(-e1 * tsep) + gv.exp(-e1 * (lt - tsep)))
    )


def pt3_re_fcn(tsep, tins, p):
    e0, e1 = energy_levels(p)
    a0, a1 = state_amplitudes(p)
    return (
        a0 * a0 * p["O00"] * gv.exp(-e0 * tsep)
        + a0
        * a1
        * p["O01"]
        * (
            gv.exp(-e0 * (tsep - tins)) * gv.exp(-e1 * tins)
            + gv.exp(-e1 * (tsep - tins)) * gv.exp(-e0 * tins)
        )
        + a1 * a1 * p["O11"] * gv.exp(-e1 * tsep)
    )


def ratio_fcn(x_values, p, lt):
    values = []
    for tsep, tins in x_values:
        values.append(pt3_re_fcn(tsep, tins, p) / pt2_re_fcn(tsep, p, lt))
    return values


def collect_ratio_data(data, operator_name, psink_key, flow, w, tsep_fit_list, tins_skip):
    x_all = []
    y_all = []
    by_tsep = {}
    ratios = ratio_data_dict(data, operator_name)

    for tsep in tsep_fit_list:
        key = ratio_key(tsep, psink_key, flow, w)
        if key not in ratios:
            raise KeyError(f"missing {operator_name} ratio data for {key}")

        ratio_values = list(ratios[key])
        x_centered = data.get("x_centered", {}).get(
            key, np.arange(len(ratio_values), dtype=float) - 0.5 * tsep
        )

        x_tsep = []
        y_tsep = []
        x_centered_tsep = []

        for tins, (x_shifted, y) in enumerate(zip(x_centered, ratio_values)):
            if tins < tins_skip or tins > tsep - tins_skip:
                continue

            x = (float(tsep), float(tins))
            y = ratio_sign * y

            x_all.append(x)
            y_all.append(y)
            x_tsep.append(x)
            y_tsep.append(y)
            x_centered_tsep.append(float(x_shifted))

        by_tsep[tsep] = {
            "x": x_tsep,
            "x_centered": np.array(x_centered_tsep, dtype=float),
            "y": y_tsep,
        }

    return x_all, y_all, by_tsep


def fit_one_combination(data, lt, operator_name, psink_key, flow, w, tsep_fit_list, tins_skip):
    x_fit, y_fit, ratio_by_tsep = collect_ratio_data(
        data, operator_name, psink_key, flow, w, tsep_fit_list, tins_skip
    )

    fit = lsf.nonlinear_fit(
        data=(x_fit, y_fit),
        prior=make_prior(psink_key),
        fcn=lambda x_values, p: ratio_fcn(x_values, p, lt),
        maxit=10000,
    )

    print("=" * 72)
    print(
        f"joint ratio fit {operator_name}, {psink_key}, tgf={flow}, z={w}, "
        f"tsep={min(tsep_fit_list)}-{max(tsep_fit_list)}, tins_skip={tins_skip}"
    )
    print(fit)
    print("fit parameters:")
    fit_e0, fit_e1 = energy_levels(fit.p)
    print("E0 =", fit.p["E0"])
    print("dE1 =", fit.p["dE1"])
    print("E1 =", fit_e1)
    print("z0 =", fit.p["z0"])
    print("z1 =", fit.p["z1"])
    print("O00 =", fit.p["O00"])
    print("O01 =", fit.p["O01"])
    print("O11 =", fit.p["O11"])
    print("Q =", fit.Q)
    print("chi2/dof =", fit.chi2 / fit.dof)

    output_path = plot_ratio_fit(
        fit, ratio_by_tsep, lt, operator_name, psink_key, flow, w, tsep_fit_list, tins_skip
    )

    return {
        "fit": fit,
        "output_path": output_path,
        "operator": operator_name,
        "psink_key": psink_key,
        "flow": flow,
        "w": w,
        "tsep_fit_list": list(tsep_fit_list),
        "tins_skip": tins_skip,
    }


def plot_ratio_fit(
    fit, ratio_by_tsep, lt, operator_name, psink_key, flow, w, tsep_fit_list, tins_skip
):
    os.makedirs(output_dir, exist_ok=True)
    output_prefix = (
        f"joint_ratio_fit_{operator_name}_{psink_key}_tgf{flow}_z{w}_"
        f"tsep{min(tsep_fit_list)}to{max(tsep_fit_list)}_skip{tins_skip}"
    )
    output_path = os.path.join(output_dir, f"{output_prefix}.png")

    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(tsep_fit_list)))
    fig, ax = plt.subplots(figsize=(8, 5.5))

    all_y_low = []
    all_y_high = []

    for color, tsep in zip(colors, tsep_fit_list):
        tsep_data = ratio_by_tsep[tsep]
        if not tsep_data["y"]:
            continue

        x_centered = tsep_data["x_centered"]
        y_data = tsep_data["y"]

        ax.errorbar(
            x_centered,
            gv.mean(y_data),
            yerr=gv.sdev(y_data),
            fmt="o",
            mfc="white",
            color=color,
            capsize=3,
            label=f"tsep={tsep}",
        )

        tins_min = min(tins for _, tins in tsep_data["x"])
        tins_max = max(tins for _, tins in tsep_data["x"])
        tins_plot = np.linspace(tins_min, tins_max, 200)
        x_plot = tins_plot - 0.5 * tsep
        curve_x = [(float(tsep), float(tins)) for tins in tins_plot]
        curve_y = ratio_fcn(curve_x, fit.p, lt)
        curve_mean = np.asarray(gv.mean(curve_y), dtype=float)
        curve_sdev = np.asarray(gv.sdev(curve_y), dtype=float)

        ax.plot(x_plot, curve_mean, color=color)
        ax.fill_between(
            x_plot,
            curve_mean - curve_sdev,
            curve_mean + curve_sdev,
            color=color,
            alpha=0.18,
        )

        data_mean = np.asarray(gv.mean(y_data), dtype=float)
        data_sdev = np.asarray(gv.sdev(y_data), dtype=float)
        all_y_low.extend((data_mean - data_sdev).tolist())
        all_y_high.extend((data_mean + data_sdev).tolist())
        all_y_low.extend((curve_mean - curve_sdev).tolist())
        all_y_high.extend((curve_mean + curve_sdev).tolist())

    o00_mean = float(gv.mean(fit.p["O00"]))
    o00_sdev = float(gv.sdev(fit.p["O00"]))

    if all_y_low and all_y_high:
        all_y_low.append(o00_mean - o00_sdev)
        all_y_high.append(o00_mean + o00_sdev)
        ymin = min(all_y_low)
        ymax = max(all_y_high)
        ypad = plot_y_pad_fraction * (ymax - ymin)
        if ypad > 0:
            ax.set_ylim(ymin - ypad, ymax + ypad)

    ax.axhspan(
        o00_mean - o00_sdev,
        o00_mean + o00_sdev,
        color="gray",
        alpha=0.18,
        label="O00 error",
    )
    ax.axhline(
        o00_mean,
        color="black",
        linestyle="--",
        linewidth=2.0,
        label=f"O00={fit.p['O00']}",
    )
    ax.set_xlabel("t - tsep/2")
    ax.set_ylabel("3pt / 2pt")
    ax.set_title(
        f"{operator_name}, {psink_key}, tgf={flow}, z={w}, "
        f"chi2/dof={fit.chi2 / fit.dof:.3g}, Q={fit.Q:.3g}"
    )
    ax.grid(linestyle=":")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print("saved ratio plot to", output_path)
    return output_path


def plot_o00_vs_z_by_flow(results):
    groups = {}
    for result in results:
        group_key = (
            result["operator"],
            result["psink_key"],
            tuple(result["tsep_fit_list"]),
            result["tins_skip"],
        )
        groups.setdefault(group_key, []).append(result)

    for (operator_name, psink_key, tsep_fit_tuple, tins_skip), group_results in groups.items():
        flow_values = sorted({item["flow"] for item in group_results})
        colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(flow_values)))

        fig, ax = plt.subplots(figsize=(7, 5))

        for color, flow in zip(colors, flow_values):
            flow_results = [
                item for item in group_results if item["flow"] == flow
            ]
            flow_results = sorted(flow_results, key=lambda item: item["w"])

            z_values = np.array([item["w"] for item in flow_results], dtype=float)
            o00_values = [item["fit"].p["O00"] for item in flow_results]

            ax.errorbar(
                z_values,
                np.asarray(gv.mean(o00_values), dtype=float),
                yerr=np.asarray(gv.sdev(o00_values), dtype=float),
                fmt="o-",
                color=color,
                capsize=4,
                label=f"tgf={flow}",
            )

        ax.set_xlabel("Wilson-line length z / a")
        ax.set_ylabel("O00")
        ax.set_title(
            f"joint fit {operator_name} O00 vs Wilson-line length, {psink_key}, "
            f"tsep={min(tsep_fit_tuple)}-{max(tsep_fit_tuple)}, "
            f"tins_skip={tins_skip}"
        )
        ax.grid(linestyle=":")
        ax.legend()
        fig.tight_layout()

        output_prefix = (
            f"joint_{operator_name}_O00_vs_wilson_length_by_tgf_{psink_key}_"
            f"tsep{min(tsep_fit_tuple)}to{max(tsep_fit_tuple)}_skip{tins_skip}"
        )
        output_path = os.path.join(output_dir, f"{output_prefix}.png")
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

        print("=" * 72)
        print(
            f"joint fit {operator_name} O00 vs z by tgf for {psink_key}, "
            f"tsep={min(tsep_fit_tuple)}-{max(tsep_fit_tuple)}, tins_skip={tins_skip}"
        )
        for flow in flow_values:
            flow_results = [
                item for item in group_results if item["flow"] == flow
            ]
            flow_results = sorted(flow_results, key=lambda item: item["w"])
            for item in flow_results:
                fit = item["fit"]
                print(
                    f"tgf={flow}, z={item['w']}: "
                    f"O00={fit.p['O00']}, "
                    f"E0={fit.p['E0']}, "
                    f"dE1={fit.p['dE1']}, "
                    f"E1={energy_levels(fit.p)[1]}, "
                    f"Q={fit.Q:.4g}, chi2/dof={fit.chi2 / fit.dof:.4g}"
                )
        print("saved joint O00-vs-z plot to", output_path)


def main():
    resolved_input_path = resolve_input_path(input_path)
    print("loading data from", resolved_input_path)
    data = gv.load(resolved_input_path)

    if "Lt" in data:
        lt = int(data["Lt"])
    elif "Gt" in data:
        lt = int(data["Gt"])
    else:
        lt = 96

    fit_results = []
    for operator_name in operator_list:
        for psink_key in psink_key_list:
            for flow in flow_list:
                for w in w_list:
                    for tsep_fit_list in tsep_fit_windows:
                        for tins_skip in tins_skip_list:
                            fit_results.append(
                                fit_one_combination(
                                    data,
                                    lt,
                                    operator_name,
                                    psink_key,
                                    flow,
                                    w,
                                    tsep_fit_list,
                                    tins_skip,
                                )
                            )

    plot_o00_vs_z_by_flow(fit_results)


if __name__ == "__main__":
    main()
