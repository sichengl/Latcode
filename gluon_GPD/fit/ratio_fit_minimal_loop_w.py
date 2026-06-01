import os

import gvar as gv
import lsqfit as lsf
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


input_path = "./gvar_data/pdf_O3_conn_gvar_jk_manual.p"

psink_key_list = ["p3"]
flow_list = [40]
w_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
tsep_fit_windows = [[4, 5, 6, 7, 8, 9]]
tins_skip_list = [1]
nstate = 2
ratio_sign = 1.0

# Fill these by hand from the two-state C2 fit for the same psink_key.
# You can use either gv.gvar(mean, sdev) or gv.gvar("mean(sdev)").
C2_FIT_GV_BY_PSINK = {
    "p3": {
        "E0": gv.gvar("0.6068(56)"),
        "dE1": gv.gvar("0.329(20)"),
        "z0": gv.gvar("0.00745(28)"),
        "z1": gv.gvar("-0.01145(15)"),
    },
}

def make_prior():
    prior = gv.BufferDict()
    prior["O00"] = gv.gvar(0.0, 10.0)
    prior["O01"] = gv.gvar(0.0, 10.0)  # here assume O01=O10
    prior["O11"] = gv.gvar(0.0, 10.0)
    return prior


output_dir = "./plots"
plot_y_pad_fraction = 0.15


def ratio_key(tsep, psink_key, flow, w):
    return f"tsep{tsep}_{psink_key}_flow{flow}_w{w}"


def energy_levels(c2_params):
    if nstate != 2:
        raise ValueError("ratio_fit_minimal.py currently implements the two-state ratio model only")

    e0 = c2_params["E0"]
    e1 = c2_params["E0"] + c2_params["dE1"]
    return e0, e1


def state_amplitudes(c2_params):
    e0, e1 = energy_levels(c2_params)
    a0 = c2_params["z0"] / gv.sqrt(2.0 * e0)
    a1 = c2_params["z1"] / gv.sqrt(2.0 * e1)
    return a0, a1


def pt2_re_fcn(tsep, c2_params, lt):
    e0, e1 = energy_levels(c2_params)
    a0, a1 = state_amplitudes(c2_params)

    return (
        a0 * a0 * (gv.exp(-e0 * tsep) + gv.exp(-e0 * (lt - tsep)))
        + a1 * a1 * (gv.exp(-e1 * tsep) + gv.exp(-e1 * (lt - tsep)))
    )


def pt3_re_fcn(tsep, tins, c2_params, params):
    e0, e1 = energy_levels(c2_params)
    a0, a1 = state_amplitudes(c2_params)

    return (
        a0 * a0 * params["O00"] * gv.exp(-e0 * tsep)
        + a0
        * a1
        * params["O01"]
        * (
            gv.exp(-e0 * (tsep - tins)) * gv.exp(-e1 * tins)
            + gv.exp(-e1 * (tsep - tins)) * gv.exp(-e0 * tins)
        )
        + a1 * a1 * params["O11"] * gv.exp(-e1 * tsep)
    )


def ratio_fcn(x_values, params, c2_params, lt):
    values = []
    for tsep, tins in x_values:
        c3 = pt3_re_fcn(tsep, tins, c2_params, params)
        c2 = pt2_re_fcn(tsep, c2_params, lt)
        values.append(c3 / c2)
    return values


def collect_ratio_data(data, psink_key, flow, w, tsep_fit_list, tins_skip):
    x_all = []
    y_all = []
    by_tsep = {}

    for tsep in tsep_fit_list:
        key = ratio_key(tsep, psink_key, flow, w)
        if key not in data["ratio"]:
            raise KeyError(f"missing ratio data for {key}")

        ratio_values = list(data["ratio"][key])
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


def fit_one_combination(data, lt, psink_key, flow, w, tsep_fit_list, tins_skip):
    if psink_key not in C2_FIT_GV_BY_PSINK:
        raise KeyError(f"missing C2 fit inputs for {psink_key}")

    c2_fit_gv = C2_FIT_GV_BY_PSINK[psink_key]
    c2_fit = {key: gv.mean(value) for key, value in c2_fit_gv.items()}
    x_fit, y_fit, ratio_by_tsep = collect_ratio_data(
        data, psink_key, flow, w, tsep_fit_list, tins_skip
    )

    def fit_function(x_values, params):
        return ratio_fcn(x_values, params, c2_fit, lt)

    fit = lsf.nonlinear_fit(
        data=(x_fit, y_fit),
        prior=make_prior(),
        fcn=fit_function,
        maxit=10000,
    )

    print("=" * 72)
    print(
        f"fit {psink_key}, flow={flow}, w={w}, "
        f"tsep={min(tsep_fit_list)}-{max(tsep_fit_list)}, tins_skip={tins_skip}"
    )
    print(fit)
    print("C2 inputs:")
    print("E0 =", c2_fit_gv["E0"])
    print("dE1 =", c2_fit_gv["dE1"])
    print("z0 =", c2_fit_gv["z0"])
    print("z1 =", c2_fit_gv["z1"])
    print("ratio matrix elements:")
    print("O00 =", fit.p["O00"])
    print("O01 =", fit.p["O01"])
    print("O11 =", fit.p["O11"])
    print("Q =", fit.Q)
    print("chi2/dof =", fit.chi2 / fit.dof)

    os.makedirs(output_dir, exist_ok=True)
    output_prefix = (
        f"ratio_fit_{psink_key}_flow{flow}_w{w}_"
        f"tsep{min(tsep_fit_list)}to{max(tsep_fit_list)}_skip{tins_skip}"
    )
    output_path = os.path.join(output_dir, f"{output_prefix}.png")

    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(tsep_fit_list)))
    plt.figure(figsize=(8, 5.5))

    all_y_low = []
    all_y_high = []

    for color, tsep in zip(colors, tsep_fit_list):
        tsep_data = ratio_by_tsep[tsep]
        if not tsep_data["y"]:
            continue

        x_centered = tsep_data["x_centered"]
        y_data = tsep_data["y"]

        plt.errorbar(
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
        curve_y = ratio_fcn(curve_x, fit.p, c2_fit, lt)
        curve_mean = gv.mean(curve_y)
        curve_sdev = gv.sdev(curve_y)

        plt.plot(x_plot, curve_mean, color=color)
        plt.fill_between(
            x_plot,
            curve_mean - curve_sdev,
            curve_mean + curve_sdev,
            color=color,
            alpha=0.18,
        )

        data_mean = np.asarray(gv.mean(y_data), dtype=float)
        data_sdev = np.asarray(gv.sdev(y_data), dtype=float)
        curve_mean = np.asarray(curve_mean, dtype=float)
        curve_sdev = np.asarray(curve_sdev, dtype=float)

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
            plt.ylim(ymin - ypad, ymax + ypad)

    plt.axhspan(
        o00_mean - o00_sdev,
        o00_mean + o00_sdev,
        color="black",
        alpha=0.08,
        label="O00 error",
    )
    plt.axhline(
        o00_mean,
        color="black",
        linestyle="--",
        linewidth=2.0,
        label=f"O00={fit.p['O00']}",
    )
    plt.xlabel("t - tsep/2")
    plt.ylabel("ratio")
    plt.title(
        f"{psink_key}, flow={flow}, w={w}, "
        f"chi2/dof={fit.chi2 / fit.dof:.3g}, Q={fit.Q:.3g}"
    )
    plt.grid(linestyle=":")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("saved plot to", output_path)
    return {
        "fit": fit,
        "output_path": output_path,
        "psink_key": psink_key,
        "flow": flow,
        "w": w,
        "tsep_fit_list": list(tsep_fit_list),
        "tins_skip": tins_skip,
    }


def plot_matrix_elements_vs_w(results):
    groups = {}
    for result in results:
        tsep_fit_list = result["tsep_fit_list"]
        group_key = (
            result["psink_key"],
            result["flow"],
            tuple(tsep_fit_list),
            result["tins_skip"],
        )
        groups.setdefault(group_key, []).append(result)

    for (psink_key, flow, tsep_fit_tuple, tins_skip), group_results in groups.items():
        group_results = sorted(group_results, key=lambda item: item["w"])
        w_values = np.array([item["w"] for item in group_results], dtype=float)

        fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
        matrix_keys = ["O00", "O01", "O11"]

        for ax, matrix_key in zip(axes, matrix_keys):
            matrix_values = [item["fit"].p[matrix_key] for item in group_results]
            ax.errorbar(
                w_values,
                gv.mean(matrix_values),
                yerr=gv.sdev(matrix_values),
                fmt="o-",
                capsize=4,
            )
            ax.set_ylabel(matrix_key)
            ax.grid(linestyle=":")

        axes[-1].set_xlabel("w")
        fig.suptitle(
            f"{psink_key}, flow={flow}, "
            f"tsep={min(tsep_fit_tuple)}-{max(tsep_fit_tuple)}, "
            f"tins_skip={tins_skip}"
        )
        fig.tight_layout()

        output_prefix = (
            f"matrix_elements_vs_w_{psink_key}_flow{flow}_"
            f"tsep{min(tsep_fit_tuple)}to{max(tsep_fit_tuple)}_skip{tins_skip}"
        )
        output_path = os.path.join(output_dir, f"{output_prefix}.png")
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

        print("=" * 72)
        print(
            f"matrix elements vs w for {psink_key}, flow={flow}, "
            f"tsep={min(tsep_fit_tuple)}-{max(tsep_fit_tuple)}, tins_skip={tins_skip}"
        )
        for item in group_results:
            fit = item["fit"]
            print(
                f"w={item['w']}: "
                f"O00={fit.p['O00']}, "
                f"O01={fit.p['O01']}, "
                f"O11={fit.p['O11']}, "
                f"Q={fit.Q:.4g}, chi2/dof={fit.chi2 / fit.dof:.4g}"
            )
        print("saved matrix-element plot to", output_path)


data = gv.load(input_path)

if "Lt" in data:
    Lt = int(data["Lt"])
elif "Gt" in data:
    Lt = int(data["Gt"])
else:
    Lt = 96

fit_results = []

for psink_key in psink_key_list:
    for flow in flow_list:
        for w in w_list:
            for tsep_fit_list in tsep_fit_windows:
                for tins_skip in tins_skip_list:
                    fit_results.append(
                        fit_one_combination(
                            data,
                            Lt,
                            psink_key,
                            flow,
                            w,
                            tsep_fit_list,
                            tins_skip,
                        )
                    )

plot_matrix_elements_vs_w(fit_results)

