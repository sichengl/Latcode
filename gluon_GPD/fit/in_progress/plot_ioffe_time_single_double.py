import csv
import os

import gvar as gv
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


base_dir = os.path.dirname(os.path.abspath(__file__))
input_csv = os.path.join(
    base_dir,
    "plots_joint_ratio",
    "ioffe_time",
    "joint_O3_reduced_ioffe_time_physical_nu_tgf25_30_35_40.csv",
)
output_dir = os.path.join(base_dir, "plots_joint_ratio", "ioffe_time")

output_tag = "joint_O3_physical_nu_tgf25_30_35_40"


def read_fit_table(path):
    rows = []
    o00_by_key = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                "flow": int(row["flow"]),
                "tau_label": row.get("tau_label", row["flow"]),
                "z": int(row["z"]),
                "psink": row["psink"],
                "p": int(row["p"]),
                "nu": float(row["nu"]),
                "O00": gv.gvar(float(row["O00_mean"]), float(row["O00_sdev"])),
                "double": gv.gvar(
                    float(row["reduced_mean"]), float(row["reduced_sdev"])
                ),
            }
            rows.append(item)
            o00_by_key[(item["flow"], item["z"], item["p"])] = item["O00"]

    return rows, o00_by_key


def build_ratio_rows(rows, o00_by_key):
    single_rows = []
    double_rows = []

    for row in rows:
        flow = row["flow"]
        z = row["z"]
        p = row["p"]

        # Single ratio:
        #   M(p,z) / M(0,z)
        single = row["O00"] / o00_by_key[(flow, z, 0)]

        # Double ratio:
        #   [M(p,z) / M(p,0)] / [M(0,z) / M(0,0)]
        # This column was already produced by the joint-fit postprocessing table.
        double = row["double"]

        base = {
            "flow": flow,
            "tau_label": row["tau_label"],
            "z": z,
            "psink": row["psink"],
            "p": p,
            "nu": row["nu"],
        }
        single_rows.append({**base, "ratio": single})
        double_rows.append({**base, "ratio": double})

    return single_rows, double_rows


def write_ratio_csv(rows, path):
    with open(path, "w", newline="") as f:
        fieldnames = [
            "flow",
            "tau_label",
            "z",
            "psink",
            "p",
            "nu",
            "ratio_mean",
            "ratio_sdev",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "flow": row["flow"],
                    "tau_label": row["tau_label"],
                    "z": row["z"],
                    "psink": row["psink"],
                    "p": row["p"],
                    "nu": row["nu"],
                    "ratio_mean": float(gv.mean(row["ratio"])),
                    "ratio_sdev": float(gv.sdev(row["ratio"])),
                }
            )


def plot_ratio_panels(
    rows,
    title,
    ylabel,
    output_path,
    legend_loc="lower left",
    connect_points=False,
):
    flows = sorted({row["flow"] for row in rows})
    z_values = sorted({row["z"] for row in rows})
    y_low = [gv.mean(row["ratio"]) - gv.sdev(row["ratio"]) for row in rows]
    y_high = [gv.mean(row["ratio"]) + gv.sdev(row["ratio"]) for row in rows]
    ymin = min(0.0, float(np.nanmin(y_low)))
    ymax = max(1.2, float(np.nanmax(y_high)))
    ypad = 0.08 * (ymax - ymin)
    ymin -= ypad
    ymax += ypad

    markers = ["o", "*", "D", "^", "s", "v", "P", "X"]
    colors = ["crimson", "blue", "green", "magenta", "gray", "c", "orange", "purple"]
    marker_by_z = {z: markers[i % len(markers)] for i, z in enumerate(z_values)}
    color_by_z = {z: colors[i % len(colors)] for i, z in enumerate(z_values)}

    ncol = 2
    nrow = int(np.ceil(len(flows) / ncol))
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(11.0, 4.0 * nrow),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, flow in zip(axes, flows):
        flow_rows = [row for row in rows if row["flow"] == flow]
        tau_label = flow_rows[0]["tau_label"]

        for z in z_values:
            z_rows = sorted(
                [row for row in flow_rows if row["z"] == z],
                key=lambda item: item["p"],
            )
            x = np.array([row["nu"] for row in z_rows], dtype=float)
            y = np.array([gv.mean(row["ratio"]) for row in z_rows], dtype=float)
            yerr = np.array([gv.sdev(row["ratio"]) for row in z_rows], dtype=float)

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt=marker_by_z[z],
                color=color_by_z[z],
                linestyle="-" if connect_points else "none",
                capsize=3,
                linewidth=1.0 if connect_points else 0.0,
                markersize=6 if connect_points else 5,
                label=rf"$z={z}a$",
            )

        ax.set_title(rf"$\tau={tau_label}$")
        ax.axhline(1.0, color="0.45", linestyle="--", linewidth=0.9)
        ax.set_xlim(0.0, 7.2)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(direction="in")
        ax.grid(False)
        ax.legend(frameon=False, loc=legend_loc, fontsize=9)

    for ax in axes[len(flows) :]:
        ax.axis("off")
    for ax in axes[-ncol:]:
        ax.set_xlabel(r"$\nu=2\pi pz/L_s$")
    for ax in axes[::ncol]:
        ax.set_ylabel(ylabel)

    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(output_dir, exist_ok=True)

    rows, o00_by_key = read_fit_table(input_csv)
    single_rows, double_rows = build_ratio_rows(rows, o00_by_key)

    single_csv = os.path.join(output_dir, f"{output_tag}_single_ratio_ioffe_time.csv")
    double_csv = os.path.join(output_dir, f"{output_tag}_double_ratio_ioffe_time.csv")
    single_png = os.path.join(output_dir, f"{output_tag}_single_ratio_ioffe_time.png")
    double_png = os.path.join(output_dir, f"{output_tag}_double_ratio_ioffe_time.png")

    write_ratio_csv(single_rows, single_csv)
    write_ratio_csv(double_rows, double_csv)

    plot_ratio_panels(
        single_rows,
        r"single ratio $M(p,z)/M(0,z)$",
        r"$M(p,z)/M(0,z)$",
        single_png,
        legend_loc="upper right",
        connect_points=True,
    )
    plot_ratio_panels(
        double_rows,
        r"double ratio reduced pseudo-ITD",
        r"$\mathfrak{M}(\nu,z^2)$",
        double_png,
    )

    print("saved", single_csv)
    print("saved", single_png)
    print("saved", double_csv)
    print("saved", double_png)


if __name__ == "__main__":
    main()
