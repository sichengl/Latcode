import argparse
import glob
import os
import re

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Edit this list directly, or pass repeated --set arguments on the command line.
# Each set reads files produced by 2pt_off_forward_production_splitx.py.
COMPARE_SETS = [
    {
        "label": "N40_rho5p0_frac0p35_G45",
        "N": 40,
        "rho": "5.0",
        "frac": "0p35",
        "gamma": "45",
    },

    {

        "label": "N40_rho5p0_frac0p2_G45",
        "N": 40,
        "rho": "5.0",
        "frac": "0p2",
        "gamma": "45",
    },


    #{
        
     #   "label": "N40_rho5p0_frac-0p3_G45",
      #  "N": 40,
       # "rho": "5.0",
        #"frac": "-0p3",
        #"gamma": "45",
        
    #},
    {
        
        "label": "N40_rho5p0_frac0p7_G45",
        "N": 40,
        "rho": "5.0",
        "frac": "0p7",
        "gamma": "45",

        },
    # Example:
    # {
    #     "label": "N40_rho5p5_frac0p35_G45",
    #     "N": 40,
    #     "rho": "5.5",
    #     "frac": "0p35",
    #     "gamma": "45",
    # },
]


LT = 96
LS = 32
CFG_START = 204
CFG_STEP = 6
N_CFG = 0
T_SRC_BASE = list(range(0, LT, 12))
NORM_T = 3
PLOT_T_MAX = 16
MOMENTA_TO_PLOT = [(0, 0, pz) for pz in range(0, 7)]


def frac_to_str(frac):
    if isinstance(frac, str):
        return frac.replace(".", "p")
    return ("%.3f" % float(frac)).rstrip("0").rstrip(".").replace(".", "p")


def parse_set_spec(spec):
    out = {}
    for item in spec.split(","):
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "N":
            value = int(value)
        out[key] = value
    for key in ["label", "N", "rho", "frac", "gamma"]:
        if key not in out:
            raise ValueError(f"--set is missing {key}: {spec}")
    out["frac"] = frac_to_str(out["frac"])
    out["gamma"] = str(out["gamma"])
    return out


def jackknife(data):
    n_cfg = data.shape[0]
    if n_cfg < 2:
        raise ValueError("Need at least two configurations for jackknife")
    return (np.sum(data, axis=0, keepdims=True) - data) / (n_cfg - 1)


def jk_mean_err(samples):
    n_jk = samples.shape[0]
    avg = np.mean(samples, axis=0)
    err = np.sqrt((n_jk - 1) / n_jk * np.sum((samples - avg) ** 2, axis=0))
    return avg, err


def splitx_file_path(base_dir, run_set, cfg, ix, shifted_x):
    gamma = str(run_set["gamma"])
    dataset = "pion_45" if gamma == "45" else "pion_5"
    gamma_dir = f"G{gamma}"
    frac = frac_to_str(run_set["frac"])
    rho = str(run_set["rho"])
    N = int(run_set["N"])

    directory = os.path.join(base_dir, f"N{N}_rho{rho}_{gamma_dir}_ez_momfrac{frac}")
    filename = f"pion_ix{ix}_x{shifted_x}_N{N}_rho{rho}_frac{frac}_{gamma_dir}_cfg{cfg}.h5"
    return os.path.join(directory, filename), dataset


def splitx_directory(base_dir, run_set):
    gamma = str(run_set["gamma"])
    gamma_dir = f"G{gamma}"
    frac = frac_to_str(run_set["frac"])
    rho = str(run_set["rho"])
    N = int(run_set["N"])

    return os.path.join(base_dir, f"N{N}_rho{rho}_{gamma_dir}_ez_momfrac{frac}")


def dataset_name(run_set):
    gamma = str(run_set["gamma"])
    return "pion_45" if gamma == "45" else "pion_5"


def scan_splitx_files(base_dir, run_set):
    gamma = str(run_set["gamma"])
    gamma_dir = f"G{gamma}"
    frac = frac_to_str(run_set["frac"])
    rho = str(run_set["rho"])
    N = int(run_set["N"])

    directory = splitx_directory(base_dir, run_set)
    pattern = os.path.join(
        directory,
        f"pion_ix*_x*_N{N}_rho{rho}_frac{frac}_{gamma_dir}_cfg*.h5",
    )
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No split-x files found: {pattern}")

    def split_key(path):
        match = re.search(r"pion_ix(\d+)_x(\d+)_.*_cfg(\d+)\.h5$", os.path.basename(path))
        if match is None:
            return (10**9, 10**9, 10**9, path)
        return (int(match.group(3)), int(match.group(1)), int(match.group(2)), path)

    by_cfg = {}
    for path in sorted(paths, key=split_key):
        match = re.search(r"_cfg(\d+)\.h5$", os.path.basename(path))
        if match is None:
            continue
        cfg = int(match.group(1))
        by_cfg.setdefault(cfg, []).append(path)

    return by_cfg


def find_splitx_files(file_map, cfg):
    if cfg not in file_map:
        raise FileNotFoundError(f"No split-x files found for cfg {cfg}")
    return file_map[cfg]


def choose_cfg_list(base_dir, run_set, requested_cfg_list):
    file_map = scan_splitx_files(base_dir, run_set)
    available = sorted(file_map)
    if requested_cfg_list is None:
        return available, file_map
    selected = [cfg for cfg in requested_cfg_list if cfg in file_map]
    if not selected:
        raise FileNotFoundError(
            f"Requested cfgs are not available. First available cfgs: {available[:10]}"
        )
    missing = [cfg for cfg in requested_cfg_list if cfg not in file_map]
    if missing:
        print(f"warning: skipping missing cfgs: {missing[:10]}", flush=True)
    return selected, file_map


def read_one_set(base_dir, run_set, cfg_list, momenta, file_map):
    corr = None
    momentum_ref = None
    mom_indices = None
    dataset = dataset_name(run_set)

    for icfg, cfg in enumerate(cfg_list):
        cfg_blocks = []
        cfg_t_src_list = None
        paths = find_splitx_files(file_map, cfg)
        for path in paths:
            with h5py.File(path, "r") as f:
                block = f[dataset][0, :, 0, :, :, :, :]
                momentum_list = f["momentum_list"][:]
                if cfg_t_src_list is None and "t_src_list" in f[dataset].attrs:
                    cfg_t_src_list = np.array(f[dataset].attrs["t_src_list"], dtype=np.int64)

            if momentum_ref is None:
                momentum_ref = momentum_list
                mom_to_idx = {tuple(p): i for i, p in enumerate(momentum_ref.tolist())}
                missing = [p for p in momenta if tuple(p) not in mom_to_idx]
                if missing:
                    raise ValueError(f"Missing momenta in {path}: {missing}")
                mom_indices = np.array([mom_to_idx[tuple(p)] for p in momenta], dtype=np.int64)
            elif not np.array_equal(momentum_list, momentum_ref):
                raise ValueError(f"momentum_list mismatch in {path}")

            cfg_blocks.append(block)

        cfg_data = np.stack(cfg_blocks, axis=1)
        # cfg_data: [t_src, ix, y_src, z_src, momentum, t]
        if cfg_t_src_list is None:
            ncfg = (cfg - CFG_START) // CFG_STEP
            time_shift = ncfg * 5
            cfg_t_src_list = np.array([(t + time_shift) % LT for t in T_SRC_BASE], dtype=np.int64)
        if len(cfg_t_src_list) != cfg_data.shape[0]:
            raise ValueError(
                f"t_src_list length {len(cfg_t_src_list)} does not match data shape "
                f"{cfg_data.shape[0]} for cfg {cfg}"
            )
        for it, shifted_t in enumerate(cfg_t_src_list):
            cfg_data[it] = np.roll(cfg_data[it], -shifted_t, axis=-1)

        cfg_avg = np.mean(cfg_data[..., mom_indices, :], axis=(0, 1, 2, 3))
        # cfg_avg: [selected_momentum, t]
        if corr is None:
            corr = np.zeros((len(cfg_list), len(momenta), LT), dtype="<c16")
        corr[icfg] = cfg_avg

    return corr


def effective_mass(corr_jk):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.abs(corr_jk[:, :, :-1] / corr_jk[:, :, 1:]))


def normalized_corr(corr_jk, norm_t):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs((corr_jk / corr_jk[:, :, norm_t][:, :, None]).real)


def signal_to_noise(corr_jk):
    avg, err = jk_mean_err(corr_jk.real)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(avg) / err


def effective_overlap_ratio(corr_jk, norm_t):
    meff = effective_mass(corr_jk)
    t = np.arange(meff.shape[-1], dtype=np.float64)
    denom = np.exp(-meff * t[None, None, :]) + np.exp(-meff * (LT - t)[None, None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        zeff = np.sqrt(np.abs(2.0 * meff * corr_jk[:, :, :-1].real / denom))
        return zeff / zeff[:, :, norm_t][:, :, None]


def plot_with_errors(ax, x, jk_values, label, marker):
    avg, err = jk_mean_err(jk_values)
    ax.errorbar(x, avg, yerr=err, fmt=marker, ms=5, capsize=3, label=label)


def save_comparison_plots(all_data, momenta, out_dir, t_max, norm_t):
    os.makedirs(out_dir, exist_ok=True)

    for imom, mom in enumerate(momenta):
        mom_label = f"p{mom[0]}_{mom[1]}_{mom[2]}".replace("-", "m")

        fig, ax = plt.subplots(figsize=(8, 5))
        for iset, item in enumerate(all_data):
            corr_jk = item["corr_jk"]
            meff_jk = effective_mass(corr_jk)[:, imom, :t_max]
            plot_with_errors(ax, np.arange(t_max), meff_jk, item["label"], "o")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$m_{\rm eff}(t)=\log(|C(t)/C(t+1)|)$")
        ax.set_title(f"Effective mass, p={mom}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"meff_{mom_label}.png"), dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for item in all_data:
            corr_jk = item["corr_jk"]
            norm_jk = normalized_corr(corr_jk, norm_t)[:, imom, : t_max + 1]
            plot_with_errors(ax, np.arange(t_max + 1), norm_jk, item["label"], "o")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$|C(t)/C(t_{\rm norm})|$")
        ax.set_title(f"Normalized two-point, p={mom}, t_norm={norm_t}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"normalized_c2_{mom_label}.png"), dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for item in all_data:
            snr = signal_to_noise(item["corr_jk"])[imom, : t_max + 1]
            ax.plot(np.arange(t_max + 1), snr, marker="o", ms=5, label=item["label"])
        ax.set_xlabel("t")
        ax.set_ylabel(r"$|\bar C(t)|/\sigma_{\bar C(t)}$")
        ax.set_yscale("log")
        ax.set_title(f"Signal-to-noise of mean, p={mom}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"snr_{mom_label}.png"), dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for item in all_data:
            corr_jk = item["corr_jk"]
            zeff_jk = effective_overlap_ratio(corr_jk, norm_t)[:, imom, :t_max]
            plot_with_errors(ax, np.arange(t_max), zeff_jk, item["label"], "o")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$Z_{\rm eff}(t)/Z_{\rm eff}(t_{\rm norm})$")
        ax.set_title(f"Effective overlap proxy, p={mom}, t_norm={norm_t}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"overlap_proxy_{mom_label}.png"), dpi=300)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--out-dir", default="./plots_2pt_tuning")
    parser.add_argument("--cfg-start", type=int, default=CFG_START)
    parser.add_argument("--cfg-step", type=int, default=CFG_STEP)
    parser.add_argument("--n-cfg", type=int, default=N_CFG, help="Use first N available cfgs; default 0 means all")
    parser.add_argument("--norm-t", type=int, default=NORM_T)
    parser.add_argument("--t-max", type=int, default=PLOT_T_MAX)
    parser.add_argument(
        "--momenta",
        default=";".join(",".join(str(x) for x in p) for p in MOMENTA_TO_PLOT),
        help="Semicolon-separated momenta, e.g. '0,0,0;0,0,1;0,0,2'",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=None,
        help="Repeatable: label=...,N=40,rho=5.0,frac=0p35,gamma=45",
    )
    args = parser.parse_args()

    momenta = [tuple(int(v) for v in item.split(",")) for item in args.momenta.split(";") if item]
    run_sets = [parse_set_spec(spec) for spec in args.set] if args.set else COMPARE_SETS

    all_data = []
    for run_set in run_sets:
        print(f"reading {run_set['label']}", flush=True)
        requested_cfgs = None
        if args.n_cfg > 0:
            requested_cfgs = list(
                range(args.cfg_start, args.cfg_start + args.cfg_step * args.n_cfg, args.cfg_step)
            )
        cfg_list, file_map = choose_cfg_list(args.base_dir, run_set, requested_cfgs)
        print(f"using {len(cfg_list)} cfgs: {cfg_list[:5]}{'...' if len(cfg_list) > 5 else ''}", flush=True)
        corr = read_one_set(args.base_dir, run_set, cfg_list, momenta, file_map)
        all_data.append(
            {
                "label": run_set["label"],
                "corr": corr,
                "corr_jk": jackknife(corr),
            }
        )

    save_comparison_plots(all_data, momenta, args.out_dir, args.t_max, args.norm_t)
    print(f"saved plots in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

