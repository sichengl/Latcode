import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gvar as gv
import h5py
import lsqfit
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "ratio_jk_manual_real2pt"
OUTPUT_DIR = SCRIPT_DIR / "bare_matrix_element_jk"

CFG_LIST = list(range(204, 204 + 200 * 6, 6))
TSEP_LIST = [4, 5, 6, 7, 8]
Q_LIST = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
PF_LIST = [(0, 0, pz) for pz in range(7)]
W_LIST = [0, 1, 2, 3,4,5,6,7,8,9]
TGF_LIST = [20, 25, 30, 35, 40]
OPERATOR_LIST = ["TXTX", "TYTY"]

# Drop this many insertion-time points from both source and sink ends.
TAU_SKIP = 1

FIT_TAG = "two_state"
OVERWRITE = True
N_WORKERS = int(os.environ.get("N_WORKERS", "8"))
TASK_BATCH_SIZE = int(os.environ.get("TASK_BATCH_SIZE", str(N_WORKERS)))

SVD_CUT = 1e-4
COV_EPS = 1e-12
MAXIT = 20000

PRIORS = {
    "M00": (0.0, 2.0),
    "Ai": (0.0, 5.0),
    "Af": (0.0, 5.0),
    "Afi": (0.0, 5.0),
    "dEi": (0.5, 5),
    "dEf": (0.5, 5),
}

MODEL_TEXT = (
    "R(T,tau) = M00 + Ai*exp(-dEi*tau) + Af*exp(-dEf*(T-tau)) "
    "+ Afi*exp(-dEi*tau-dEf*(T-tau))"
)


def ratio_model(x, p):
    w_index = x["w_index"]
    tau = x["tau"]
    t_minus_tau = x["T"] - x["tau"]
    return (
        p["M00"][w_index]
        + p["Ai"][w_index] * gv.exp(-p["dEi"] * tau)
        + p["Af"][w_index] * gv.exp(-p["dEf"] * t_minus_tau)
        + p["Afi"][w_index] * gv.exp(-p["dEi"] * tau - p["dEf"] * t_minus_tau)
    )


def jackknife_covariance(samples):
    n_jk = samples.shape[0]
    diff = samples - np.mean(samples, axis=0, keepdims=True)
    cov = (n_jk - 1.0) / n_jk * diff.T @ diff
    cov = 0.5 * (cov + cov.T)
    diag = np.diag(cov)
    positive = diag[diag > 0]
    scale = float(np.mean(positive)) if positive.size else 1.0
    return cov + np.eye(cov.shape[0]) * COV_EPS * scale


def fit_jackknife_samples(samples, x, prior, label):
    n_jk = samples.shape[0]
    fit_params = {
        "M00": np.full((n_jk, len(W_LIST)), np.nan),
        "Ai": np.full((n_jk, len(W_LIST)), np.nan),
        "Af": np.full((n_jk, len(W_LIST)), np.nan),
        "Afi": np.full((n_jk, len(W_LIST)), np.nan),
        "dEi": np.full(n_jk, np.nan),
        "dEf": np.full(n_jk, np.nan),
    }
    chi2 = np.full(n_jk, np.nan)
    dof = np.full(n_jk, -1, dtype=np.int64)
    q_value = np.full(n_jk, np.nan)
    status = np.zeros(n_jk, dtype=np.int64)
    cov = jackknife_covariance(samples)

    for ijk in range(n_jk):
        try:
            y = gv.gvar(samples[ijk], cov)
            fit = lsqfit.nonlinear_fit(
                data=(x, y),
                prior=prior,
                fcn=ratio_model,
                svdcut=SVD_CUT,
                maxit=MAXIT,
            )
            for name in fit_params:
                fit_params[name][ijk] = gv.mean(fit.p[name])
            chi2[ijk] = fit.chi2
            dof[ijk] = fit.dof
            q_value[ijk] = fit.Q
        except Exception as err:
            status[ijk] = 1
            print(f"fit failed: {label}, jackknife sample {ijk}: {err}", flush=True)

    return fit_params, chi2, dof, q_value, status


def process_parameter_point(operator, q_tuple, pf_tuple, flow):
    q_label = f"q{q_tuple[0]}_{q_tuple[1]}_{q_tuple[2]}"
    pf_label = f"pf{pf_tuple[0]}_{pf_tuple[1]}_{pf_tuple[2]}"

    fit_values = []
    fit_w_index = []
    fit_T = []
    fit_tau = []

    for iw, w in enumerate(W_LIST):
        ratio_name = (
            f"{operator}_ratio_jk_{q_label}_{pf_label}_"
            f"w{w}_flow{flow}_tsep{TSEP_LIST}_cfgs{len(CFG_LIST)}.h5"
        )
        path = INPUT_DIR / ratio_name

        if not path.exists():
            print(f"missing {path}", flush=True)
            return

        with h5py.File(path, "r") as f:
            ratio_jk = f["ratio_jk"][:].astype(np.complex128, copy=False)
            tsep_list = f["tsep_list"][:].astype(np.int64)

        if ratio_jk.shape[0] != len(CFG_LIST):
            raise ValueError(f"{path.name} has {ratio_jk.shape[0]} jackknife samples")
        if not np.array_equal(tsep_list, np.array(TSEP_LIST, dtype=np.int64)):
            raise ValueError(f"{path.name} tsep_list does not match TSEP_LIST")

        for itsep, tsep in enumerate(tsep_list):
            for tau in range(TAU_SKIP, int(tsep) - TAU_SKIP + 1):
                values = ratio_jk[:, itsep, tau]
                if np.all(np.isfinite(values.real)) and np.all(np.isfinite(values.imag)):
                    fit_values.append(values)
                    fit_w_index.append(iw)
                    fit_T.append(float(tsep))
                    fit_tau.append(float(tau))

    if not fit_values:
        print(f"skip no fit points: {operator}, {q_label}, {pf_label}, flow{flow}", flush=True)
        return

    y_jk = np.stack(fit_values, axis=1)
    x = {
        "w_index": np.array(fit_w_index, dtype=np.int64),
        "T": np.array(fit_T, dtype=np.float64),
        "tau": np.array(fit_tau, dtype=np.float64),
    }
    prior = gv.BufferDict()
    for name in ["M00", "Ai", "Af", "Afi"]:
        mean, sdev = PRIORS[name]
        prior[name] = gv.gvar(
            np.full(len(W_LIST), mean, dtype=np.float64),
            np.full(len(W_LIST), sdev, dtype=np.float64),
        )
    for name in ["dEi", "dEf"]:
        prior[name] = gv.gvar(*PRIORS[name])

    out_name = (
        f"{operator}_bareM_jk_{q_label}_{pf_label}_"
        f"flow{flow}_fit{FIT_TAG}_cfgs{len(CFG_LIST)}.h5"
    )
    out_path = OUTPUT_DIR / out_name

    if out_path.exists() and not OVERWRITE:
        print(f"skip existing {out_path}", flush=True)
        return

    print(f"fitting {operator}, {q_label}, {pf_label}, flow{flow}", flush=True)
    real_fit = fit_jackknife_samples(
        y_jk.real,
        x,
        prior,
        f"{operator} {q_label} {pf_label} flow{flow} real",
    )
    imag_fit = fit_jackknife_samples(
        y_jk.imag,
        x,
        prior,
        f"{operator} {q_label} {pf_label} flow{flow} imag",
    )

    real_params, real_chi2, real_dof, real_Q, real_status = real_fit
    imag_params, imag_chi2, imag_dof, imag_Q, imag_status = imag_fit
    bare_matrix_element_jk = real_params["M00"] + 1j * imag_params["M00"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("bare_matrix_element_jk", data=bare_matrix_element_jk.astype("<c16"))
        f.create_dataset("bare_matrix_element_jk_real", data=real_params["M00"].astype("<f8"))
        f.create_dataset("bare_matrix_element_jk_imag", data=imag_params["M00"].astype("<f8"))
        f.create_dataset("cfg_list", data=np.array(CFG_LIST, dtype=np.int64))
        f.create_dataset("q_code", data=np.array(q_tuple, dtype=np.int64))
        f.create_dataset("pf_code", data=np.array(pf_tuple, dtype=np.int64))
        f.create_dataset("w_list", data=np.array(W_LIST, dtype=np.int64))
        f.create_dataset("tsep_list", data=np.array(TSEP_LIST, dtype=np.int64))
        f.create_dataset("fit_w_index", data=np.array(fit_w_index, dtype=np.int64))
        f.create_dataset("fit_T", data=np.array(fit_T, dtype=np.int64))
        f.create_dataset("fit_tau", data=np.array(fit_tau, dtype=np.int64))

        fit_group = f.create_group("fit_params_jk")
        for part, params in (("real", real_params), ("imag", imag_params)):
            part_group = fit_group.create_group(part)
            for name, values in params.items():
                part_group.create_dataset(name, data=values.astype("<f8"))

        quality_group = f.create_group("fit_quality")
        for part, chi2, dof, q_val, status in (
            ("real", real_chi2, real_dof, real_Q, real_status),
            ("imag", imag_chi2, imag_dof, imag_Q, imag_status),
        ):
            part_group = quality_group.create_group(part)
            part_group.create_dataset("chi2", data=chi2.astype("<f8"))
            part_group.create_dataset("dof", data=dof)
            part_group.create_dataset("Q", data=q_val.astype("<f8"))
            part_group.create_dataset("status", data=status)

        f.attrs["operator"] = operator
        f.attrs["flow"] = flow
        f.attrs["matrix_element_dataset"] = "bare_matrix_element_jk"
        f.attrs["matrix_element_parameter"] = "M00"
        f.attrs["dim_bare_matrix_element_jk"] = "jackknife,w_index"
        f.attrs["dim_fit_w_dependent_params"] = "jackknife,w_index"
        f.attrs["jk_convention"] = "delete-one cfg jackknife, ordered by cfg_list"
        f.attrs["fit_model"] = MODEL_TEXT
        f.attrs["fit_part"] = "real_imag_separate"
        f.attrs["fit_strategy"] = "simultaneous over w,tsep,tau; shared dEi,dEf across w"
        f.attrs["fit_tag"] = FIT_TAG
        f.attrs["tau_skip"] = TAU_SKIP
        f.attrs["svdcut"] = SVD_CUT
        f.attrs["cov_eps"] = COV_EPS

        for name, value in PRIORS.items():
            f.attrs[f"prior_{name}"] = str(value)

    print(f"saved {out_path}", flush=True)


def main():
    tasks = []
    for operator in OPERATOR_LIST:
        for q_tuple in Q_LIST:
            for pf_tuple in PF_LIST:
                for flow in TGF_LIST:
                    tasks.append((operator, q_tuple, pf_tuple, flow))

    print(f"total simultaneous fits = {len(tasks)}, N_WORKERS = {N_WORKERS}", flush=True)
    print(f"TASK_BATCH_SIZE = {TASK_BATCH_SIZE}", flush=True)

    for batch_start in range(0, len(tasks), TASK_BATCH_SIZE):
        batch = tasks[batch_start : batch_start + TASK_BATCH_SIZE]
        batch_end = batch_start + len(batch)
        print(f"starting task batch {batch_start + 1}-{batch_end}/{len(tasks)}", flush=True)

        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {
                executor.submit(process_parameter_point, *task): task
                for task in batch
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    future.result()
                except Exception as err:
                    print(f"task failed {task}: {err}", flush=True)
                    raise


if __name__ == "__main__":
    main()
