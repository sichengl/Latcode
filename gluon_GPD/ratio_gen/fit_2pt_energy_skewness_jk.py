import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import re

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import gvar as gv
import h5py
import lsqfit
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "gpd_data"
OUTPUT_PATH = SCRIPT_DIR / "two_point_energy_skewness_jk.h5"

# Set CFG_LIST to None to use every matching file in INPUT_DIR.
CFG_LIST = None
# CFG_LIST = list(range(204, 204 + 200 * 6, 6))

N_STATE = 2
T_FIT_MIN = 3
T_FIT_MAX = 12

LS = 32
LT = 96

# Use -1.0 for p_phys = -p_code. Change to +1.0 if you choose the opposite convention.
MOMENTUM_SIGN = -1.0

FIT_ONLY_USED_MOMENTA = True
USE_REAL_PART = True
USE_PERIODIC_MODEL = False
USE_CORRELATED_FIT = True
N_WORKERS = int(os.environ.get("N_WORKERS", "4"))

SVD_CUT = 1e-8
COV_REGULATOR = 1e-12
MAXIT = 10000

LOG_E0_WIDTH = 1.0
DELTA_E_PRIOR_CENTER = 0.5
LOG_DELTA_E_WIDTH = 1.0
AMP_PRIOR_WIDTH_FACTOR = 10.0


def make_prior(n_state, p_code, amp_scale, e0_center):
    prior = gv.BufferDict()
    prior["log_E0"] = gv.gvar(np.log(e0_center), LOG_E0_WIDTH)

    for istate in range(1, n_state):
        prior[f"log_dE{istate}"] = gv.gvar(
            np.log(DELTA_E_PRIOR_CENTER),
            LOG_DELTA_E_WIDTH,
        )

    prior["amp0"] = gv.gvar(amp_scale, AMP_PRIOR_WIDTH_FACTOR * abs(amp_scale))
    for istate in range(1, n_state):
        prior[f"amp{istate}"] = gv.gvar(0.0, AMP_PRIOR_WIDTH_FACTOR * abs(amp_scale))

    return prior


def fit_function(t, p):
    t = np.asarray(t, dtype=np.float64)
    c2 = 0.0
    energy = gv.exp(p["log_E0"])

    for istate in range(N_STATE):
        if istate > 0:
            energy = energy + gv.exp(p[f"log_dE{istate}"])

        term = p[f"amp{istate}"] * gv.exp(-energy * t)
        if USE_PERIODIC_MODEL:
            term = term + p[f"amp{istate}"] * gv.exp(-energy * (LT - t))
        c2 = c2 + term

    return c2


def fit_one_momentum(task):
    count, total, imom, p_code, y_jk_all, t_fit, n_jk = task

    energy_col = np.full(n_jk, np.nan, dtype=np.float64)
    chi2_col = np.full(n_jk, np.nan, dtype=np.float64)
    q_col = np.full(n_jk, np.nan, dtype=np.float64)
    success_col = np.zeros(n_jk, dtype=np.int8)

    y_mean = np.mean(y_jk_all, axis=0)
    y_diff = y_jk_all - y_mean[None, :]
    cov = (n_jk - 1.0) / n_jk * (y_diff.T @ y_diff)

    if not USE_CORRELATED_FIT:
        cov = np.diag(np.diag(cov))

    diag = np.diag(cov).copy()
    positive_diag = diag[diag > 0]
    regulator_scale = np.max(positive_diag) if positive_diag.size else 1.0
    cov = cov + COV_REGULATOR * regulator_scale * np.eye(len(t_fit))

    amp_scale = float(y_mean[0])
    if not np.isfinite(amp_scale) or abs(amp_scale) < 1e-14:
        amp_scale = 1.0

    if y_mean[0] > 0 and y_mean[1] > 0:
        e0_center = float(np.log(y_mean[0] / y_mean[1]))
    else:
        e0_center = 0.3
    e0_center = float(np.clip(e0_center, 0.03, 2.0))

    prior = make_prior(N_STATE, p_code, amp_scale, e0_center)
    p0 = None
    mean_fit_error = ""

    try:
        y_gv_mean = gv.gvar(y_mean, cov)
        mean_fit = lsqfit.nonlinear_fit(
            data=(t_fit, y_gv_mean),
            prior=prior,
            fcn=fit_function,
            svdcut=SVD_CUT,
            maxit=MAXIT,
        )
        p0 = {key: gv.mean(mean_fit.p[key]) for key in prior}
    except Exception as err:
        mean_fit_error = str(err)

    first_jk_errors = []
    for ijk in range(n_jk):
        try:
            y_gv = gv.gvar(y_jk_all[ijk], cov)
            fit = lsqfit.nonlinear_fit(
                data=(t_fit, y_gv),
                prior=prior,
                fcn=fit_function,
                p0=p0,
                svdcut=SVD_CUT,
                maxit=MAXIT,
            )

            e0 = gv.exp(fit.p["log_E0"])
            energy_col[ijk] = gv.mean(e0)
            chi2_col[ijk] = fit.chi2 / fit.dof if fit.dof > 0 else np.nan
            q_col[ijk] = fit.Q
            success_col[ijk] = 1
        except Exception as err:
            if len(first_jk_errors) < 3:
                first_jk_errors.append(f"jk={ijk}: {err}")

    n_success = int(np.sum(success_col))
    print(
        f"fit momentum {count}/{total} {tuple(p_code.tolist())}: "
        f"success {n_success}/{n_jk}",
        flush=True,
    )
    if mean_fit_error:
        print(
            f"mean fit failed for momentum {tuple(p_code.tolist())}: {mean_fit_error}",
            flush=True,
        )
    if first_jk_errors:
        print(
            f"first jk fit failures for momentum {tuple(p_code.tolist())}: "
            + " | ".join(first_jk_errors),
            flush=True,
        )

    return imom, energy_col, chi2_col, q_col, success_col


if __name__ == "__main__":
    all_paths = sorted(INPUT_DIR.glob("3pt_conn_inputs_man_cfg*.h5"))
    if not all_paths:
        raise FileNotFoundError(f"No 3pt_conn_inputs_man_cfg*.h5 files found in {INPUT_DIR}")

    cfg_to_path = {}
    for path in all_paths:
        match = re.search(r"cfg(\d+)", path.name)
        if match is None:
            continue
        cfg_to_path[int(match.group(1))] = path

    if CFG_LIST is None:
        cfg_list = np.array(sorted(cfg_to_path), dtype=np.int64)
    else:
        cfg_list = np.array(CFG_LIST, dtype=np.int64)
        missing_cfgs = [int(cfg) for cfg in cfg_list if int(cfg) not in cfg_to_path]
        if missing_cfgs:
            raise FileNotFoundError(f"Missing cfg files: {missing_cfgs[:10]}")

    if len(cfg_list) < 2:
        raise ValueError("Need at least two cfgs to build jackknife samples")

    c2_cfg = []
    momentum_ref = None
    pf_ref = None
    q_ref = None

    for cfg in cfg_list:
        path = cfg_to_path[int(cfg)]
        with h5py.File(path, "r") as f:
            pt2_mom = f["pt2_mom"][:].astype(np.complex128, copy=False)
            momentum_list = f["momentum_list"][:].astype(np.int64)
            pf_list = f["pf_list"][:].astype(np.int64)
            q_list = f["q_list"][:].astype(np.int64)

        if momentum_ref is None:
            momentum_ref = momentum_list
            pf_ref = pf_list
            q_ref = q_list
        else:
            if not np.array_equal(momentum_ref, momentum_list):
                raise ValueError(f"{path.name} momentum_list differs from previous cfg")
            if not np.array_equal(pf_ref, pf_list):
                raise ValueError(f"{path.name} pf_list differs from previous cfg")
            if not np.array_equal(q_ref, q_list):
                raise ValueError(f"{path.name} q_list differs from previous cfg")

        c2_cfg.append(pt2_mom)

    c2_cfg = np.stack(c2_cfg, axis=0)
    n_cfg, n_mom, n_t = c2_cfg.shape
    n_jk = n_cfg

    if T_FIT_MIN < 0 or T_FIT_MAX >= n_t or T_FIT_MIN >= T_FIT_MAX:
        raise ValueError(f"Bad fit window [{T_FIT_MIN}, {T_FIT_MAX}] for n_t={n_t}")

    t_fit = np.arange(T_FIT_MIN, T_FIT_MAX + 1, dtype=np.float64)
    mom_to_index = {tuple(mom): imom for imom, mom in enumerate(momentum_ref.tolist())}

    used_momentum_indices = set()
    if FIT_ONLY_USED_MOMENTA:
        for pf_code in pf_ref:
            pf_tuple = tuple(int(x) for x in pf_code)
            if pf_tuple not in mom_to_index:
                raise ValueError(f"Missing pf momentum {pf_tuple} in momentum_list")
            used_momentum_indices.add(mom_to_index[pf_tuple])

            for q_code in q_ref:
                pi_tuple = tuple(int(x) for x in (pf_code + q_code))
                if pi_tuple not in mom_to_index:
                    raise ValueError(f"Missing pi momentum {pi_tuple} in momentum_list")
                used_momentum_indices.add(mom_to_index[pi_tuple])
    else:
        used_momentum_indices = set(range(n_mom))

    used_momentum_indices = np.array(sorted(used_momentum_indices), dtype=np.int64)

    c2_sum = np.sum(c2_cfg, axis=0)
    c2_jk = (c2_sum[None, :, :] - c2_cfg) / (n_cfg - 1.0)

    energy_jk = np.full((n_jk, n_mom), np.nan, dtype=np.float64)
    fit_chi2_dof = np.full((n_jk, n_mom), np.nan, dtype=np.float64)
    fit_Q = np.full((n_jk, n_mom), np.nan, dtype=np.float64)
    fit_success = np.zeros((n_jk, n_mom), dtype=np.int8)

    fit_tasks = []
    for count, imom in enumerate(used_momentum_indices, start=1):
        p_code = momentum_ref[imom]
        y_jk_all = c2_jk[:, imom, T_FIT_MIN : T_FIT_MAX + 1]
        if USE_REAL_PART:
            y_jk_all = y_jk_all.real
        else:
            y_jk_all = np.abs(y_jk_all)

        fit_tasks.append(
            (
                count,
                len(used_momentum_indices),
                int(imom),
                p_code.copy(),
                np.ascontiguousarray(y_jk_all, dtype=np.float64),
                t_fit.copy(),
                n_jk,
            )
        )

    print(
        f"fitting {len(fit_tasks)} momenta with N_WORKERS={N_WORKERS}",
        flush=True,
    )

    if N_WORKERS <= 1:
        for task in fit_tasks:
            imom, energy_col, chi2_col, q_col, success_col = fit_one_momentum(task)
            energy_jk[:, imom] = energy_col
            fit_chi2_dof[:, imom] = chi2_col
            fit_Q[:, imom] = q_col
            fit_success[:, imom] = success_col
    else:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = [executor.submit(fit_one_momentum, task) for task in fit_tasks]
            for future in as_completed(futures):
                imom, energy_col, chi2_col, q_col, success_col = future.result()
                energy_jk[:, imom] = energy_col
                fit_chi2_dof[:, imom] = chi2_col
                fit_Q[:, imom] = q_col
                fit_success[:, imom] = success_col

    energy_mean = np.nanmean(energy_jk, axis=0)
    energy_diff = energy_jk - energy_mean[None, :]
    energy_sdev = np.sqrt((n_jk - 1.0) / n_jk * np.nansum(energy_diff**2, axis=0))

    n_q = len(q_ref)
    n_pf = len(pf_ref)
    momentum_unit = 2.0 * np.pi / LS

    pi_list = np.zeros((n_q, n_pf, 3), dtype=np.int64)
    xi_jk = np.full((n_jk, n_q, n_pf), np.nan, dtype=np.float64)
    quasi_xi_jk = np.full((n_jk, n_q, n_pf), np.nan, dtype=np.float64)
    pf_momentum_indices = np.full(n_pf, -1, dtype=np.int64)
    pi_momentum_indices = np.full((n_q, n_pf), -1, dtype=np.int64)

    for ipf, pf_code in enumerate(pf_ref):
        pf_tuple = tuple(int(x) for x in pf_code)
        pf_index = mom_to_index[pf_tuple]
        pf_momentum_indices[ipf] = pf_index
        pf_z = MOMENTUM_SIGN * momentum_unit * float(pf_code[2])

        for iq, q_code in enumerate(q_ref):
            pi_code = pf_code + q_code
            pi_tuple = tuple(int(x) for x in pi_code)
            pi_index = mom_to_index[pi_tuple]
            pi_list[iq, ipf, :] = pi_code
            pi_momentum_indices[iq, ipf] = pi_index

            pi_z = MOMENTUM_SIGN * momentum_unit * float(pi_code[2])
            e_f = energy_jk[:, pf_index]
            e_i = energy_jk[:, pi_index]

            pf_plus = e_f + pf_z
            pi_plus = e_i + pi_z
            xi_jk[:, iq, ipf] = -((pf_plus - pi_plus) / (pf_plus + pi_plus))

            quasi_den = pf_z + pi_z
            if abs(quasi_den) > 1e-14:
                quasi_value = -((pf_z - pi_z) / quasi_den)
                quasi_xi_jk[:, iq, ipf] = quasi_value

    xi_mean = np.nanmean(xi_jk, axis=0)
    xi_diff = xi_jk - xi_mean[None, :, :]
    xi_sdev = np.sqrt((n_jk - 1.0) / n_jk * np.nansum(xi_diff**2, axis=0))

    quasi_xi_mean = np.nanmean(quasi_xi_jk, axis=0)
    quasi_xi_diff = quasi_xi_jk - quasi_xi_mean[None, :, :]
    quasi_xi_sdev = np.sqrt((n_jk - 1.0) / n_jk * np.nansum(quasi_xi_diff**2, axis=0))

    with h5py.File(OUTPUT_PATH, "w") as f:
        f.create_dataset("cfg_list", data=cfg_list.astype(np.int64))
        f.create_dataset("momentum_list", data=momentum_ref.astype(np.int64))
        f.create_dataset("pf_list", data=pf_ref.astype(np.int64))
        f.create_dataset("q_list", data=q_ref.astype(np.int64))
        f.create_dataset("pi_list", data=pi_list.astype(np.int64))
        f.create_dataset("pf_momentum_indices", data=pf_momentum_indices)
        f.create_dataset("pi_momentum_indices", data=pi_momentum_indices)
        f.create_dataset("fitted_momentum_indices", data=used_momentum_indices)
        f.create_dataset("t_fit", data=t_fit.astype(np.int64))

        f.create_dataset("energy_jk", data=energy_jk.astype("<f8"))
        f.create_dataset("energy_mean", data=energy_mean.astype("<f8"))
        f.create_dataset("energy_sdev", data=energy_sdev.astype("<f8"))
        f.create_dataset("fit_success", data=fit_success)
        f.create_dataset("fit_chi2_dof", data=fit_chi2_dof.astype("<f8"))
        f.create_dataset("fit_Q", data=fit_Q.astype("<f8"))

        f.create_dataset("xi_jk", data=xi_jk.astype("<f8"))
        f.create_dataset("xi_mean", data=xi_mean.astype("<f8"))
        f.create_dataset("xi_sdev", data=xi_sdev.astype("<f8"))
        f.create_dataset("quasi_xi_jk", data=quasi_xi_jk.astype("<f8"))
        f.create_dataset("quasi_xi_mean", data=quasi_xi_mean.astype("<f8"))
        f.create_dataset("quasi_xi_sdev", data=quasi_xi_sdev.astype("<f8"))

        f.attrs["input_dir"] = str(INPUT_DIR)
        f.attrs["n_state"] = N_STATE
        f.attrs["fit_model"] = "sum_n amp_n * exp(-E_n*t), E_n=E_0+positive gaps"
        f.attrs["use_periodic_model"] = int(USE_PERIODIC_MODEL)
        f.attrs["use_real_part"] = int(USE_REAL_PART)
        f.attrs["use_correlated_fit"] = int(USE_CORRELATED_FIT)
        f.attrs["n_workers"] = N_WORKERS
        f.attrs["fit_parallelization"] = "momentum"
        f.attrs["svd_cut"] = SVD_CUT
        f.attrs["cov_regulator"] = COV_REGULATOR
        f.attrs["momentum_sign"] = MOMENTUM_SIGN
        f.attrs["momentum_unit"] = momentum_unit
        f.attrs["momentum_convention"] = "p_z = momentum_sign * (2*pi/Ls) * p_code_z"
        f.attrs["xi_definition"] = "-(p_f^+ - p_i^+) / (p_f^+ + p_i^+), p^+=E+p_z"
        f.attrs["quasi_xi_definition"] = "-(p_fz - p_iz) / (p_fz + p_iz)"
        f.attrs["dim_energy_jk"] = "jackknife,momentum_index"
        f.attrs["dim_xi_jk"] = "jackknife,q_index,pf_index"
        f.attrs["dim_quasi_xi_jk"] = "jackknife,q_index,pf_index"
        f.attrs["q_convention"] = "pi_code = pf_code + q_code"

    print(f"saved {OUTPUT_PATH}", flush=True)

