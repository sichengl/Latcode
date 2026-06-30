import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import h5py
from opt_einsum import contract
data_dir = f"/lustre/orion/lgt132/scratch/sicheng/GPD_calc"
pt2_dir = f"/lustre/orion/lgt132/scratch/sicheng/GPD_calc_v3/2pt_production"
cfg_list = list(range(204, 234, 6))
tsrc_list = list(range(0, 96, 32))
x_src_list = list(range(0,32,16))
y_src_list = list(range(0,32,8))
z_src_list = list(range(0,32,8))
T = 96
Ls = 32
tsep_max = 16
tins_max = 16
pt2_rho = "5.0"
pt2_mom_frac = "0p35"
q_list = []
for qx in [0, -1, 1]:
    for qy in [0, -1, 1]:
        for qz in [0, -1, 1]:
            q_list.append([qx, qy, qz])
q_list = np.array(q_list, dtype=np.int64)
tgf_list = np.array([20, 25, 30, 35, 40], dtype=np.int64)
lorentz_list = np.array([(0, 0), (3, 3), (4, 4), (3, 0), (0, 3)], dtype=np.int64)
pf_list = np.array([[0, 0, pz] for pz in range(0, 6)], dtype=np.int64)
ff_input = (
    f"{data_dir}/FF_data/"
    "FF_fixed_smear_410_cfg{cfg}.h5"
)
pt2_input = (
    f"{pt2_dir}/"
    "N40_rho{rho}_G45_ez_momfrac{mom_frac}/"
    "pion_ix{ix}_x{shifted_x}_N40_rho{rho}_frac{mom_frac}_G45_cfg{cfg}.h5"
)
pt3_output = "./gpd_data/3pt_conn_inputs_man_cfg{cfg}.h5"

N_WORKERS = int(os.environ.get("N_WORKERS", "2"))
os.makedirs("./gpd_data", exist_ok=True)
def shifted_source_positions(cfg):
    ncfg = (cfg - 204) // 6
    shift = ncfg * 3
    x = np.array([(xsrc + shift) % Ls for xsrc in x_src_list], dtype=np.float64)
    y = np.array([(ysrc + shift) % Ls for ysrc in y_src_list], dtype=np.float64)
    z = np.array([(zsrc + shift) % Ls for zsrc in z_src_list], dtype=np.float64)
    return x, y, z


def make_source_phase(cfg):
    x, y, z = shifted_source_positions(cfg)
    qx = q_list[:, 0]
    qy = q_list[:, 1]
    qz = q_list[:, 2]
    q_dot_x = (
        x[:, None, None, None] * qx[None, None, None, :]
        + y[None, :, None, None] * qy[None, None, None, :]
        + z[None, None, :, None] * qz[None, None, None, :]
    )
    return np.exp(-1j * (2 * np.pi / Ls) * q_dot_x).astype("<c16")


def load_splitx_pt2(cfg):
    ncfg = (cfg - 204) // 6
    pt2_raw = None
    momentum_ref = None

    for ix, xsrc in enumerate(x_src_list):
        shifted_x = (xsrc + ncfg * 3) % Ls
        path = pt2_input.format(
            data_dir=data_dir,
            rho=pt2_rho,
            mom_frac=pt2_mom_frac,
            ix=ix,
            shifted_x=shifted_x,
            cfg=cfg,
        )

        with h5py.File(path, "r") as f:
            block = f["pion_45"][0, :, 0, :, :, :, :]
            momentum_list = f["momentum_list"][:]

        if pt2_raw is None:
            n_tsrc, n_y, n_z, n_mom, n_t = block.shape
            pt2_raw = np.zeros((n_tsrc, len(x_src_list), n_y, n_z, n_mom, n_t), "<c16")
            momentum_ref = momentum_list
        elif not np.array_equal(momentum_list, momentum_ref):
            raise ValueError(f"momentum_list mismatch in cfg {cfg}, ix {ix}")

        pt2_raw[:, ix, :, :, :, :] = block

    return pt2_raw, momentum_ref


def build_momentum_indices(momentum_list):
    mom_to_idx = {tuple(p): i for i, p in enumerate(momentum_list.tolist())}

    pf_indices = np.zeros(len(pf_list), dtype=np.int64)
    for ipf, pf in enumerate(pf_list):
        pf_tuple = tuple(pf.tolist())
        if pf_tuple not in mom_to_idx:
            raise ValueError(f"Missing final momentum pf={pf_tuple}")
        pf_indices[ipf] = mom_to_idx[pf_tuple]

    pi_indices = np.zeros((len(pf_list), len(q_list)), dtype=np.int64)
    for ipf, pf in enumerate(pf_list):
        for iq, q in enumerate(q_list):
            pi_tuple = tuple((pf + q).tolist())
            if pi_tuple not in mom_to_idx:
                raise ValueError(
                    f"Missing initial momentum pi={pi_tuple} "
                    f"for pf={tuple(pf.tolist())}, q={tuple(q.tolist())}"
                )
            pi_indices[ipf, iq] = mom_to_idx[pi_tuple]

    return pf_indices, pi_indices


def run_one_cfg(cfg):
    ncfg = (cfg - 204) // 6
    print(cfg)
    
    #[x,y,z,q]
    phase = make_source_phase(cfg)
    
    with h5py.File(ff_input.format(data_dir=data_dir,cfg=cfg), "r") as f:
        # [src_lorentz, sink_lorentz, tgf, z_WL, i_qext, tFF]
        FF_raw = f["corr"][0, ...]
        FF = np.stack(
            [FF_raw[i, j, tgf_list, :, :, :] for i, j in lorentz_list],
            axis=0,
        )

    # [tsrc, ix, ysrc, zsrc, momentum, t]
    pt2, momentum_list = load_splitx_pt2(cfg)
    pf_indices, pi_indices = build_momentum_indices(momentum_list)
    n_spatial_src = len(x_src_list) * len(y_src_list) * len(z_src_list)

    FF_tsrc_sum = np.zeros((tins_max, len(lorentz_list), len(tgf_list), 10, len(q_list)), "<c16")
    pt2_mom_tsrc_sum = np.zeros((len(momentum_list), tsep_max), "<c16")
    pt2FF_tsrc = np.zeros(
        (tsep_max, tins_max, len(pf_list), len(lorentz_list), len(tgf_list), 10, len(q_list)),
        "<c16",
    )

    for tsrc in tsrc_list:
        shifted_t = (tsrc + ncfg * 5) % T
        it = tsrc_list.index(tsrc)

        # [ix, ysrc, zsrc, momentum, tsep], source-at-origin and cut to tsep=0..15
        pt2_roll_full = np.roll(pt2[it, ...], -shifted_t, axis=-1)
        pt2_roll_cut = pt2_roll_full[..., 0:tsep_max]
        pt2_mom_tsrc_sum += np.mean(pt2_roll_cut, axis=(0, 1, 2))
        
        # [lorentz, tgf, z_WL, q, tins], source-at-origin
        FF_roll = np.roll(FF, -shifted_t, axis=-1)[...,0:tins_max]
        # [tins, lorentz, tgf, z_WL, q]
        FF_tsrc_sum += contract("q,lfwqi->ilfwq", np.mean(phase, axis=(0, 1, 2)), FF_roll)

        # [ix, ysrc, zsrc, pf, tsep]
        pt2_pf_roll_cut = pt2_roll_cut[..., pf_indices, :]
        # [q, pf, tsep]
        pt2_phase_avg = contract("xyzq,xyzpt->qpt", phase, pt2_pf_roll_cut) / n_spatial_src
        # [tsep, tins, pf, lorentz, tgf, z_WL, q]
        pt2FF_tsrc += contract(
            "qpt,lfwqi->tiplfwq",
            pt2_phase_avg,
            FF_roll,
            )
    FF_tsrc_avg = FF_tsrc_sum / len(tsrc_list)
    pt2_mom_tsrc_avg = pt2_mom_tsrc_sum / len(tsrc_list)
    pt2_f_tsrc_avg = pt2_mom_tsrc_avg[pf_indices, :]
    pt2_i_tsrc_avg = pt2_mom_tsrc_avg[pi_indices, :]
    pt2FF_tsrc_avg = pt2FF_tsrc / len(tsrc_list)

    with h5py.File(pt3_output.format(cfg=cfg), "w") as f_out:
        f_out.create_dataset("pt2FF", data=pt2FF_tsrc_avg)
        f_out.create_dataset("FF", data=FF_tsrc_avg)
        f_out.create_dataset("pt2_mom", data=pt2_mom_tsrc_avg)
        f_out.create_dataset("pt2_f", data=pt2_f_tsrc_avg)
        f_out.create_dataset("pt2_i", data=pt2_i_tsrc_avg)
        f_out.create_dataset("momentum_list", data=momentum_list)
        f_out.create_dataset("pf_list", data=pf_list)
        f_out.create_dataset("q_list", data=q_list)
        f_out.create_dataset("lorentz_list", data=lorentz_list)
        f_out.create_dataset("tgf_list", data=tgf_list)

        f_out.attrs["dim_pt2FF"] = "tsep,tins,pf,lorentz,tgf,z_WL,q"
        f_out.attrs["dim_FF"] = "tins,lorentz,tgf,z_WL,q"
        f_out.attrs["dim_pt2_mom"] = "momentum,t"
        f_out.attrs["dim_pt2_f"] = "pf,t"
        f_out.attrs["dim_pt2_i"] = "pf,q,t"
        f_out.attrs["q_convention"] = "FF q uses exp(+i q x); code-momentum pi = pf + q"
        f_out.attrs["source_phase_convention"] = "source averaging phase = exp(-i q x_src)"
        f_out.attrs["physical_momentum_convention"] = "C2 uses exp(+i p x), so P_phys = -p_code and Delta_phys = -q"
        f_out.attrs["vacuum_subtraction"] = ("Done afterwards in jackknife")
    print(f"done cfg {cfg}", flush=True)
    return cfg

if __name__ == "__main__":
    tasks = list(cfg_list)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(run_one_cfg, cfg) for cfg in tasks]

        for future in as_completed(futures):
            cfg = future.result()
            print(f"finished task cfg {cfg}", flush=True)

