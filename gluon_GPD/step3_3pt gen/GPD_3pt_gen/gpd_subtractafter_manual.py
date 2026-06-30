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
cfg_list = list(range(204, 1404, 6))
tsrc_list = list(range(0, 96, 12))
x_src_list = list(range(0,32,8))
y_src_list = list(range(0,32,8))
z_src_list = list(range(0,32,4))
T = 96
Ls = 32
tsep_max = 16
tins_max = 16
q_list = []
for qx in [0, -1, 1]:
    for qy in [0, -1, 1]:
        for qz in [0, -1, 1]:
            q_list.append([qx, qy, qz])
q_list = np.array(q_list, dtype=np.int64)
ff_input = (
    f"{data_dir}/FF_data/"
    "FF_fixed_smear_410_cfg{cfg}.h5"
)
pt2_input = (
    "{data_dir}/pion_2pt_functions/N40_G45_ez/"
    "pion_smear40_mom2p5_G45_ix{ix}_x{shifted_x}_cfg{cfg}.h5"
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


def run_one_cfg(cfg):
    ncfg = (cfg - 204) // 6
    print(cfg)
    
    #[x,y,z,q]
    phase = make_source_phase(cfg)
    
    with h5py.File(ff_input.format(data_dir=data_dir,cfg=cfg), "r") as f:
        # [src_lorentz, sink_lorentz, tgf, z_WL, i_qext, tFF]
        FF = f["corr"][0, ...]

    # [tsrc, ix, ysrc, zsrc, psink, t]
    pt2 = np.zeros((8, 4, 4, 8, 11, T), "<c16")
    for ix, xsrc in enumerate(range(0, 32, 8)):
        shifted_x = (xsrc + ncfg * 3) % 32
        path = pt2_input.format(data_dir=data_dir,ix=ix, shifted_x=shifted_x, cfg=cfg)

        with h5py.File(path, "r") as f:
            pt2[:, ix, :, :, :, :] = f["pion_45"][0, :, 0, :, :, :, :] #two 0's are cfg index and xsrc index


    FF_tsrc_sum = np.zeros((tins_max,6,6, 41, 10,27), "<c16")
    pt2_tsrc_sum = np.zeros((11, tsep_max), "<c16")
    pt2FF_tsrc = np.zeros((tsep_max, tins_max, 11,6,6, 41, 10,27), "<c16")

    for tsrc in tsrc_list:
        shifted_t = (tsrc + ncfg * 5) % T
        it = tsrc_list.index(tsrc)

        # [psink, tsep], source-at-origion and cut to tsep=0..15
        pt2_roll_full = np.roll(pt2[it, ...], -shifted_t, axis=-1)
        pt2_roll_cut = pt2_roll_full[..., 0:tsep_max]
        pt2_tsrc_sum += np.mean(pt2_roll_cut,axis=(0,1,2))
        
        # [src_lorentz, sink_lorentz, tgf, z_WL, q,t],  source-at-origion
        FF_roll = np.roll(FF, -shifted_t, axis=-1)[...,0:tins_max]
        # [tins, munu,rhosig, tgf, z_WL, q]
        FF_tsrc_sum += contract("q,abfwqi->iabfwq", np.mean(phase, axis=(0, 1, 2)), FF_roll)
        # [tsep, tins, psink, munu,rhosig, tgf, z_WL,q]
        pt2_phase_avg = contract("xyzq,xyzpt->qpt", phase, pt2_roll_cut) / (8 * 4 * 4)
        pt2FF_tsrc += contract(
            "qpt,abfwqi->tipabfwq",
            pt2_phase_avg,
            FF_roll,
            )
    FF_tsrc_avg = FF_tsrc_sum / len(tsrc_list)
    pt2_tsrc_avg = pt2_tsrc_sum / len(tsrc_list)
    pt2FF_tsrc_avg = pt2FF_tsrc / len(tsrc_list)

    with h5py.File(pt3_output.format(cfg=cfg), "w") as f_out:
        f_out.create_dataset("pt2FF", data=pt2FF_tsrc_avg)
        f_out.create_dataset("FF", data=FF_tsrc_avg)
        f_out.create_dataset("pt2", data=pt2_tsrc_avg)

        f_out.attrs["dim_CO"] = "tsep,tins,psink,munu,rhosig,tgf,z_WL,q"
        f_out.attrs["dim_O_tsrc_avg"] = "tins,munu,rhosig,tgf,z_WL,q"
        f_out.attrs["dim_C"] = "psink,t"
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
