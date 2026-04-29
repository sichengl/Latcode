import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import h5py
import numpy as np
from opt_einsum import contract
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor
import itertools

FFattrs = SimpleNamespace()
psink_list = list(range(-5, 6))

q_list = []
for px in [0, -1, 1]:
    for py in [0, -1, 1]:
        for pz in [0, -1, 1]:
            q_list.append([px, py, pz])

munu_list = list(range(0, 6))
rhosig_list = list(range(0, 6))

task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

all_cfgs = list(range(1008, 3396, 12))

if task_id >= len(all_cfgs):
    print(f"Task ID {task_id} exceeds the number of available cfgs. Exiting.")
    sys.exit(0)

cfg = all_cfgs[task_id]
ncfg = (cfg - 1008) // 12
print(f"--- Node started: Processing cfg = {cfg} (ncfg = {ncfg}) ---")

pion_2pt_cache = {}
FF_rolled_cache = {}

print("Loading FF correlation data into memory...")
ff_path = "/gpfs/scratch/sicheliu/FF_corr_complete_3.h5"
with h5py.File(ff_path, "r") as f_ff:
    for key, value in f_ff["corr"].attrs.items():
        setattr(FFattrs, key, value)
    
    i_cfg = task_id 
    FF = f_ff["corr"][i_cfg, ...]
    FF_avg = np.mean(f_ff["corr"][:],axis=0)
    FF = FF-FF_avg
print("Loading Pion 2pt data and building caches...")
for xsrc in range(0, 32, 4):
    shifted_x = (xsrc + ncfg*3) % 32
    p2pt_path = f"/gpfs/scratch/sicheliu/N40_G45_splitx/xsrc_{shifted_x}pion_smear40_mom0_G45_cfg{cfg}.h5"
    with h5py.File(p2pt_path, "r") as f_2pt:
        pion_2pt_cache[xsrc] = f_2pt["pion_45"][:]

for tsrc in range(0, 96, 12):
    shifted_t = (tsrc + ncfg*3) % 96
    FF_rolled_cache[tsrc] = np.roll(FF, shift=-shifted_t, axis=5)

print("Data loading complete. Starting parallel computation...")

def compute_single_block(task_args):
    psink, munu, rhosig = task_args
    pt3 = np.zeros((27, 8, 16, 32, 10), "<c16")

    for i_xsrc, xsrc in enumerate(list(range(0, 32, 4))):
        pion_2pt = pion_2pt_cache[xsrc]

        for i_tsrc, tsrc in enumerate(list(range(0, 96, 12))):
            shifted_t = (tsrc + ncfg*3) % 96
            FFrolled = FF_rolled_cache[tsrc][...,0:16]
            tmp = np.roll(pion_2pt, shift=-shifted_t, axis=-1)[...,8:16]

            for i_ysrc, ysrc in enumerate(list(range(0, 32, 8))):
                for i_zsrc, zsrc in enumerate(list(range(0, 32, 8))):
                    for i_q, q in enumerate(q_list):
                        
                        shifted_x = (xsrc+ncfg*3)%32
                        shifted_y = (ysrc+ncfg*3)%32
                        shifted_z = (zsrc+ncfg*3)%32
                        iqx = 1j *(2 * np.pi / 32)* (q[0]*shifted_x + q[1]*shifted_y + q[2]*shifted_z)
                        
                        pt3[i_q, :, :, :, :] += contract(
                            "s,twq ->sqwt", 
                            tmp[0, i_tsrc, 0, i_ysrc, i_zsrc, psink_list.index(psink), :], 
                            FFrolled[munu, rhosig, :, :, i_q, :] * np.exp(-iqx) 
                        )

    pt3 = pt3 / (8 * 8 * 4 * 4)
                                           
    save_dir = "/gpfs/scratch/sicheliu/3pt_data_background/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"cfg{cfg}_psink{psink}_munu{munu}_rhosig{rhosig}200cfgs_3pt.h5")
    
    with h5py.File(save_path, "w") as f_out:
        f_out.create_dataset("pt3", data=pt3)

    return f"[Success] cfg={cfg} | psink={psink:2d}, munu={munu}, rhosig={rhosig}"

if __name__ == "__main__":
    tasks = [(psink, munu, rhosig) for psink, munu, rhosig in itertools.product(psink_list, munu_list, rhosig_list) if psink >= 0 and munu == rhosig]
    NUM_WORKERS = 90
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        for result in pool.map(compute_single_block, tasks):
            print(result)
            
    print(f"--- Configuration {cfg} totally finished! ---")
