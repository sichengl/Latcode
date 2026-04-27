import sys
sys.path.append('/ccs/home/sicheng/LaMETLat')
import lametlat.utils.resampling as resamp
import h5py
import numpy as np
import matplotlib.pyplot as plt
import gvar as gv
from lametlat.gsfit.pt2_fit import pt2_two_state_fit, plot_2pt_fit_on_meff

gamma=5
N=40
#raw = np.real(h5py.File(f"pion_large_mom_smear{N}_hyp_corrected_gamma{gm}.h5", "r")[f"pion_{gm}"][:])
#t_src_list = h5py.File(f"pion_large_mom_smear{N}_hyp_corrected_gamma{gm}.h5", "r")[f"pion_{gm}"].attrs["t_src_list"]
raw = np.real(h5py.File(f"/lustre/orion/lgt132/scratch/sicheng/gluon_gpd_benchmark/pion_2pt/complete/N{N}_G{gamma}_complete/N{N}G{gamma}MOM2p5_complete_200.h5", "r")[f"pion_{gamma}"][:])

t_src_list = list(range(0,96,32))
for t_idx, t_src in enumerate(t_src_list):
    raw[:,t_idx,:,:,:,:,:] = np.roll(raw[:,t_idx,:,:,:,:,:],-t_src,-1)

raw = np.mean(raw,axis=(1,2,3,4))

jacked = resamp.jackknife(raw)
#print(jacked.shape)
pt2_N40_gamma5 = resamp.jk_ls_avg(jacked[:,20,:])
#print(gv.mean(pt2_N40_gamma5))

tmin = 1
tmax = 25
Lt = 96

px = py = pz = 0
b = 0
z = 0
err_t_ls = np.arange(32)
fill_t_ls = np.arange(tmin,tmax)
id_label = {"px": px, "py": py, "pz": pz, "b": b, "z": z}
pt2_fit_res = pt2_two_state_fit(pt2_N40_gamma5, tmin, tmax, Lt, normalize=True, label="Test")
print(pt2_fit_res)

fig, ax = plot_2pt_fit_on_meff(pt2_N40_gamma5[err_t_ls], pt2_fit_res, err_t_ls, fill_t_ls, id_label, Lt)



