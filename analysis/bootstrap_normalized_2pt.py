import sys
sys.path.append('/ccs/home/sicheng/Correlation_Analysis')
import module.resampling as resamp
import h5py
import numpy as np
import matplotlib.pyplot as plt
import gvar as gv

p=3
#rho = [4.961,6.076,7.845]
#N=[40,60,100]
rho = [4.961,6.076,7.845]
N = [40,60,100]
tstart = 10
tend = 20
fm_inverse = 197.3
rsqr = []
px_list = list(range(-20,21))
for i_n, n in enumerate(list(N)):
    
    with h5py.File(f"N{n}complete45.h5", "r") as f:

        dset45 = f["pion_45"]
        pion_45 = dset45[:]
    with h5py.File(f"N{n}complete55.h5", "r") as f:

        dset55 = f["pion_55"]
        pion_55 = dset55[:]
        if n==40:
            t_src_list = dset55.attrs["t_src_list"]

    pion_45 = np.mean(pion_45, axis=(2,3,4))
    pion_55 = np.mean(pion_55, axis=(2,3,4))
    
    for t_idx, t_src in enumerate(t_src_list):
        pion_45[:,t_idx,:,:] = np.roll(pion_45[:,t_idx,:,:],-t_src,-1)
        pion_55[:,t_idx,:,:] = np.roll(pion_55[:,t_idx,:,:],-t_src,-1)
    
    pion_45 = np.mean(pion_45,axis=1)
    pion_55 = np.mean(pion_55,axis=1)
    
    corr_bs_45 = resamp.bootstrap(pion_45, 128)
    corr_bs_55 = resamp.bootstrap(pion_55, 128)
    
    meff_45 = np.abs(corr_bs_45[:,:, :] / corr_bs_45[:,:, 0][:,:,None])
    meff_55 = np.abs(corr_bs_55[:,:, :] / corr_bs_55[:,:, 0][:,:,None])

    for i_px, px in enumerate(px_list):
        if px not in [p]:
            continue
        meff_avg_45 = resamp.bs_ls_avg(meff_45[:,i_px,:])
        meff_avg_55 = resamp.bs_ls_avg(meff_55[:,i_px,:])

        if i_n == 0:
            norm = gv.mean(meff_avg_55)
        meff_avg_45 = meff_avg_45[:]/norm
        meff_avg_55 = meff_avg_55[:]/norm

        plt.errorbar(
                np.arange(len(meff_avg_45)),
                gv.mean(meff_avg_45),
                gv.sdev(meff_avg_45),
                fmt="o",ms=6,
                capsize=4,
                label=f"gamma=45, N={n}, P=[{px},0,0]",
            )
        plt.errorbar(
                np.arange(len(meff_avg_55)),
                gv.mean(meff_avg_55),
                gv.sdev(meff_avg_55),
                fmt="x",ms=6,
                capsize=4,
                label=f"gamma=5, N={n}, P=[{px},0,0]",
            )

#plt.ylim([0, 3])
plt.xlim([0, 15])
plt.legend()
plt.title(f"bootstrap 2pt N={N} GAMMA45&GAMMA5 P=[{p},0,0]")
plt.xlabel("time")
plt.ylabel("Normalized 2pt")
plt.show() #? bootstrap gives a similar plot to jackknife, as we expected

