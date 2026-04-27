import h5py
import numpy as np
from opt_einsum import contract
from types import SimpleNamespace

#==========================================================
#Assemble the 3pt function of gluon GPD

FFattrs = SimpleNamespace()
psink_list = list(range(-5,6))
q_list = []
for px in [0, -1, 1]:
    for py in [0, -1, 1]:
        for pz in [0, -1, 1]:
            q_list.append([px, py, pz])
munu_list = list(range(0,6))
rhosig_list = list(range(0,6))


with h5py.File(f"/lustre/orion/lgt132/scratch/sicheng/gluon_gpd_benchmark/Fmunu/complete_corr/FF_corr_complete_3.h5","r") as f_ff:
    for key, value in f_ff["corr"].attrs.items():
        setattr(FFattrs, key, value)

    for i_cfg, cfg in enumerate(list(range(1008,1020,12))):#3396
        ncfg = (cfg - 1008) // 12
        
        FF = f_ff["corr"][i_cfg,...] 
        
        pion_2pt_cache = {}
        FF_rolled_cache = {}
        
        for xsrc in range(0, 32, 4):
            shifted_x = (xsrc + ncfg*3) % 32
            with h5py.File(f"pion_2pt/incomplete/N40_G45_splitx/xsrc_{shifted_x}pion_smear40_mom0_G45_cfg{cfg}.h5", "r") as f_2pt:
                pion_2pt_cache[xsrc] = f_2pt["pion_45"][:]

        for tsrc in range(0, 96, 12):
            shifted_t = (tsrc + ncfg*3) % 32
            FF_rolled_cache[tsrc] = np.roll(FF, shift=-shifted_t, axis=5)
        
        for i_psink, psink in enumerate(psink_list):
            for i_munu, munu in enumerate(munu_list):
                for i_rhosig, rhosig in enumerate(rhosig_list):

                    pt3 = np.zeros((27,96,96,32,10), "<c16")

                    for i_xsrc, xsrc in enumerate(list(range(0,32,4))):
                        
                        pion_2pt = pion_2pt_cache[xsrc]

                        for i_tsrc, tsrc in enumerate(list(range(0,96,12))):
                            shifted_t = (tsrc + ncfg*3) % 32
                            
                            FFrolled = FF_rolled_cache[tsrc]
                            tmp = np.roll(pion_2pt, shift=-shifted_t, axis=-1)
                            print(FFrolled.shape)
                            print(tmp.shape)
                            for i_ysrc, ysrc in enumerate(list(range(0,32,8))):
                                for i_zsrc, zsrc in enumerate(list(range(0,32,8))):

                                    for i_q, q in enumerate(q_list):
                                        iqx = 1j * (q[0]*xsrc + q[1]*ysrc + q[2]*zsrc)
                                        
                                        pt3[i_q,:,:,:,:] += contract(
                                            "s,twq ->sqwt", 
                                            tmp[0,i_tsrc,0,i_ysrc,i_zsrc,i_psink,:], 
                                            FFrolled[i_munu,i_rhosig,:,:,i_q,:] * np.exp(-iqx) 
                                        )

                    pt3 = pt3 / (8*8*4*4)
                    save_path = f"/lustre/orion/lgt132/scratch/sicheng/gluon_gpd_benchmark/3pt_data/cfg{cfg}_psink{psink}_munu{munu}_rhosig{rhosig}200cfgs_3pt.h5"
                    with h5py.File(save_path, "w") as f_out:
                        dset = f_out.create_dataset("pt3", data=pt3)

                    print(f"cfg = {cfg}, psink = {psink}, munu = {munu}, rhosig = {rhosig} saved")
