import numpy as np
import h5py
from opt_einsum import contract


cfg_list = list(range(204,1404,6))
FF = np.zeros((6,6,41,10,27,96),"<c16")
with h5py.File("/lustre/orion/lgt132/scratch/sicheng/GPD_calc/FF_avg_tsrcroll.h5","r") as f_avg:
            # [munu, rhosig, tgf, z_WL, i_qext, tFF]
            FF_avg = f_avg["FF_avg"][...]
#FF_avg = np.mean(FF_avg,axis=-1)
pt2=np.zeros((8,4,4,8,11,96),"<c16")
               #psink,lorentz
pt3=np.zeros((16,96,11,6,41,10),"<c16")
pt3_counter=np.zeros((7),"<c16")
for icfg, cfg in enumerate(cfg_list):
    print(cfg)
    with h5py.File(f"./FF_data/FF_opp_smear_410_cfg{cfg}.h5","r") as f:
        dset = f["corr"][0,...]
        #src_lorentz,sink_lorentz,41,10,27,96
        FF  = (dset - FF_avg[...])
       
    ncfg = (cfg-204)//6
    for ix,xsrc in enumerate(range(0,32,8)):

        shifted_x = (xsrc+ncfg*3)%32

        with h5py.File(f"./pion_2pt_functions/N40_G45_ez/pion_smear40_mom2p5_G45_ix{ix}_x{shifted_x}_cfg{cfg}.h5","r") as f:

            tmp_2pt = f["pion_45"][0,:,0,:,:,:,:]

        pt2[:,ix,:,:,:,:] = tmp_2pt

    pt2_src_avg = np.mean(pt2,axis=(1,2,3))
    
    #now average over tsrc for both 2pt and FF
    pt2_tsrc_avg_tmp = np.zeros((8,11,16),"<c16")
    ff_tsrc_avg_tmp = np.zeros((8,6,6,41,10,27,96),"<c16")
    for it, tsrc in enumerate(range(0,96,12)):

        shifted_t = (tsrc+ncfg*5)%96
        pt2_tsrc_avg_tmp[it,...] = np.roll(pt2_src_avg[it,...],-shifted_t,axis=-1)[...,0:16]
        ff_tsrc_avg_tmp[it,...] = np.roll(FF,-shifted_t,axis=-1)
    pt2_tsrc_avg = np.mean(pt2_tsrc_avg_tmp,axis=0)
    ff_tsrc_avg = np.mean(ff_tsrc_avg_tmp,axis=0)[:,:,:,:,0,:]
    
    #psink,lorentz
    pt3 = contract("bc,ddefh->chbdef",pt2_tsrc_avg,ff_tsrc_avg)
    print(f"pt3 einsum: tsep=6,tins=0-6,psink=10,lorentz=3+4,3;tgf=30,z=0, is {pt3[6,0:7,10,3,30,0]+pt3[6,0:7,10,4,30,0]}")
    #do it manually
    #print(f"pt3 without einsum {pt2_tsrc_avg[10,0]*ff_tsrc_avg[3,3,30,0,0]}")
    pt3_counter += (pt3[6,0:7,10,3,30,0]+pt3[6,0:7,10,4,30,0])
    running_avg = pt3_counter / (icfg + 1)
    print(f"pt3_sum  = {pt3_counter}",flush=True)
    print(f"average over {icfg+1} configs is {running_avg}")
    save_path = f"./pdf_data/3pt_vacroll_avgpersrc_cfg{cfg}.h5"

    #with h5py.File(save_path, "w") as f_out:
    #    f_out.create_dataset("pt3", data=pt3)


"""
FF = FF/len(cfg_list)
FF = np.mean(FF,axis=-1)
x = np.arange(FF.shape[3])
for i in range(0,6):
    plt.figure(figsize=(10, 6))
    for j in range(0,41):
        diag = FF[i, i, j, :, iq]
        plt.plot(x, diag.real, marker="o", linewidth=1, markersize=3,label=f"tgf=10*j")
        
    plt.xlabel("z")
    plt.ylabel(f"FF{i}{i}")
    plt.title(f"FF[{i},{i},iq={iq}]")
    plt.savefig( f"./plots/FF_diag_i{i}_iq{iq}_real.png", dpi=300, bbox_inches="tight")
    plt.close()
"""
