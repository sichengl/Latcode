import sys
sys.path.append('/ccs/home/sicheng/Correlation_Analysis')
import module.resampling as resamp
import h5py
import numpy as np
import matplotlib.pyplot as plt
import gvar as gv


MOM_idx = 0
with h5py.File(f"/lustre/orion/lgt132/scratch/sicheng/gluon_gpd_benchmark/Fmunu/complete_corr/FF_corr_complete_2.h5", "r") as f:

    dset = f[f"corr"]
    momentum_list_sink = dset.attrs["momentum_list_sink"] 
    smear_list = dset.attrs["number_of_smears"]
    wilson_line_list = dset.attrs["wilson_line_list"]
    measurement_list = dset.attrs["config_list"]
    smear_len = dset.attrs["number_of_steps_each_cycle"]
    smear_step_size = dset.attrs["size_of_each_step"]
    data = dset[:].real



print(data.shape) 
FF = np.zeros( (data.shape[0], data.shape[3],data.shape[4],data.shape[5]), "<c16" )

corr_bs_gamma1 = resamp.jackknife(data)

for i in range(0,6):
    
    FF += data[:,i,i,:,:,:]  
    
    


#meff = np.log(FF[:,:, :-1] / FF[:,:, 1:])
meff = FF

for i_px, px in enumerate(momentum_list_sink):
    for i_smear, smear in enumerate(smear_list):
        
        meff_to_plot = meff[:,i_smear,:,i_px]
        meff_avg =  resamp.jk_ls_avg(meff_to_plot)
        plt.errorbar(
                np.arange(len(meff_avg)),
                gv.mean(meff_avg),
                gv.sdev(meff_avg),
                fmt="o",ms=6,
                capsize=4,
                label=f"P={px}, smear={smear*smear_step_size}",
                )

    #plt.ylim([0, 2])
    #plt.xlim([0, 15])
    plt.legend()
    plt.title(f"jackknife Effective mass FF, P={px} ")
    plt.xlabel("sep")
    plt.ylabel("Meff")
    #plt.show() #? bootstrap gives a similar plot to jackknife, as we expected
    plt.savefig(f"/ccs/home/sicheng/plots/FFv2_p{px[0]}{px[1]}{px[2]}.png", dpi=300, bbox_inches='tight')
    plt.close()
