import matplotlib.pyplot as plt
import h5py
import numpy as np
tsink=7
tgf_list = [9]
cfg_list = list(range(1008,1440,12))
psink = 3
iq = 0
pt3_plot = np.zeros((16,32,10),"<c16")
for i_cfg, cfg in enumerate(cfg_list):
    for i in [0,1,2]:
        with h5py.File(f"./3pt_data_background/cfg{cfg}_psink{psink}_munu{i}_rhosig{i}200cfgs_3pt.h5","r") as f:
            pt3 = f["pt3"][iq,tsink,:,:,:]
            print(pt3.shape)
            pt3_plot  += pt3
    for j in [3,4,5]:        
        with h5py.File(f"./3pt_data_background/cfg{cfg}_psink{psink}_munu{j}_rhosig{j}200cfgs_3pt.h5","r") as f:
            pt3 = f["pt3"][iq,tsink,:,:,:]
            pt3_plot  += pt3


pt2 = h5py.File(f"pion_2pt_avg_psink{psink}.h5", "r")["pt2"][:]
for w in range(0,5):
    for t_gf in tgf_list:
        plt.plot(np.arange(len(pt3_plot[:,w,t_gf])),(pt3_plot[:,w,t_gf])/pt2[tsink+8], label=f"wilson_len={w}, smear={t_gf}")
    


plt.legend()
plt.xlim(0 ,8+tsink)
plt.show()
