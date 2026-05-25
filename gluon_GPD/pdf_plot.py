import sys
sys.path.append('/ccs/home/sicheng/Correlation_Analysis')
import matplotlib.pyplot as plt
import h5py
import numpy as np
from module.resampling import *
tsep=4
w_list = [0,1,2,3,4,5,6,7,8,9]
tgf_list = [5]
cfg_list = list(range(204,1404,6))
tsrc_list = list(range(0,96,12))
T=96
psink=3
pt2 = np.zeros((len(cfg_list),T), dtype="<c16")
total_count=0
for i_cfg,cfg in enumerate(cfg_list):
    ncfg = (cfg - 204) // 6

    for ix,xsrc in enumerate(range(0,32,8)):
        shifted_x = (xsrc + ncfg * 3) % 32
        file_path = f"/lustre/orion/lgt132/scratch/sicheng/GPD_calc/pion_2pt_functions/N40_G45_ez/pion_smear40_mom2p5_G45_ix{ix}_x{shifted_x}_cfg{cfg}.h5"

        try:
            with h5py.File(file_path, "r") as f:
                data = f["pion_45"][0, :, :, :, :, psink + 5, :]

                for i_tsrc, tsrc in enumerate(tsrc_list):
                    shifted_t = (tsrc + ncfg * 5) % T

                    temp_data = np.roll(data[i_tsrc, ...], shift=-shifted_t, axis=-1)

                    pt2[i_cfg] += np.mean(temp_data, axis=(0, 1, 2))
                    total_count += 1


        except OSError:
            print(f"Warning: File {file_path} not found, skipping.")
    pt2[i_cfg] = pt2[i_cfg]/32

jk_ls_pt2 = jackknife(pt2)
pt2_mean = jk_ls_avg(jk_ls_pt2)
print("pt2 imag/real =", np.linalg.norm(np.mean(pt2,axis=0).imag) / (np.linalg.norm(np.mean(pt2,axis=0).real) + 1e-300))
print(f"pt2_mean tsink = {tsep}is {pt2_mean[tsep]}")



pt3=np.zeros((200,16,96,11,6,41,10),"<c16")
for i_cfg, cfg in enumerate(cfg_list):
        with h5py.File(f"/lustre/orion/lgt132/scratch/sicheng/GPD_calc/pdf_data/3pt_long_avg8timeslice_vacsubtsrcroll_cfg{cfg}.h5","r") as f:
            pt3[i_cfg] = f["pt3"][...] 

#print("pt3 imag/real =", np.linalg.norm(np.mean(pt3_plot[:,:,2,30],axis=0).imag) / (np.linalg.norm(np.mean(pt3_plot[:,:,2,30],axis=1).real) + 1e-300))
#print("ratio imag/real =", np.linalg.norm(jk_ls_ratio.imag) / (np.linalg.norm(jk_ls_ratio.real) + 1e-300))

O3 = 2*pt3[:,:,:,:,0,:,:]  + pt3[:,:,:,:,3,:,:]  + pt3[:,:,:,:,4,:,:]
#O3 = pt3[:,:,:,:,0,:,:] # + pt3[:,:,:,:,4,:,:]
i_plot = 0
print(O3.shape)
plt.figure(figsize=(14, 8))
for w in w_list:
    for tgf in tgf_list:
        jk_ls_pt3 = jackknife(O3[:,tsep,:,psink+5,tgf,w])
        jk_ls_ratio = jk_ls_pt3[:, :] / jk_ls_pt2[:, tsep:tsep+1]
        #jk_ls_ratio = jk_ls_ratio / jk_ls_ratio[:, norm_index:norm_index+1]
        ratio_avg = jk_ls_avg(jk_ls_ratio)
        real = np.mean(gv.mean(ratio_avg.real))
        imag = np.mean(gv.mean(ratio_avg.imag))
        print(f"for {w} {tgf}, mean of ratio.real is {real}, mean of ratio.imag is {imag} ")
        #ratio_avg = ratio_avg / gv.mean(ratio_avg[3])
        plt.errorbar(
                np.arange(len(ratio_avg)) + i_plot*.025,
                gv.mean(ratio_avg),
                yerr = gv.sdev(ratio_avg),
                capsize=3,
                label=f"wilsonlen={w},gradient flow time={(tgf)*0.1:.2f}",
        )
        i_plot += 1
plt.xlabel("Insertion Time (t_ins)", fontsize=12)
plt.ylabel("Ratio <C3> / <C2>", fontsize=12)
plt.title(f"3pt/2pt Ratio  tsink = {tsep}; q=0", fontsize=14)
plt.legend(
    loc="upper center",
    bbox_to_anchor=(1.02, 0.5),
    fontsize=11,
    frameon=True,
    )
#plt.ylim(0,2)
#plt.xlim(0 ,tsep)
#plt.xticks(np.arange(0, tsep + 1, 1))
plt.show()
#plt.savefig(f"/gpfs/scratch/sicheliu/3pt_figs/traceless/ratio_psink{psink}tsink{tsink}.png")

plt.savefig(
        f"./plots/3pt_long_4tsrc_p{psink}_tsep{tsep}_w{w_list}_flow{tgf_list}cfgs{len(cfg_list)}.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
    )

print("mean:", gv.mean(ratio_avg))
print("err :", gv.sdev(ratio_avg))
print("rel :", gv.sdev(ratio_avg) / np.abs(gv.mean(ratio_avg)))

