import h5py
import numpy as np

cfg_list = range(1008, 3408, 12)  
xsrc_indices = range(0, 32, 4)   
# 你想要提取的那个 psink 的索引
tsrc_list = range(0, 96, 12)
T = 96       
target_psink_idx = 5       

pt2_sum = np.zeros(T, dtype="<c16")
total_count = 0

for cfg in cfg_list:
    ncfg = (cfg - 1008) // 12

    for xsrc in xsrc_indices:
        shifted_x = (xsrc + ncfg * 3) % 32
        file_path = f"/gpfs/scratch/sicheliu/N40_G45_splitx/xsrc_{shifted_x}pion_smear40_mom0_G45_cfg{cfg}.h5"

        try:
            with h5py.File(file_path, "r") as f:
                data = f["pion_45"][0, :, :, :, :, target_psink_idx, :]

                for i_tsrc, tsrc in enumerate(tsrc_list):
                    shifted_t = (tsrc + ncfg * 3) % T

                    temp_data = np.roll(data[i_tsrc, ...], shift=-shifted_t, axis=-1)

                    pt2_sum += np.mean(temp_data, axis=(0, 1, 2))
                    total_count += 1

            if cfg % 120 == 0 and xsrc == 0:
                print(f"Finished cfg {cfg}...")

        except OSError:
            print(f"Warning: File {file_path} not found, skipping.")

pt2_avg = pt2_sum / total_count

print("Done! pt2_avg shape:", pt2_avg.shape)

with h5py.File(f"pion_2pt_avg_psink{target_psink_idx}.h5", "w") as f_out:
    f_out.create_dataset("pt2", data=pt2_avg)


