import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict

def concat_h5_in_directory(input_dir, output_dir, output_filename, dataset_name,num_xsrc=8):
    """
    read all the HDF5 files under designated directory, group them by cfg number (check == num_xsrc),
    concatenate them along axis=2, mean over axis=(2,3,4), roll along axis=-1, mean over axis=1,
    and finally concatenate all cfgs along axis=0.

    parameters:
        input_dir (str): the directory that stores all the .h5 files
        output_dir (str): diectory of the merged file (name not included)
        output_filename (str): name of the merged file (eg. "N60G45MOM2p5_complete_.h5")
        dataset_name (str): name of the dataset within the final merged file
        num_xsrc : the # of x_src's (defulted to 8)
    """
    input_path = Path(input_dir)

    # 1. 按照 cfg 后面的数字对文件进行分组
    cfg_groups = defaultdict(list)
    for f in input_path.glob("*.h5"):
        try:
            # 提取 cfg 后面的数字作为字典的 key
            cfg_num = int(f.stem.split("cfg")[-1])
            cfg_groups[cfg_num].append(f)
        except ValueError:
            # 忽略格式不符的文件
            continue

    if not cfg_groups:
        print(f"No .h5 file found under path {input_dir} ！")
        return

    print(f"Found {sum(len(v) for v in cfg_groups.values())} .h5 files in {input_dir}. Grouping by cfg number...")

    all_data = []

    # 2. 遍历分组，按 cfg 数字顺序处理
    for cfg_num in sorted(cfg_groups.keys()):
        files = sorted(cfg_groups[cfg_num])
        
        # 检查是否正好有 8 个文件
        if len(files) != num_xsrc:
            print(f"Warning: cfg {cfg_num} has {len(files)} files, expected {num_xsrc}. Skipping this cfg.")
            continue
            
        group_data = []
        t_src_list_group = None
        read_success = True
        
        # 依次读取这 8 个文件的数据
        for fname in files:
            try:
                with h5py.File(fname, "r") as f:
                    if dataset_name in f:
                        dset = f[dataset_name]
                        group_data.append(dset[:])
                        
                        if t_src_list_group is None:
                            t_src_list_group = dset.attrs["t_src_list"]
                    else:
                        print(f"Warning: {fname.name} doesn't have dataset '{dataset_name}'")
                        read_success = False
                        break
            except Exception as e:
                print(f"Error when reading {fname.name}: {e}")
                read_success = False
                break
                
        # 如果读取过程中发生错误，跳过当前 cfg
        if not read_success or len(group_data) != 8:
            print(f"Skipped cfg {cfg_num} due to reading errors.")
            continue

        # 3. 沿着 axis=2 拼接这 8 个文件的数据
        data_concat = np.concatenate(group_data, axis=2)
        
        # 4. 对 axis=2, 3, 4 求平均
        data_mean = np.mean(data_concat, axis=(2, 3, 4))
        
        # 应用滚动操作 (保持之前的物理意义)
        if t_src_list_group is not None:
            for i_tsrc, tsrc in enumerate(t_src_list_group):
                data_mean[:, i_tsrc, :, :] = np.roll(data_mean[:, i_tsrc, :, :], -tsrc, axis=-1)

        # 再次对 axis=1 求平均
        data_mean = np.mean(data_mean, axis=1)
        
        all_data.append(data_mean)
        print(f"Successfully processed cfg {cfg_num} (merged {num_xsrc} files).")

    if not all_data:
        print("No data extracted")
        return

    # 5. 最后对所有 cfg 的数据沿 axis=0 拼接
    complete_data = np.concatenate(all_data, axis=0)
    print(f"Shape of merged file: {complete_data.shape}")

    output_path = Path(output_dir) / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing merged data to: {output_path}")
    with h5py.File(output_path, "w") as f:
        f.create_dataset(dataset_name, data=complete_data)

    print(f"Done processing {dataset_name}\n")
    print("-" * 50)


N = 40
gamma_list = ["5", "15", "25", "35", "45", "p5"]

if __name__ == "__main__":
    for gamma in gamma_list:
        concat_h5_in_directory(
            input_dir=f"/ccs/proj/lgt132/sicheng/N{N}_G{gamma}_splitx",
            output_dir=f"/lustre/orion/lgt132/scratch/sicheng/gluon_gpd_benchmark/pion_2pt/complete/N{N}_G{gamma}_complete_splitx/",
            output_filename=f"N{N}G{gamma}MOM2p5_complete_200.h5",
            dataset_name=f"pion_{gamma}"
        )
