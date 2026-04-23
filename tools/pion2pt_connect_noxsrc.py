import h5py
import numpy as np
from pathlib import Path

def concat_h5_in_directory(input_dir, output_dir, output_filename, dataset_name):
    """
    read all the HDF5 files under designated directory, and concatenate them with 
    designated axis, defulted to 0

    parameters:
        input_dir (str): the directory that stores all the .h5 files
        output_dir (str): diectory of the merged file (name not included)
        output_filename (str): name of the merged file (eg. "N60G45MOM2p5_complete_.h5")
        dataset_name (str): name of the dataset within the final merged file
    """
    input_path = Path(input_dir)
    
    file_list = sorted(
        input_path.glob("*.h5"),
        key=lambda f: int(f.stem.split("cfg")[-1]) #sort files according to the number after str "cfg"
    )
    
    if not file_list:
        print(f"No .h5 file found under path {input_dir} ！")
        return

    print(f" Found {len(file_list)} .h5 files. Now merging them...")

    all_data = []
    
    #loop over files found
    for fname in file_list:
        try:
            with h5py.File(fname, "r") as f:
                if dataset_name in f:
                    data = f[dataset_name][:]
                    all_data.append(data)
                    print(f" File found with name: {fname.name}")
                else:
                    print(f"Warining: {fname.name} doesn't have dataset '{dataset_name}'")
        except Exception as e:
            print(f"error when reading {fname.name} : {e}")

    if not all_data:
        print("No data extracted")
        return

    #merging data
    complete_data = np.concatenate(all_data, axis=0)
    print(f"shape of merged file: {complete_data.shape}")

    output_path = Path(output_dir) / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"writing merged data to : {output_path}")
    with h5py.File(output_path, "w") as f:
        f.create_dataset(dataset_name, data=complete_data)
        
    print("Done")


if __name__ == "__main__":
    concat_h5_in_directory(
        input_dir="/lustre/orion/lgt132/scratch/sicheng/gluon_gpd_benchmark/pion_2pt/raw_data_folder/",
        output_dir="/lustre/orion/lgt132/scratch/sicheng/gluon_gpd_benchmark/pion_2pt/N60_G45_complete/",
        output_filename="N60G45MOM2p5_complete_.h5",
        dataset_name="pion_45"
    )
