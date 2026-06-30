from pathlib import Path

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "bare_matrix_element_jk"
OUTPUT_DIR = SCRIPT_DIR / "bare_matrix_element_linear_combination"

CFG_LIST = list(range(204, 204 + 50 * 6, 6))
Q_LIST = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
PF_LIST = [(0, 0, pz) for pz in range(7)]
TGF_LIST = [20, 25, 30, 35, 40]
FIT_TAG = "two_state"

# Edit this dictionary for the combinations you want.
# Each term is (operator_name, coefficient).
LINEAR_COMBINATIONS = {
    "TXTX_plus_TYTY_minus_2XYXY": [
        ("TXTX", 1.0),
        ("TYTY", 1.0),
        ("XYXY", -2.0),
    ],
}


def jackknife_mean_sdev(jk_list):
    n_jk = jk_list.shape[0]
    avg = np.mean(jk_list, axis=0)
    diff = jk_list - avg[None, :]
    sdev = np.sqrt((n_jk - 1.0) / n_jk * np.sum(diff**2, axis=0))
    return avg, sdev


def process_one_combination(combo_name, terms, q_tuple, pf_tuple, flow):
    q_label = f"q{q_tuple[0]}_{q_tuple[1]}_{q_tuple[2]}"
    pf_label = f"pf{pf_tuple[0]}_{pf_tuple[1]}_{pf_tuple[2]}"

    combined_jk = None
    cfg_list = None
    w_list = None
    input_files = []

    for operator, coefficient in terms:
        input_name = (
            f"{operator}_bareM_jk_{q_label}_{pf_label}_"
            f"flow{flow}_fit{FIT_TAG}_cfgs{len(CFG_LIST)}.h5"
        )
        input_path = INPUT_DIR / input_name
        if not input_path.exists():
            print(f"missing {input_path}", flush=True)
            return

        with h5py.File(input_path, "r") as f:
            matrix_jk = f["bare_matrix_element_jk"][:]
            this_cfg_list = f["cfg_list"][:].astype(np.int64)
            this_w_list = f["w_list"][:].astype(np.int64)
            this_q_code = f["q_code"][:].astype(np.int64)
            this_pf_code = f["pf_code"][:].astype(np.int64)

        if matrix_jk.shape[0] != len(CFG_LIST):
            raise ValueError(f"{input_path.name} has inconsistent jackknife count")
        if not np.array_equal(this_q_code, np.array(q_tuple, dtype=np.int64)):
            raise ValueError(f"{input_path.name} q_code does not match")
        if not np.array_equal(this_pf_code, np.array(pf_tuple, dtype=np.int64)):
            raise ValueError(f"{input_path.name} pf_code does not match")
        if cfg_list is None:
            cfg_list = this_cfg_list
            w_list = this_w_list
            combined_jk = np.zeros_like(matrix_jk, dtype=np.complex128)
        else:
            if not np.array_equal(cfg_list, this_cfg_list):
                raise ValueError(f"{input_path.name} cfg_list does not match")
            if not np.array_equal(w_list, this_w_list):
                raise ValueError(f"{input_path.name} w_list does not match")
            if combined_jk.shape != matrix_jk.shape:
                raise ValueError(f"{input_path.name} bare_matrix_element_jk shape does not match")

        combined_jk += coefficient * matrix_jk
        input_files.append(str(input_path))

    avg = np.mean(combined_jk, axis=0)
    real_avg, real_sdev = jackknife_mean_sdev(combined_jk.real)
    imag_avg, imag_sdev = jackknife_mean_sdev(combined_jk.imag)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_name = (
        f"{combo_name}_bareM_jk_{q_label}_{pf_label}_"
        f"flow{flow}_fit{FIT_TAG}_cfgs{len(CFG_LIST)}.h5"
    )
    output_path = OUTPUT_DIR / output_name

    operator_names = np.array([term[0] for term in terms], dtype=h5py.string_dtype())
    coefficients = np.array([term[1] for term in terms], dtype=np.complex128)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("bare_matrix_element_jk", data=combined_jk.astype("<c16"))
        f.create_dataset("bare_matrix_element_mean", data=avg.astype("<c16"))
        f.create_dataset("bare_matrix_element_real_mean", data=real_avg.astype("<f8"))
        f.create_dataset("bare_matrix_element_real_sdev", data=real_sdev.astype("<f8"))
        f.create_dataset("bare_matrix_element_imag_mean", data=imag_avg.astype("<f8"))
        f.create_dataset("bare_matrix_element_imag_sdev", data=imag_sdev.astype("<f8"))
        f.create_dataset("cfg_list", data=cfg_list)
        f.create_dataset("q_code", data=np.array(q_tuple, dtype=np.int64))
        f.create_dataset("pf_code", data=np.array(pf_tuple, dtype=np.int64))
        f.create_dataset("w_list", data=w_list)
        f.create_dataset("operator_terms", data=operator_names)
        f.create_dataset("coefficients", data=coefficients.astype("<c16"))

        f.attrs["combination_name"] = combo_name
        f.attrs["flow"] = flow
        f.attrs["fit_tag"] = FIT_TAG
        f.attrs["matrix_element_dataset"] = "bare_matrix_element_jk"
        f.attrs["dim_bare_matrix_element_jk"] = "jackknife,w_index"
        f.attrs["dim_bare_matrix_element_mean"] = "w_index"
        f.attrs["jk_average_convention"] = "mean over jackknife samples"
        f.attrs["jk_sdev_convention"] = "sqrt((N-1)/N * sum((x_jk-mean)^2))"
        f.attrs["source_fit_files"] = "\n".join(input_files)

    print(f"saved {output_path}", flush=True)


def main():
    n_tasks = len(LINEAR_COMBINATIONS) * len(Q_LIST) * len(PF_LIST) * len(TGF_LIST)
    print(f"total linear-combination tasks = {n_tasks}", flush=True)
    for combo_name, terms in LINEAR_COMBINATIONS.items():
        for q_tuple in Q_LIST:
            for pf_tuple in PF_LIST:
                for flow in TGF_LIST:
                    process_one_combination(combo_name, terms, q_tuple, pf_tuple, flow)


if __name__ == "__main__":
    main()
