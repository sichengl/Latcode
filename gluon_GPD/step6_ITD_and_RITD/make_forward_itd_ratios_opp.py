from pathlib import Path
import csv
import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "bare_matrix_element_linear_combination_opp"
OUTPUT_DIR = SCRIPT_DIR / "forward_itd_ratios_opp"

COMBO_NAME = "TXTX_plus_TYTY_minus_2XYXY"
FIT_TAG = "two_state"
CFG_LIST = list(range(204, 204 + 400 * 6, 6))
Q_TUPLE = (0, 0, 0)
PZ_LIST = list(range(7))
FLOW_LIST = [20, 25, 30, 35, 40]

LS = 32
PZ_REF = 0
W_REF = 0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

q_label = f"q{Q_TUPLE[0]}_{Q_TUPLE[1]}_{Q_TUPLE[2]}"
expected_cfg_list = np.array(CFG_LIST, dtype=np.int64)

for flow in FLOW_LIST:
    matrix_by_pz = []
    pz_used = []
    source_files = []
    cfg_ref = None
    w_ref_list = None

    for pz in PZ_LIST:
        pf_label = f"pf0_0_{pz}"
        input_name = (
            f"{COMBO_NAME}_bareM_jk_opp_{q_label}_{pf_label}_"
            f"flow{flow}_fit{FIT_TAG}_cfgs{len(CFG_LIST)}.h5"
        )
        input_path = INPUT_DIR / input_name

        if not input_path.exists():
            print(f"missing {input_path}", flush=True)
            continue

        with h5py.File(input_path, "r") as f:
            matrix_jk = f["bare_matrix_element_jk"][:].astype(np.complex128, copy=False)
            cfg_list = f["cfg_list"][:].astype(np.int64)
            w_list = f["w_list"][:].astype(np.int64)
            q_code = f["q_code"][:].astype(np.int64)
            pf_code = f["pf_code"][:].astype(np.int64)

        if matrix_jk.ndim != 2:
            raise ValueError(f"{input_path.name} expected bare_matrix_element_jk[jackknife,w]")
        if not np.array_equal(q_code, np.array(Q_TUPLE, dtype=np.int64)):
            raise ValueError(f"{input_path.name} is not forward q={Q_TUPLE}")
        if not np.array_equal(cfg_list, expected_cfg_list):
            raise ValueError(f"{input_path.name} cfg_list does not match CFG_LIST")
        if pf_code[0] != 0 or pf_code[1] != 0 or int(pf_code[2]) != pz:
            raise ValueError(f"{input_path.name} pf_code={tuple(pf_code)} does not match pf0_0_{pz}")

        if cfg_ref is None:
            cfg_ref = cfg_list
            w_ref_list = w_list
        else:
            if not np.array_equal(cfg_ref, cfg_list):
                raise ValueError(f"{input_path.name} cfg_list differs from previous pz")
            if not np.array_equal(w_ref_list, w_list):
                raise ValueError(f"{input_path.name} w_list differs from previous pz")

        matrix_by_pz.append(matrix_jk)
        pz_used.append(pz)
        source_files.append(str(input_path))

    if not matrix_by_pz:
        print(f"skip flow {flow}: no input files found", flush=True)
        continue

    pz_list = np.array(pz_used, dtype=np.int64)
    w_list = np.array(w_ref_list, dtype=np.int64)
    if PZ_REF not in pz_list:
        raise ValueError(f"flow {flow}: missing pz_ref={PZ_REF}")
    if W_REF not in w_list:
        raise ValueError(f"flow {flow}: missing w_ref={W_REF}")

    pz_ref_index = int(np.where(pz_list == PZ_REF)[0][0])
    w_ref_index = int(np.where(w_list == W_REF)[0][0])

    itd_jk = np.stack(matrix_by_pz, axis=1)
    nu = (2.0 * np.pi / LS) * pz_list[:, None] * w_list[None, :]

    with np.errstate(divide="ignore", invalid="ignore"):
        single_ratio_jk = itd_jk / itd_jk[:, pz_ref_index : pz_ref_index + 1, :]
        double_ratio_jk = (
            itd_jk / itd_jk[:, :, w_ref_index : w_ref_index + 1]
        ) / (
            itd_jk[:, pz_ref_index : pz_ref_index + 1, :]
            / itd_jk[:, pz_ref_index : pz_ref_index + 1, w_ref_index : w_ref_index + 1]
        )

    n_jk = itd_jk.shape[0]

    itd_mean = np.mean(itd_jk, axis=0)
    itd_real_mean = np.mean(itd_jk.real, axis=0)
    itd_imag_mean = np.mean(itd_jk.imag, axis=0)
    itd_real_diff = itd_jk.real - itd_real_mean[None, :, :]
    itd_imag_diff = itd_jk.imag - itd_imag_mean[None, :, :]
    itd_real_sdev = np.sqrt((n_jk - 1.0) / n_jk * np.sum(itd_real_diff**2, axis=0))
    itd_imag_sdev = np.sqrt((n_jk - 1.0) / n_jk * np.sum(itd_imag_diff**2, axis=0))

    single_mean = np.mean(single_ratio_jk, axis=0)
    single_real_mean = np.mean(single_ratio_jk.real, axis=0)
    single_imag_mean = np.mean(single_ratio_jk.imag, axis=0)
    single_real_diff = single_ratio_jk.real - single_real_mean[None, :, :]
    single_imag_diff = single_ratio_jk.imag - single_imag_mean[None, :, :]
    single_real_sdev = np.sqrt((n_jk - 1.0) / n_jk * np.sum(single_real_diff**2, axis=0))
    single_imag_sdev = np.sqrt((n_jk - 1.0) / n_jk * np.sum(single_imag_diff**2, axis=0))

    double_mean = np.mean(double_ratio_jk, axis=0)
    double_real_mean = np.mean(double_ratio_jk.real, axis=0)
    double_imag_mean = np.mean(double_ratio_jk.imag, axis=0)
    double_real_diff = double_ratio_jk.real - double_real_mean[None, :, :]
    double_imag_diff = double_ratio_jk.imag - double_imag_mean[None, :, :]
    double_real_sdev = np.sqrt((n_jk - 1.0) / n_jk * np.sum(double_real_diff**2, axis=0))
    double_imag_sdev = np.sqrt((n_jk - 1.0) / n_jk * np.sum(double_imag_diff**2, axis=0))

    output_name = (
        f"{COMBO_NAME}_forward_itd_ratios_"
        f"{q_label}_flow{flow}_zref{W_REF}_cfgs{len(CFG_LIST)}.h5"
    )
    output_path = OUTPUT_DIR / output_name
    csv_path = OUTPUT_DIR / output_name.replace(".h5", ".csv")

    with h5py.File(output_path, "w") as f:
        f.create_dataset("cfg_list", data=cfg_ref)
        f.create_dataset("pz_list", data=pz_list)
        f.create_dataset("w_list", data=w_list)
        f.create_dataset("nu", data=nu.astype("<f8"))
        f.create_dataset("source_files", data=np.array(source_files, dtype=h5py.string_dtype()))

        itd = f.create_group("itd")
        itd.create_dataset("itd_jk", data=itd_jk.astype("<c16"))
        itd.create_dataset("itd_mean", data=itd_mean.astype("<c16"))
        itd.create_dataset("real_mean", data=itd_real_mean.astype("<f8"))
        itd.create_dataset("real_sdev", data=itd_real_sdev.astype("<f8"))
        itd.create_dataset("imag_mean", data=itd_imag_mean.astype("<f8"))
        itd.create_dataset("imag_sdev", data=itd_imag_sdev.astype("<f8"))

        ratios = f.create_group("ratios")
        ratios.create_dataset("single_ratio_jk", data=single_ratio_jk.astype("<c16"))
        ratios.create_dataset("double_ratio_jk", data=double_ratio_jk.astype("<c16"))
        ratios.create_dataset("single_mean", data=single_mean.astype("<c16"))
        ratios.create_dataset("double_mean", data=double_mean.astype("<c16"))
        ratios.create_dataset("single_real_mean", data=single_real_mean.astype("<f8"))
        ratios.create_dataset("single_real_sdev", data=single_real_sdev.astype("<f8"))
        ratios.create_dataset("single_imag_mean", data=single_imag_mean.astype("<f8"))
        ratios.create_dataset("single_imag_sdev", data=single_imag_sdev.astype("<f8"))
        ratios.create_dataset("double_real_mean", data=double_real_mean.astype("<f8"))
        ratios.create_dataset("double_real_sdev", data=double_real_sdev.astype("<f8"))
        ratios.create_dataset("double_imag_mean", data=double_imag_mean.astype("<f8"))
        ratios.create_dataset("double_imag_sdev", data=double_imag_sdev.astype("<f8"))

        f.attrs["combo_name"] = COMBO_NAME
        f.attrs["fit_tag"] = FIT_TAG
        f.attrs["flow"] = flow
        f.attrs["q_code"] = Q_TUPLE
        f.attrs["Ls"] = LS
        f.attrs["pz_ref"] = PZ_REF
        f.attrs["w_ref"] = W_REF
        f.attrs["dim_itd_jk"] = "jackknife,pz_index,w_index"
        f.attrs["nu_convention"] = "nu = (2*pi/Ls) * pf_code_z * w"
        f.attrs["single_ratio_definition"] = "M(pz,w) / M(pz_ref,w)"
        f.attrs["double_ratio_definition"] = (
            "(M(pz,w)/M(pz,w_ref)) / (M(pz_ref,w)/M(pz_ref,w_ref))"
        )
        f.attrs["source"] = "bare_matrix_element_linear_combination_off"
        f.attrs["nu_convention"] = (
        "nu_code = (2*pi/Ls) * pf_code_z * w; "
        "wilson_line flipped to negative direction, +i for sink coordinate and insertion, pi=pf+q"
            )
    with csv_path.open("w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(
            [
                "flow",
                "pz",
                "w",
                "nu",
                "itd_real_mean",
                "itd_real_sdev",
                "itd_imag_mean",
                "itd_imag_sdev",
                "single_real_mean",
                "single_real_sdev",
                "single_imag_mean",
                "single_imag_sdev",
                "double_real_mean",
                "double_real_sdev",
                "double_imag_mean",
                "double_imag_sdev",
            ]
        )
        for ipz, pz in enumerate(pz_list):
            for iw, w in enumerate(w_list):
                writer.writerow(
                    [
                        flow,
                        int(pz),
                        int(w),
                        nu[ipz, iw],
                        itd_real_mean[ipz, iw],
                        itd_real_sdev[ipz, iw],
                        itd_imag_mean[ipz, iw],
                        itd_imag_sdev[ipz, iw],
                        single_real_mean[ipz, iw],
                        single_real_sdev[ipz, iw],
                        single_imag_mean[ipz, iw],
                        single_imag_sdev[ipz, iw],
                        double_real_mean[ipz, iw],
                        double_real_sdev[ipz, iw],
                        double_imag_mean[ipz, iw],
                        double_imag_sdev[ipz, iw],
                    ]
                )

    print(f"saved {output_path}", flush=True)
    print(f"saved {csv_path}", flush=True)


