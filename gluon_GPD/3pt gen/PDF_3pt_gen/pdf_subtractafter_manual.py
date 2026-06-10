import numpy as np
import h5py
from opt_einsum import contract


cfg_list = list(range(204, 1404, 6))
tsrc_list = list(range(0, 96, 12))
T = 96

ff_template = "./FF_data/FF_fixed_smear_410_cfg{cfg}.h5"
pt2_template = (
    "./pion_2pt_functions/N40_G45_ez/"
    "pion_smear40_mom2p5_G45_ix{ix}_x{shifted_x}_cfg{cfg}.h5"
)
out_template = "./pdf_data/3pt_conn_inputs_man_cfg{cfg}.h5"


for icfg, cfg in enumerate(cfg_list):
    ncfg = (cfg - 204) // 6
    print(cfg)

    with h5py.File(ff_template.format(cfg=cfg), "r") as f:
        # [src_lorentz, sink_lorentz, tgf, z_WL, i_qext, tFF]
        FF = f["corr"][0, ...]

    # [tsrc, ix, ysrc, zsrc, psink, t]
    pt2 = np.zeros((8, 4, 4, 8, 11, T), "<c16")
    for ix, xsrc in enumerate(range(0, 32, 8)):
        shifted_x = (xsrc + ncfg * 3) % 32
        path = pt2_template.format(ix=ix, shifted_x=shifted_x, cfg=cfg)

        with h5py.File(path, "r") as f:
            pt2[:, ix, :, :, :, :] = f["pion_45"][0, :, 0, :, :, :, :]

    # [tsrc, psink, t]
    pt2_src_avg = np.mean(pt2, axis=(1, 2, 3)) #avg out xyz

    # take the diagonal in lorentz; sum rolled FF over tsrc
    # Shape: [tins, mu_diag, tgf, z_WL]
    FF_tsrc_sum = np.zeros((T, 6, 41, 10), "<c16")

    # sum rolled 2pt over tsrc
    # Shape: [psink, t]
    pt2_tsrc_sum = np.zeros((11, T), "<c16")

    # sum 2pt * O over tsrc, no vacuum subtraction
    # Shape: [tsep, tins, psink, mu_diag, tgf, z_WL]
    pt2FF_tsrc = np.zeros((16, T, 11, 6, 41, 10), "<c16")

    for tsrc in tsrc_list:
        shifted_t = (tsrc + ncfg * 5) % T
        it = tsrc_list.index(tsrc)

        # [psink, tsep], source-at-origion and cut to tsep=0..15
        pt2_roll_full = np.roll(pt2_src_avg[it, ...], -shifted_t, axis=-1)
        pt2_roll_cut = pt2_roll_full[..., 0:16]

        # [src_lorentz, sink_lorentz, tgf, z_WL, t], q=0, source-at-origion
        FF_roll = np.roll(FF, -shifted_t, axis=-1)[..., 0, :]

        # take the diagonal of FF in lorentz space
        # [tins, mu_diag, tgf, z_WL]
        FF_tsrc_sum += contract("ddefh->hdef", FF_roll)
        pt2_tsrc_sum += pt2_roll_full

        # [tsep, tins, psink, mu_diag, tgf, z_WL]
        pt2FF_tsrc += contract("bc,ddefh->chbdef", pt2_roll_cut, FF_roll)

    FF_tsrc_avg = FF_tsrc_sum / len(tsrc_list)
    pt2_tsrc_avg = pt2_tsrc_sum / len(tsrc_list)
    pt2FF_tsrc_avg = pt2FF_tsrc / len(tsrc_list)

    with h5py.File(out_template.format(cfg=cfg), "w") as f_out:
        f_out.create_dataset("pt2FF", data=pt2FF_tsrc_avg)
        f_out.create_dataset("FF", data=FF_tsrc_avg)
        f_out.create_dataset("pt2", data=pt2_tsrc_avg)

        f_out.attrs["dim_CO"] = "tsep,tins,psink,mu_diag,tgf,z_WL"
        f_out.attrs["dim_O_tsrc_avg"] = "tins,mu_diag,tgf,z_WL"
        f_out.attrs["dim_C"] = "psink,t"
        f_out.attrs["vacuum_subtraction"] = ("Done afterwards in jackknife")

