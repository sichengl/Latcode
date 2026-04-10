import yaml

N_nodes = 7
cfg_per_job = 5
monte_carlo_sep = 12

smear_mom_x = 2.5
start_cfg = 1480
smear_steps = 40

# ===== cfg list =====
cfg_list = [
    start_cfg + i * monte_carlo_sep * cfg_per_job
    for i in range(N_nodes)
]

for cfg in cfg_list:

    pars = {
        "Ls": 32,
        "Lt": 96,
        "n": cfg_per_job,
        "step": monte_carlo_sep,
        "start_cfg": cfg,
        "rho": 4.961,
        "smear_steps": smear_steps,
        "smear_mom": [smear_mom_x, 0, 0],
        "mom_min": -20,
        "mom_max": 21,
        "n_t": 3,
        "n_s": 2
    }

    smom_str = str(smear_mom_x).replace(".", "p")

    fname = (
        f"/lustre/orion/lgt132/scratch/sicheng/gluon_gpd_benchmark/"
        f"pion_2pt/cfg_files/"
        f"cfg_{cfg}_N{smear_steps}_smom{smom_str}.yaml"
    )

    with open(fname, "w") as f:
        yaml.dump(pars, f, default_flow_style=False)
