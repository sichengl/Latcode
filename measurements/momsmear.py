#using python dic, easy to distribute over nodes
import argparse
import h5py
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_utils import core, io, gamma, phase, source, fft, phase_v2
from pyquda_comm.array import arrayExp
from time import perf_counter
from cupy.cuda.runtime import deviceSynchronize
from tqdm import tqdm
import sys
sys.path.append('/ccs/home/sicheng/LaMETLat')
sys.path.append('/ccs/home/sicheng/LatCoding')
from pyquda_benchmark.meas.mom_smearing import *
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config",type=str)
parameters = parser.parse_args()

if parameters.config:
    with open(parameters.config) as f:
        cfg = yaml.safe_load(f)

    for k, v in cfg.items():
        setattr(parameters, k, v)

print(vars(parameters))
#transcribe parameters
Ls = parameters.Ls
Lt = parameters.Lt
n = parameters.cfgs_to_meas
step = parameters.cfgs_step_size
start_cfg = parameters.start_cfg
stop_cfg = start_cfg + n * step
rho = parameters.rho
smear_steps = parameters.smear_steps
smear_mom = parameters.smear_mom
smear_mom_x_str = ("%.3f" % smear_mom[0]).rstrip("0").rstrip(".").replace(".", "p") #the string of momentum smearing info, to be used in saving
k = np.array(smear_mom) * 2 * np.pi / Ls
k1 =  -k
k2 = k
mom_min = parameters.mom_min
mom_max = parameters.mom_max
n_t = parameters.n_t
n_s = parameters.n_s

core.init([1, 2, 2, 2], resource_path="/lustre/orion/lgt132/scratch/sicheng/gluon_gpd_benchmark/.cache/quda")
latt_info = core.LatticeInfo([Ls, Ls, Ls, Lt], -1, 1.0)
dirac = core.getDirac(latt_info, -0.05138, 1e-12, 1000, 1.0, 1.04243, 1.04243, [[4, 4, 4, 4],[2,2,2,3]])
G5 = gamma.gamma(15)
GT5 = gamma.gamma(7)
momentum_list = [[i, 0, 0] for i in range(mom_min,mom_max)]
t_src_list = list(range(0, Lt, Lt//n_t))
x_src_list = list(range(0, Ls, Ls//n_s))
y_src_list = list(range(0, Ls, Ls//n_s))
z_src_list = list(range(0, Ls, Ls//n_s))
measurement_list = list(range(start_cfg,stop_cfg,step))
pion_45 = cp.zeros((len(measurement_list),len(t_src_list),len(x_src_list),len(y_src_list),len(z_src_list), len(momentum_list), latt_info.Lt), "<c16")
pion_55 = cp.zeros((len(measurement_list),len(t_src_list),len(x_src_list),len(y_src_list),len(z_src_list), len(momentum_list), latt_info.Lt), "<c16")
count = 0
mean_rsqr = 0.0
for i_cfg, cfg in tqdm(enumerate(measurement_list),desc=f"starting from configuration {start_cfg}"):

    #READ GAUGE
    deviceSynchronize()
    s = perf_counter()
    gauge_ape = io.readMILCGauge(f"/lustre/orion/lgt132/world-shared/DATA/MILC/a09m310/gauge/l3296f211b630m0074m037m440e.{cfg}",checksum=True, reunitarize_sigma=1e-6)
    deviceSynchronize()
    core.getLogger().info(f"READ GAUGE #{cfg}: {perf_counter() - s} secs")

    #HYP
    deviceSynchronize()
    s = perf_counter()
    gauge_hyp = gauge_ape.copy()
    core.getLogger().info(f"DOING HYP SMEARING")
    core.getLogger().info(f"plaq_hyp_before = {gauge_hyp.plaquette()}")
    gauge_hyp.hypSmear(1, 0.75, 0.6, 0.3, -1, True,True)
    gauge_ape.plaquette()
    deviceSynchronize()
    core.getLogger().info(f"plaq_hyp_after = {gauge_hyp.plaquette()}")
    core.getLogger().info(f"HYP SMEAR: {perf_counter() - s} secs")

    #APE
    deviceSynchronize()
    s = perf_counter()
    #with dirac.useGauge(gauge_ape):
    core.getLogger().info(f"DOING APE SMEARING")
    core.getLogger().info(f"plaq_ape_before = {gauge_ape.plaquette()}")
    gauge_ape.apeSmear(25,0.6154 , 3)
    deviceSynchronize()
    core.getLogger().info(f"plaq_ape_after = {gauge_ape.plaquette()}")
    core.getLogger().info(f"APE SMEAR: {perf_counter() - s} secs")

    #LOAD GAUGE
    #core.getLogger().info(f"LOADING GAUGE")
    #dirac.loadGauge(gauge_hyp)
    
    for t_idx, t_src in enumerate(t_src_list):
        for x_idx, x_src in enumerate(x_src_list):
            for y_idx, y_src in enumerate(y_src_list):
                for z_idx, z_src in enumerate(z_src_list):

                    inner_loop = perf_counter()

                    src_pos = [x_src,y_src,z_src,t_src]
                    core.getLogger().info(f"SOURCE POSITION = {src_pos}")
                    momentum_phases = phase.MomentumPhase(latt_info).getPhases( momentum_list, src_pos )
                    point_src = source.propagator(latt_info, "point", src_pos)

                    #SRC GAUSSIAN SMEAR
                    deviceSynchronize()
                    s = perf_counter()
                    core.getLogger().info(f"DOING SRC GAUSSIAN SMEARING")
                    momentum_source1 = momentum_smearing_propagator(latt_info, gauge_ape, k1, src_pos, rho, smear_steps)
                    momentum_source2 = momentum_smearing_propagator(latt_info, gauge_ape, k2, src_pos, rho, smear_steps)
                    #smeared_pt_src = source.gaussianSmear(point_src, gauge_ape, 4.961, 40)
                    deviceSynchronize() 
                    core.getLogger().info(f"SOURCE GAUSSIAN SMEAR: {perf_counter() - s} secs")

                    #INVERT
                    deviceSynchronize()
                    s = perf_counter()
                    #with dirac.useGauge(gauge_hyp):
                    dirac.loadGauge(gauge_hyp)
                    core.getLogger().info(f"SOLVE DIRAC EQ")
                    propag1 = core.invertPropagator(dirac, momentum_source1)
                    propag2 = core.invertPropagator(dirac, momentum_source2)
                    deviceSynchronize()
                    core.getLogger().info(f"INVERT: {perf_counter() - s} secs")


                    #SINK GAUSSIAN SMEAR
                    deviceSynchronize()
                    s = perf_counter()
                    core.getLogger().info(f"DOING SINK GAUSSIAN SMEARING")
                    propag1_sink_smeared = momentum_smearing_sink(latt_info, propag1, gauge_ape, k1, rho, smear_steps)
                    propag2_sink_smeared = momentum_smearing_sink(latt_info, propag2, gauge_ape, k2, rho, smear_steps)
                    deviceSynchronize()
                    core.getLogger().info(f"SINK GAUSSIAN SMEAR: {perf_counter() - s} secs")

                    #CONTRACT
                    deviceSynchronize()
                    s = perf_counter()
                    pion_45[i_cfg,t_idx,x_idx,y_idx,z_idx] += contract(
                        "pwtzyx,wtzyxjiba,jk,wtzyxklba,li->pt",
                        momentum_phases,
                        propag1_sink_smeared.data.conj(),
                        G5 @ GT5,
                        propag2_sink_smeared.data,
                        GT5 @ G5,
                        )
                    pion_55[i_cfg,t_idx,x_idx,y_idx,z_idx] += contract(
                        "pwtzyx,wtzyxjiba,jk,wtzyxklba,li->pt",
                        momentum_phases,
                        propag1_sink_smeared.data.conj(),
                        G5 @ G5,
                        propag2_sink_smeared.data,
                        G5 @ G5,
                        )
                    deviceSynchronize()
                    core.getLogger().info(f"CONTRACT: {perf_counter() - s} secs")

                    #calculate the size of the smeared src only for [0,0,0,0]
                    if src_pos == [0,0,0,0]:
                        if latt_info.mpi_rank == 0:
                            deviceSynchronize()
                            s = perf_counter()
                            src_wavefunction = momentum_source1.lexico()[0,:,:,:,:,:,:,:]
                            src_density = np.sum(np.abs(src_wavefunction)**2, axis=(3,4,5,6))
                            norm = np.sum(src_density, axis=(0, 1, 2))
                            core.getLogger().info(f"norm is {norm}")
                            coordinates = phase_v2.LocationPhase(latt_info).getPhase().lexico()
                            rsqr = (np.minimum(coordinates[0],Ls-coordinates[0])**2  + np.minimum(coordinates[1],Ls-coordinates[1])**2 + np.minimum(coordinates[2],Ls-coordinates[2])**2).astype(np.float64)
                            rsqr = rsqr[0,:,:,:]
                            rsqr *= src_density
                            tmp = np.sum(np.abs(rsqr),axis=(0,1,2))/norm
                            core.getLogger().info(f"mean r^2 = {tmp}")
                            mean_rsqr += tmp
                            count += 1
                            deviceSynchronize()
                            core.getLogger().info(f"GET RADIUS: {perf_counter() - s} secs")
                            tmp=0
                    
                    core.getLogger().info(f"INNER LOOP: {perf_counter() - inner_loop} secs")

dirac.freeGauge()

pion_45_np=core.gatherLattice(pion_45.real.get(),[6,-1,-1,-1])
pion_55_np=core.gatherLattice(pion_55.real.get(),[6,-1,-1,-1])
#save as h5py file
if latt_info.mpi_rank == 0:

    mean_rsqr = mean_rsqr / count

    with h5py.File(f"/lustre/orion/lgt132/scratch/sicheng/gluon_gpd_benchmark/pion_2pt/N{smear_steps}_G45/pion_smear{smear_steps}_mom{smear_mom_x_str}_G45_cfg{start_cfg}.h5", "w") as f:
        dset = f.create_dataset("pion_45", data=pion_45_np)
        dset.attrs["dim_spec"] = np.array(["measurement_list", "t_src_list","x_src_list","y_src_list","z_src_list","momentum_list", "time"], dtype=h5py.string_dtype())
        dset.attrs["measurements"] = measurement_list
        dset.attrs["momentums"] = momentum_list
        dset.attrs["x_src_list"] = x_src_list
        dset.attrs["y_src_list"] = y_src_list
        dset.attrs["z_src_list"] = z_src_list
        dset.attrs["t_src_list"] = t_src_list
        dset.attrs["dim_time"] = np.arange(latt_info.Lt)
        dset.attrs["mean_rsqr"] = mean_rsqr

    with h5py.File(f"/lustre/orion/lgt132/scratch/sicheng/gluon_gpd_benchmark/pion_2pt/N{smear_steps}_G5/pion_smear{smear_steps}_mom{smear_mom_x_str}_G5_cfg{start_cfg}.h5", "w") as f:
        dset = f.create_dataset("pion_55", data=pion_55_np)
        dset.attrs["dim_spec"] = np.array(["measurement_list", "t_src_list","x_src_list","y_src_list","z_src_list","momentum_list", "time"], dtype=h5py.string_dtype())
        dset.attrs["measurements"] = measurement_list
        dset.attrs["momentums"] = momentum_list
        dset.attrs["x_src_list"] = x_src_list
        dset.attrs["y_src_list"] = y_src_list
        dset.attrs["z_src_list"] = z_src_list
        dset.attrs["t_src_list"] = t_src_list
        dset.attrs["dim_time"] = np.arange(latt_info.Lt)
        dset.attrs["mean_rsqr"] = mean_rsqr

