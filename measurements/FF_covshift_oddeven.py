#use loop function to construct the clovers and then get Fmunu
#gauge.loop takes in four sets of list of loops, and four parameters
#it calculates the product of each set of list of loops, sum them with the parameter
#then it replaces the four gauge links at each site with the four products, which are sum of one of the four list of loops, which are sumed with the parameter
#Note the fourth component of Qij and Qi4 are not used
#modified to start with sep=0 for wilson line

#==========================
#For direction, xyzt
#For data structure (index), tzyx

from pyquda_utils import core, io, phase
from pyquda_utils.core import X, Y, Z, T
import cupy as cp
from opt_einsum import contract
from pyquda_utils.core import LatticeFermion, LatticeGauge
import matplotlib.pyplot as plt
import h5py
import numpy as np
import json
import argparse
from cupy.cuda.runtime import deviceSynchronize
from time import perf_counter

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()


core.init([1, 1, 1, 1], resource_path="/lustre/orion/lgt132/scratch/sicheng/gluon_gpd_benchmark/.cache/quda")
parameters = json.loads(args.config)
start_cfg = parameters["cfg_n"]
num_cfg = 5
smear_len = 10
smear_parts = 10
measurement_list = list(range(start_cfg,start_cfg+num_cfg*12,12))
src_list = list(range(0,6))
sink_list = list(range(0,6))
smear_list = list(range(0,10))
wilson_line_list = list(range(0,32))
momentum_list_sink = []
for px in [0, -1, 1]:
    for py in [0, -1, 1]:
        for pz in [0, -1, 1]:
            momentum_list_sink.append([px, py, pz])

for i_cfg,cfg in enumerate(measurement_list):
    gauge = io.readMILCGauge(f"/lustre/orion/lgt132/world-shared/DATA/MILC/a09m310/gauge/l3296f211b630m0074m037m440e.{cfg}")
    if i_cfg == 0:
        latt_info=gauge.latt_info
        momentum_phase_sink = phase.MomentumPhase(latt_info).getPhases(momentum_list_sink)
        corr = cp.zeros((len(measurement_list),len(src_list),len(sink_list),len(smear_list), len(wilson_line_list),len(momentum_list_sink),latt_info.Lt), "<c16")
    for i_smear, smear in enumerate(smear_list):
        
        deviceSynchronize()
        s = perf_counter()        
        gauge.wilsonFlow(smear_len, 0.02)
        deviceSynchronize()
        core.getLogger().info(f"WILSON FLOW #{cfg}: {perf_counter() - s} secs")
        """
        gauge_fixing_params = {
        "gauge_dir": 4,
        "Nsteps": 10000,
        "verbose_interval": 100,
        "relax_boost": 1.7,
        "tolerance": 1e-10,
        "reunit_interval": 10,
        "stopWtheta": 0
            }
        gauge.fixingOVR(**gauge_fixing_params)
        """
        
        deviceSynchronize()
        s = perf_counter()
        Qij = gauge.loop(
            [
                [[X, Y, -X, -Y], [Y, -X, -Y, X], [-X, -Y, X, Y], [-Y, X, Y, -X]],
                [[Y, Z, -Y, -Z], [Z, -Y, -Z, Y], [-Y, -Z, Y, Z], [-Z, Y, Z, -Y]],
                [[X, Z, -X, -Z], [Z, -X, -Z, X], [-X, -Z, X, Z], [-Z, X, Z, -X]],
                [[T, -T, T, -T], [T, -T, T, -T], [T, -T, T, -T], [T, -T, T, -T]],
            ],
            [ 1 , 1 , 1 , 1 ],
        )

        Qi4 = gauge.loop(
            [
                [[X, T, -X, -T], [T, -X, -T, X], [-X, -T, X, T], [-T, X, T, -X]],
                [[Y, T, -Y, -T], [T, -Y, -T, Y], [-Y, -T, Y, T], [-T, Y, T, -Y]],
                [[Z, T, -Z, -T], [T, -Z, -T, Z], [-Z, -T, Z, T], [-T, Z, T, -Z]],
                [[T, -T, T, -T], [T, -T, T, -T], [T, -T, T, -T], [T, -T, T, -T]],
            ],
            [ 1 , 1 , 1 , 1 ],
        )

        Qij_dagger = LatticeGauge(latt_info)
        Qi4_dagger = LatticeGauge(latt_info)
        Qij_dagger.data[:] = Qij.data.conj().swapaxes(-1,-2)
        Qi4_dagger.data[:] = Qi4.data.conj().swapaxes(-1,-2)
        Fij = -1j / 8 * ( Qij - Qij_dagger )
        Fi4 = -1j / 8 * ( Qi4 - Qi4_dagger )

        gauge_local = cp.asarray(gauge.data[2,:])
        gauge_local_conj = gauge_local.conj().swapaxes(-1,-2)
        
        #Fmunu = [ Fij.data[0,:] , Fij.data[1,:] , Fij.data[2,:] , Fi4.data[0,:] , Fi4.data[1,:] , Fi4.data[2,:] ]
        Fmunu = [
        cp.asarray(Fij.data[0,:]),
        cp.asarray(Fij.data[1,:]),
        cp.asarray(Fij.data[2,:]),
        cp.asarray(Fi4.data[0,:]),
        cp.asarray(Fi4.data[1,:]),
        cp.asarray(Fi4.data[2,:]),
        ]

        Fmunu_shift = [arr.copy() for arr in Fmunu]
        
        deviceSynchronize()
        core.getLogger().info(f"BEFORE SHIFT #{cfg}: {perf_counter() - s} secs")

        deviceSynchronize()
        s = perf_counter()
        for i_W, WL_indices in enumerate(wilson_line_list):
            
            #for wilson line with zero length skip shifting
            if i_W != 0:
                Fmunu_shift = [ gauge_local @ cp.roll(arr[::-1], shift=-1, axis=2) @ gauge_local_conj  for arr in Fmunu_shift]
            
            for i_src, src in enumerate(src_list):
                for i_sink, sink in enumerate(sink_list):

                    corr[i_cfg,i_src,i_sink,i_smear,i_W] += contract("pwtzyx,wtzyxij,wtzyxji->pt", momentum_phase_sink,Fmunu[i_src],Fmunu_shift[i_sink] )


        deviceSynchronize()
        core.getLogger().info(f"SHIFT #{cfg}: {perf_counter() - s} secs")
        


#cp.save("fmunu_corr.npy",corr)



tmp = core.gatherLattice(corr.real.get(), [6, -1, -1, -1])




from pyquda_comm import getMPIRank
rank = getMPIRank()

if rank == 0:

    tmp = np.mean(tmp,axis=6)
    tmp_cpu = tmp
    filename = f"/lustre/orion/lgt132/scratch/sicheng/gluon_gpd_benchmark/Fmunu/incomplete_corr/FFv2_smear_{smear_len*len(smear_list)}_cfg{start_cfg}.h5"
    with h5py.File(filename, 'w') as f:
        data = f.create_dataset('corr', data=tmp_cpu)

        data.attrs['number_of_steps_each_cycle'] = smear_len
        data.attrs['size_of_each_step'] = 0.02
        data.attrs['number_of_smears'] = smear_list
        data.attrs['wilson_line_list'] = wilson_line_list
        data.attrs['momentum_list_sink'] = momentum_list_sink
        data.attrs['config_list'] = measurement_list
    #cp.save(f"fmunu_corr_smear_{smear_len*smear_parts}_insteps_from0_MILC.npy",tmp)

    """ 
    import matplotlib.pyplot as plt
    import numpy as np
    y_data = np.zeros((len(wilson_line_list)),"<c16")
    print(f"Rank {rank} is plotting...")
    tmp = cp.mean(tmp,axis=(3))
    for j in range(0,6):
        y_data = y_data + tmp[j,j,:,0]
    t_axis = range(len(y_data))


    print(f"ydata is {y_data}")


    plt.figure(figsize=(8, 6))
    plt.plot(t_axis,y_data, marker='o', linestyle='-', color='b')
    plt.yscale('log')
    plt.xlabel('l')
    plt.ylabel('Corr')
    plt.title('Field Strength Correlator')
    plt.grid(True)
    plt.savefig('fsc_correlation_plot.png')
    print("Plot saved successfully.")
else:
    pass
    """
