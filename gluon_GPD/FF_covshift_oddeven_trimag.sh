#!/bin/bash
# Begin LSF Directives
#SBATCH -A lgt132
#SBATCH -t 2:00:00
#SBATCH -J FF_corrected
#SBATCH -o /lustre/orion/lgt132/scratch/sicheng/GPD_calc/log/FF_logs/FF800.%J
#SBATCH -e /lustre/orion/lgt132/scratch/sicheng/GPD_calc/log/FF_logs/FF800.%J
#SBATCH -N 100
#SBATCH --distribution=cyclic
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1

rundir=$SLURM_SUBMIT_DIR
cd $rundir
date
. /ccs/home/syritsyn/build/frontier/pyquda-2025/env.sh

export QUDA_ENABLE_TUNING=1
export QUDA_RESOURCE_PATH=${QUDA_RPATH}
export QUDA_PROFILE_OUTPUT_BASE=${QUDA_RPATH}/profile_
export QUDA_ENABLE_P2P=0
export QUDA_ENABLE_MPS=1
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
export PYTHONPATH=$PYTHONPATH:/ccs/home/sicheng/packages


echo "starting multi-node job..."
for ((cfg_n=204; cfg_n<=4998; cfg_n+=6))
do
    srun -u -N 1 -n 1 --exact \
        --output="./log/FFlogs/FF800%j_cfg${cfg_n}.out" \
        --error="./log/FFlogs/FF800%j_cfg${cfg_n}.out" \
        python3 FF_covshift_oddeven_trimag2.py --config "{\"cfg_n\":${cfg_n}}" &
	
    sleep 0.1s
done

wait
echo "ALL JOB SUBMITTED！"





