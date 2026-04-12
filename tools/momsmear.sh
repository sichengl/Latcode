#!/bin/bash
# Begin LSF Directives
#SBATCH -A lgt132
#SBATCH -t 02:00:00
#SBATCH -J cfgstest1
#SBATCH -o ./logs/cfg_test1.%J
#SBATCH -e ./logs/cfg_test1.%J
#SBATCH -N 10 
#SBATCH --distribution=cyclic
#SBATCH --ntasks-per-node=8


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
for cfg_file in cfg_files/*.yaml
do
    cfg_name=$(basename $cfg_file .yaml)

    echo "running $cfg_file"

    srun -N 1 -n 8 --exclusive \
        --output="./logs/${cfg_name}_%j.out" \
        --error="./logs/${cfg_name}_%j.err" \
        python3 momsmear.py --config $cfg_file &
done

wait
echo "ALL JOB SUBMITTED！"





