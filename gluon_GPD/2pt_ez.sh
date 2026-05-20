#!/bin/bash
# Begin LSF Directives
#SBATCH -A lgt132
#SBATCH -t 6:00:00
#SBATCH -J 200cfgsez
#SBATCH -o ./logs/ez_2pt/pion2pt.%j.out
#SBATCH -e ./logs/ez_2pt/pion2pt.%j.out
#SBATCH -N 800
#SBATCH --ntasks-per-node=8
#SBATCH -p batch

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

WRAPPER="wrappers/select_gpu_Ncfg_${SLURM_JOB_ID}"

mkdir -p wrappers logs/splitx

cat << EOF > "${WRAPPER}"
#!/bin/bash
export GPU_MAP=(0 1 2 3 7 6 5 4)
export NUMA_MAP=(3 3 1 1 2 2 0 0)
export GPU=\${GPU_MAP[\$SLURM_LOCALID]}
export NUMA=\${NUMA_MAP[\$SLURM_LOCALID]}
export HIP_VISIBLE_DEVICES=\$GPU
unset ROCR_VISIBLE_DEVICES
echo RANK \$SLURM_LOCALID using GPU \$GPU
exec numactl -m \$NUMA -N \$NUMA \$*
EOF

chmod +x "${WRAPPER}"

START_CFG=${INITIAL_CFG:-204}
CFG_STEP=${CFG_STEP:-6}
X_SRC_PER_CFG=4
N_CFG=200
N_TASKS=$((N_CFG * X_SRC_PER_CFG))

echo "starting ${N_TASKS} one-node xsrc jobs inside allocation ${SLURM_JOB_ID}..."

for task_id in $(seq 0 $((N_TASKS - 1))); do
    cfg_idx=$((task_id / X_SRC_PER_CFG))
    ix=$((task_id % X_SRC_PER_CFG))
    cfg=$((START_CFG + cfg_idx * CFG_STEP))
    cfg_name="cfg${cfg}_ix${ix}"

    echo "launching task ${task_id}: cfg_idx ${cfg_idx}, cfg ${cfg}, ix ${ix}"

    srun -u --exclusive -N 1 -n 8 \
        --output="./logs/splitx/${cfg_name}_%j.out" \
        --error="./logs/splitx/${cfg_name}_%j.out" \
        "${WRAPPER}" python3 2pt_ez.py --cfg $cfg --ix $ix &
	
    sleep 0.5s
done

wait
echo "all ${N_TASKS} xsrc jobs finished"
