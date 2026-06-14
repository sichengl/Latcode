#!/bin/bash
# Begin LSF Directives
#SBATCH -A lgt132
#SBATCH -t 6:00:00
#SBATCH -J 0p35
#SBATCH -o log/tune.%j.out
#SBATCH -e log/tune.%j.out
#SBATCH -N 50
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

mkdir -p wrappers
mkdir -p log/2pt_logs
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
QUARK_MOM_FRAC=${QUARK_MOM_FRAC:-0.35}
RHO=${RHO:-5.0}
N_CFG=50
N_TASKS=$((N_CFG))
echo "starting ${N_TASKS} one-node xsrc jobs inside allocation ${SLURM_JOB_ID}..."

for itask in $(seq 0 $((N_TASKS - 1))); do
    cfg=$((START_CFG + itask * CFG_STEP))
    cfg_name="cfg${cfg}"

    echo "launching task ${itask}: cfg ${cfg}"

    srun -u --exclusive -N 1 -n 8 \
        --output="./log/2pt_logs/${cfg_name}_%j.out" \
        --error="./log/2pt_logs/${cfg_name}_%j.out" \
        "${WRAPPER}" python3 2pt_ez_papersetup_rho.py --cfg $cfg --quark $QUARK_MOM_FRAC --rho $RHO &
	
    sleep 0.5s
done

wait
echo "all ${N_TASKS} xsrc jobs finished"
