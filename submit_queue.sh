#!/bin/sh

#SBATCH --account=
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=0

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module load python/3.12-25.3.0
GRIDFM_DATAKIT_PATH="/home//atif/gridfm-datakit/" # FIX THIS PATH
source $GRIDFM_DATAKIT_PATH/venv_datakit/bin/activate 

date "+%T"

NUM_PARALLEL=2048   # number of scenarios to run in parallel

run_scenario() {
    local idx=$1
    local logfile="./logs/scenario_${idx}.log"
    mkdir -p ./logs
    echo "[$(date +%T)] Starting scenario $idx"
    srun --ntasks=1 --cpus-per-task=1 --distribution=cyclic $GRIDFM_DATAKIT_PATH/venv_datakit/bin/python Powermodels-case_ACTIVSg2000.py --start-index $idx --end-index $((idx + 1)) \
        > "$logfile" 2>&1 && \
        echo "[$(date +%T)] Scenario $idx DONE" || \
        echo "[$(date +%T)] Scenario $idx FAILED — see $logfile"
}

export -f run_scenario

START_INDEX=0
END_INDEX=81023
seq $START_INDEX $((END_INDEX - 1)) | \
    xargs -P $NUM_PARALLEL -I {} bash -c 'run_scenario "$@"' _ {}

date "+%T"
