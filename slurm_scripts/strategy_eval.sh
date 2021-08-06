#!/usr/bin/env bash
#SBATCH --job-name=strategy-eval   # create a short name for your job

#SBATCH --partition=research   # use research, or wacc for quick (< 30 min) ones

# Node configs
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core
#SBATCH --time=3:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1 # GPU needed ##SBATCH --array=0-1

# Mailing stuff
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vsundar4@wisc.edu
#SBATCH --output=slurm_outputs/log-%x.%A_%a.out

# Job info
echo "== Starting run at $(date)"
echo "== Job ID: ${SLURM_JOB_ID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir: ${SLURM_SUBMIT_DIR}"
echo "== Time limit: ${SBATCH_TIMELIMIT}"

nvidia-smi

# Conda stuff
module load nvidia/cuda/11.3.1
source ~/.zshrc
micromamba activate torch38

# NVIDIA SMI monitoring
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
  while true
   do
       nvidia-smi | cat >"slurm_outputs/nvidia-smi-${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out"
       sleep 0.1
  done &
fi

# Start Job here
export PYTHONPATH=.
python quanta_SL/vis_tools/error_evaluation/${1}.py