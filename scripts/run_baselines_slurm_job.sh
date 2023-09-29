#!/bin/bash
#SBATCH --time=0-24:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --account=hai_microbio
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --nice=100
#SBATCH --job-name=eval_model
#SBATCH --output=/p/project/hai_microbio/sb/repos/odeformer/experiments/out2/run_baseline_%j.out
#SBATCH --error=/p/project/hai_microbio/sb/repos/odeformer/experiments/out2/run_baseline_%j.err

# Move either to project root dir or the config file path.
export GIT_PYTHON_REFRESH=quiet
export TOKENIZERS_PARALLELISM=false
cd /p/project/hai_microbio/sb/dev/repos/odeformer

# Print job information
echo "Starting job ${SLURM_JOBID}"
echo "SLURM assigned me the node(s): $(squeue -j ${SLURM_JOBID} -O nodelist:1000 | tail -n +2 | sed -e 's/[[:space:]]*$//')"

# Activate Anaconda environment
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate odeformer39
echo "Submitting job for ${1}"
srun python /p/project/hai_microbio/sb/repos/odeformer/run_baselines.py --baseline_model ${1} &

wait