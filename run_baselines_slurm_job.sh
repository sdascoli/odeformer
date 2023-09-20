#!/bin/bash

# Move either to project root dir or the config file path.
export GIT_PYTHON_REFRESH=quiet
export TOKENIZERS_PARALLELISM=false

cd "${1}"
echo "cwd: ${1}"
echo "run_baseline_slurm_job.sh on $(hostname)"
echo "with args: ${@}"

# Print job information
echo "Starting job ${SLURM_JOBID}"
echo "SLURM assigned me the node(s): $(squeue -j ${SLURM_JOBID} -O nodelist:1000 | tail -n +2 | sed -e 's/[[:space:]]*$//')"

# Activate Anaconda environment
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate symbolicregression39

export SLURM_NTASKS=1

echo "Submitting job for ${2}"
python run_baselines.py \
    --model ${2} \
    --dataset ${3} \
    --eval_subsample_ratio ${4} \
    --eval_noise_type ${5} \
    --eval_noise_gamma ${6} \
    --e_task ${7} \
    --optimize_hyperparams ${8} \
    --hyper_opt_eval_fraction ${9} \
    --sorting_metric  ${10} \
    --beam_size  ${11} \
    --reload_scores_path ${12} \
    --optimize_constants ${13} \
    --optimize_constants_init_random ${14} \
    --forecasting_window_length ${15} \
    --y0_generalization_delta ${16} \
    --continue_fitting ${17}