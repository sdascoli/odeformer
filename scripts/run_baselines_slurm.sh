#!/bin/bash

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate odeformer39

MODELS=(
    # "sindy"
    # "sindy_save"
    # "sindy_poly"
    # "sindy_poly3"
    "afp"
    "feafp"
    #"ffx"
    "eplex"
    "ehc"
    "proged"
    "pysr"
)

for model in "${MODELS[@]}";
do
    sbatch run_baselines_slurm_job.sh "${model}" &
done