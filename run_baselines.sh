#!/bin/bash

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate symbolicregression39

MODELS=(
    "sindy"
    "sindy_save"
    "sindy_poly3"
    "sindy_poly6"
    "sindy_poly10"
    "sindy_full"
    # "afp"
    # "feafp"
    "ffx"
    # "eplex"
    # "ehc"
    # "proged"
    # "pysr"
)

for subsample_ratio in "0" "0.25" "0.5";
do
    for eval_noise_gamma in "0" "0.001" "0.01";
    do
        for model in "${MODELS[@]}";
        do
            echo "Evaluting ${model} with subsample_ratio=${subsample_ratio} and eval_noise_gamma=${eval_noise_gamma}"
            python run_baselines.py \
                --baseline_model="${model}" \
                --subsample_ratio="${subsample_ratio}" \
                --eval_noise_gamma="${eval_noise_gamma}"
        done
    done
done