#!/bin/bash

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate odeformer39

MODELS=(
    # "afp"
    # "feafp"
    # "eplex"
    # "ehc"
    # "proged"
    # "proged_poly"
    # "ffx"
    "pysr"
    "pysr_poly"
    # "sindy"
    # "sindy_all"
    # "sindy_full"
    # "sindy_save"
    # "sindy_poly3"
    # "sindy_poly6"
    # "sindy_poly10"
)

dataset="strogatz"
hyper_opt="True"
eval_noise_type="additive"
baseline_hyper_opt_eval_fraction="0.3"
baseline_to_sympy="True"

for subsample_ratio in "0.0" "0.25" "0.5";
do
    for eval_noise_gamma in "0.0" "0.001" "0.01" "0.02" "0.03" "0.04" "0.05"; #"0" "0.001" "0.01"; #
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