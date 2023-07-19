#!/bin/bash

export SLURM_LOCALID=0

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate symbolicregression39

MODELS=(
    # "afp"
    # "feafp"
    # "eplex"
    # "ehc"
    # "proged"
    # "proged_poly"
    # "ffx"
    # "pysr"
    # "pysr_poly"
    # "sindy_all"
    # "sindy_esc"
    # "sindy_poly"
    "odeformer"
)

dataset="strogatz"
hyper_opt="True"
eval_noise_type="additive"
baseline_hyper_opt_eval_fraction="0.3"
evaluation_task="interpolation"

for beam_size in 1 2 3 10 20 30 40 50;
do 
    for eval_subsample_ratio in 0.0 0.25 0.5; #"0.0"
    do
        for eval_noise_gamma in 0.0 0.01 0.02 0.03 0.04 0.05; #"0" "0.001" "0.01"; #
        do
            for model in "${MODELS[@]}";
            do
                echo "Evaluting ${model} with subsample_ratio=${eval_subsample_ratio} and eval_noise_gamma=${eval_noise_gamma}"
                python run_baselines.py \
                    --model="${model}" \
                    --dataset "${dataset}" \
                    --eval_subsample_ratio="${eval_subsample_ratio}" \
                    --eval_noise_type "${eval_noise_type}" \
                    --eval_noise_gamma "${eval_noise_gamma}" \
                    --e_task "${evaluation_task}" \
                    --optimize_hyperparams "${hyper_opt}" \
                    --hyper_opt_eval_fraction "${baseline_hyper_opt_eval_fraction}" \
                    --sorting_metric "r2" \
                    --e_task "${evaluation_task}" \
                    --beam_size "${beam_size}"
                    
            done
        done
    done
done