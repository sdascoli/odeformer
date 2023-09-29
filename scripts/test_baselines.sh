#!/bin/bash

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate odeformer39

MODELS=(
    "proged"
    "pysr"
    "sindy_poly3"
    "sindy"
    "afp"
    "feafp"
    "eplex"
    "ehc"
    "ffx"
)

for model in "${MODELS[@]}";
do
    echo "Evaluting ${model}"
    python test_baselines.py --baseline_model="${model}"
done