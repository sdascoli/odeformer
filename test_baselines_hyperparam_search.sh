#!/bin/bash

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate symbolicregression39

MODELS=(
    "afp"
    "feafp"
    "ffx"
    "eplex"
    "ehc"
    "proged"
    "pysr"
    "sindy"
    "sindy_save"
    "sindy_poly"
    "sindy_poly3"
)

for model in "${MODELS[@]}";
do
    echo "Evaluting ${model}"
    python test_baselines_hyperparam_search.py --baseline_model="${model}"
done