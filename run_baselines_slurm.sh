#!/bin/bash

hostname=$(hostname)

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate symbolicregression39

export SLURM_NTASKS=1
export SLURM_LOCALID=0

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
    "sindy_poly"
    # "odeformer"
)

dataset="strogatz"
# dataset="strogatz_extended"
# dataset="oscillators"

hyper_opt="True"
eval_noise_type="additive"
baseline_hyper_opt_eval_fraction="0.3"
evaluation_task="forecasting" #"y0_generalization" #
sorting_metric="r2"

reload="False"

for eval_subsample_ratio in "0.0" #"0.25" "0.5";
do
    for eval_noise_gamma in "0.0" #"0.01" "0.02" "0.03" "0.04" "0.05";
    do
        for model in "${MODELS[@]}";
        do
            # some model specific settings
            if [[ "odeformer" == *"${model}"* ]]; then
                beam_sizes=(1) # 2 3 10 20 30 40 50)
            else
                beam_sizes=(1)
            fi

            for beam_size in "${beam_sizes[@]}";
            do
                job_dir="experiments/${model}/${dataset}/hyper_opt_${hyper_opt}/baseline_hyper_opt_eval_fraction_${baseline_hyper_opt_eval_fraction}/eval_subsample_ratio_${eval_subsample_ratio}/eval_noise_type_${eval_noise_type}/eval_gamma_noise_${eval_noise_gamma}/${evaluation_task}/beam_size_${beam_size}"
                job_name="${model}_${dataset}_${eval_subsample_ratio}_${eval_noise_type}_${eval_noise_gamma}_${beam_size}"
                # some cluster specific settings
                if [[ "${hostname}" == *"juwels"* ]]; then
                    base_dir="/p/project/hai_microbio/sb/repos/odeformer"
                    model_dir="${base_dir}/${job_dir}"
                    echo "hostname: ${hostname}"
                    echo "pwd: ${PWD}"
                    echo "model_dir: ${model_dir}"
                    mkdir -p "${model_dir}"
                    if [[ "${reload}" == "True" ]]; then
                        if [[ "${dataset}" == "strogatz" ]]; then
                            reload_scores_path="/p/project/hai_microbio/sb/repos/odeformer/experiments/${model}/${dataset}/hyper_opt_${hyper_opt}/baseline_hyper_opt_eval_fraction_${baseline_hyper_opt_eval_fraction}/eval_subsample_ratio_${eval_subsample_ratio}/eval_noise_type_${eval_noise_type}/eval_gamma_noise_${eval_noise_gamma}/interpolation/beam_size_${beam_size}/eval_pmlb.csv"
                        elif [[ "${dataset}" == "strogatz_extended" ]]; then
                            reload_scores_path="/p/project/hai_microbio/sb/repos/odeformer/experiments/${model}/${dataset}/hyper_opt_${hyper_opt}/baseline_hyper_opt_eval_fraction_${baseline_hyper_opt_eval_fraction}/eval_subsample_ratio_${eval_subsample_ratio}/eval_noise_type_${eval_noise_type}/eval_gamma_noise_${eval_noise_gamma}/interpolation/beam_size_${beam_size}/eval_strogatz_extended.json.csv"
                        elif [[ "${dataset}" == "oscillators" ]]; then
                            reload_scores_path="/p/project/hai_microbio/sb/repos/odeformer/experiments/${model}/${dataset}/hyper_opt_${hyper_opt}/baseline_hyper_opt_eval_fraction_${baseline_hyper_opt_eval_fraction}/eval_subsample_ratio_${eval_subsample_ratio}/eval_noise_type_${eval_noise_type}/eval_gamma_noise_${eval_noise_gamma}/interpolation/beam_size_${beam_size}/eval_invar_datasets.pkl.csv"
                        else
                            echo "Unknown dataset ${dataset}"
                        fi
                    else
                        reload_scores_path="None"
                    fi
                        sbatch \
                            --output "${model_dir}"/log_output_%j.out \
                            --error "${model_dir}"/log_error_%j.err \
                            --mem "0" \
                            --time "0-00:10:00" \
                            --cpus-per-task "48" \
                            --job-name "${job_name}" \
                            --account "hai_microbio" \
                            --partition "booster" \
                            --nodes "1" \
                            --gres "gpu:4" \
                            --ntasks-per-node "1" \
                            run_baselines_slurm_job.sh \
                                "${base_dir}" \
                                "${model}" \
                                "${dataset}" \
                                "${eval_subsample_ratio}" \
                                "${eval_noise_type}" \
                                "${eval_noise_gamma}" \
                                "${evaluation_task}" \
                                "${hyper_opt}" \
                                "${baseline_hyper_opt_eval_fraction}" \
                                "${sorting_metric}" \
                                "${beam_size}" \
                                "${reload_scores_path}"&

                elif [[ "${hostname}" == *"hpc-submit"* ]]; then
                    base_dir="/home/haicu/soeren.becker/repos/odeformer"
                    model_dir="${base_dir}/${job_dir}"
                    echo "hostname: ${hostname}"
                    echo "pwd: ${PWD}"
                    echo "model_dir: ${model_dir}"
                    mkdir -p "${model_dir}"
                    sbatch \
                        --job-name "${job_name}" \
                        -o "${model_dir}/log_output_%j.out" \
                        -e "${model_dir}/log_error_%j.err" \
                        --mem "128G" \
                        --partition "cpu_p" \
                        run_baselines_slurm_job.sh \
                            "${base_dir}" \
                            "${model}" \
                            "${dataset}" \
                            "${eval_subsample_ratio}" \
                            "${eval_noise_type}" \
                            "${eval_noise_gamma}" \
                            "${evaluation_task}" \
                            "${hyper_opt}" \
                            "${baseline_hyper_opt_eval_fraction}" \
                            "${sorting_metric}" \
                            "${beam_size}" &
                fi

            done
        done
    done
done


#--nice "100" \