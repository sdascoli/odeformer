#!/bin/bash

hostname=$(hostname)
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate symbolicregression39

export SLURM_NTASKS=1
export SLURM_LOCALID=0

MODELS=(
    # "proged"
    # "proged_poly"
    # "pysr"
    # "pysr_poly"
    # "afp"
    # "feafp"
    # "eplex"
    # "ehc"
    # "ffx"
    # "sindy_all"
    # "sindy_esc"
    # "sindy_poly"
    "odeformer"
)

# dataset="strogatz"
dataset="strogatz_extended"
# dataset="oscillators"

# eval_noise_type="additive"
eval_noise_type="adaptive_additive"

evaluation_task="interpolation"
# evaluation_task="debug6_forecasting"
# evaluation_task="y0_generalization"

forecasting_window_length="10"

if [[ "${evaluation_task}" == "y0_generalization"  ]]; then
    # y0_generalization_deltas=(
    #     "-1.2" "-1.4" "-1.6" "-1.8" "-2.0" 
    #     "-0.2" "-0.4" "-0.6" "-0.8" "-1.0" 
    #     "0.0" "0.2" "0.4" "0.6" "0.8" "1.0" 
    #     "1.2" "1.4" "1.6" "1.8" "2.0" 
    # )
    # y0_generalization_deltas=(
    #     "-2.0" 
    #     "-1.0" 
    #     "0.0" "0.8" "1.0" 
    #     "1.4" "2.0" 
    # )
    # y0_generalization_deltas=(
    #     "-1.5" "1.5"
    # )
    y0_generalization_deltas=(
        "None"
    )
else
    y0_generalization_deltas=("0.0")
fi

continue_fitting="True"

optimize_constants="False"
optimize_constants_init_random="False"

hyper_opt="True"
baseline_hyper_opt_eval_fraction="0.3"
sorting_metric="r2"

for y0_generalization_delta in "${y0_generalization_deltas[@]}";
do

for model in "${MODELS[@]}";
do
    for eval_subsample_ratio in "0.0" # "0.5" #"0.25";
    do
        for eval_noise_gamma in "0.04" #"0.0" "0.01" "0.02" "0.03" "0.04" "0.05";
        do
            # some model specific settings
            if [[ "odeformer" == *"${model}"* ]]; then
                beam_sizes=(50 100) # (1 10 50 100)
            else
                beam_sizes=(1)
            fi

            for beam_size in "${beam_sizes[@]}";
            do
                if [[ "${evaluation_task}" == *"forecasting"* ]]; then
                    job_dir="experiments_paper/${model}/${dataset}/hyper_opt_${hyper_opt}/baseline_hyper_opt_eval_fraction_${baseline_hyper_opt_eval_fraction}/eval_subsample_ratio_${eval_subsample_ratio}/eval_noise_type_${eval_noise_type}/eval_gamma_noise_${eval_noise_gamma}/${evaluation_task}_${forecasting_window_length}/beam_size_${beam_size}"
                elif [[ "${evaluation_task}" == *"y0_generalization"* ]]; then
                    job_dir="experiments_paper/${model}/${dataset}/hyper_opt_${hyper_opt}/baseline_hyper_opt_eval_fraction_${baseline_hyper_opt_eval_fraction}/eval_subsample_ratio_${eval_subsample_ratio}/eval_noise_type_${eval_noise_type}/eval_gamma_noise_${eval_noise_gamma}/${evaluation_task}_${y0_generalization_delta}/beam_size_${beam_size}"
                else
                    job_dir="experiments_paper/${model}/${dataset}/hyper_opt_${hyper_opt}/baseline_hyper_opt_eval_fraction_${baseline_hyper_opt_eval_fraction}/eval_subsample_ratio_${eval_subsample_ratio}/eval_noise_type_${eval_noise_type}/eval_gamma_noise_${eval_noise_gamma}/${evaluation_task}/beam_size_${beam_size}"
                fi
                job_dir_reload="experiments_paper/${model}/${dataset}/hyper_opt_${hyper_opt}/baseline_hyper_opt_eval_fraction_${baseline_hyper_opt_eval_fraction}/eval_subsample_ratio_${eval_subsample_ratio}/eval_noise_type_${eval_noise_type}/eval_gamma_noise_${eval_noise_gamma}/interpolation/beam_size_${beam_size}"
                job_name="${model}_${dataset}_${eval_subsample_ratio}_${eval_noise_type}_${eval_noise_gamma}_${beam_size}"

                if [[ "${optimize_constants}" == "True" ]]; then
                    if [[ "${optimize_constants_init_random}" == "True" ]]; then
                        job_dir="${job_dir}/optimize_constants/init_random"
                        job_name="${job_name}_optimize_constants_init_random"
                        if [[ "${evaluation_task}" == *"forecasting"* ]] || [[ "${evaluation_task}" == *"y0_generalization"* ]]; then
                            job_dir_reload="${job_dir_reload}/optimize_constants/init_random"
                        fi
                    else
                        job_dir="${job_dir}/optimize_constants/init_predicted"
                        job_name=="${job_name}_optimize_constants_init_predicted"
                        if [[ "${evaluation_task}" == *"forecasting"* ]] || [[ "${evaluation_task}" == *"y0_generalization"* ]]; then
                            job_dir_reload="${job_dir_reload}/optimize_constants/init_predicted"
                        fi
                    fi
                fi

                # some cluster specific settings
                if [[ "${hostname}" == *"juwels"* ]]; then
                    base_dir="/p/project/hai_microbio/sb/repos/odeformer"
                    model_dir="${base_dir}/${job_dir}"
                    #echo "hostname: ${hostname}"
                    #echo "pwd: ${PWD}"
                    #echo "model_dir: ${model_dir}"
                    mkdir -p "${model_dir}"

                    path_final_scores="${base_dir}/${job_dir}"
                    if [[ "${dataset}" == "strogatz" ]]; then
                        path_final_scores="${path_final_scores}/eval_pmlb.pkl"
                    elif [[ "${dataset}" == "strogatz_extended" ]]; then
                        path_final_scores="${path_final_scores}/eval_strogatz_extended.json.pkl"
                    elif [[ "${dataset}" == "oscillators" ]]; then
                        path_final_scores="${path_final_scores}/eval_invar_datasets.pkl"
                    else
                        echo "Unknown dataset ${dataset}"
                    fi

                    if [[ "${continue_fitting}" == "True" ]] || [[ "${optimize_constants}" == "True" ]] || [[ "${evaluation_task}" == *"debug"* ]] || [[ "${evaluation_task}" == *"forecasting"* ]] || [[ "${evaluation_task}" == *"y0_generalization"* ]]; then
                        reload_scores_path="${base_dir}/${job_dir_reload}"
                        if [[ "${dataset}" == "strogatz" ]]; then
                            reload_scores_path="${reload_scores_path}/eval_pmlb.pkl"
                        elif [[ "${dataset}" == "strogatz_extended" ]]; then
                            
                            if test -f "${reload_scores_path}/eval_strogatz_extended.json.pkl"; then
                                # use final results
                                reload_scores_path="${reload_scores_path}/eval_strogatz_extended.json.pkl"
                            else 
                                echo "using intermediate results"
                                # use intermediate results
                                reload_scores_path="${reload_scores_path}/eval_strogatz_extended.json.pkl_intermediate.pkl"
                            fi

                        elif [[ "${dataset}" == "oscillators" ]]; then
                            reload_scores_path="${reload_scores_path}/eval_invar_datasets.pkl.pkl"
                        else
                            echo "Unknown dataset ${dataset}"
                        fi

                        if test -f "${reload_scores_path}"; then 
                            true
                            # pass
                        else
                            echo "Reload scores not found: ${reload_scores_path}"
                            continue
                        fi

                    else
                        reload_scores_path="None"
                    fi

                    if test -f "${path_final_scores}"; then
                        echo "skipping ${path_final_scores}" 
                        continue
                    else
                        echo "did not find ${path_final_scores}"
                    fi

                    sbatch \
                        --output "${model_dir}"/log_output_%j.out \
                        --error "${model_dir}"/log_error_%j.err \
                        --mem "0" \
                        --time "0-01:00:00" \
                        --cpus-per-task "48" \
                        --job-name "${job_name}" \
                        --account "hai_symsys" \
                        --partition "develbooster" \
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
                            "${reload_scores_path}" \
                            "${optimize_constants}" \
                            "${optimize_constants_init_random}" \
                            "${forecasting_window_length}" \
                            "${y0_generalization_delta}" \
                            "${continue_fitting}"

                elif [[ "${evaluation_task}" == *"debug"* ]] || [[ "${hostname}" == *"hpc-submit"* ]] || [[ "${hostname}" == *"hpc-build01.scidom.de"* ]]; then
                    base_dir="/home/haicu/soeren.becker/repos/odeformer"
                    model_dir="${base_dir}/${job_dir}"
                    echo "hostname: ${hostname}"
                    echo "pwd: ${PWD}"
                    echo "model_dir: ${model_dir}"
                    mkdir -p "${model_dir}"
                    if [[ "${evaluation_task}" == *"forecasting"* ]] || [[ "${evaluation_task}" == *"y0_generalization"* ]]; then
                        if [[ "${dataset}" == "strogatz" ]]; then
                            reload_scores_path="${base_dir}/${job_dir_reload}/eval_pmlb.pkl"
                            path_final_scores="${base_dir}/${job_dir}/eval_pmlb.pkl"
                        elif [[ "${dataset}" == "strogatz_extended" ]]; then
                            reload_scores_path="${base_dir}/${job_dir_reload}/eval_strogatz_extended.json.pkl.pkl"
                            path_final_scores="${base_dir}/${job_dir}/eval_strogatz_extended.json.pkl.pkl"
                        elif [[ "${dataset}" == "oscillators" ]]; then
                            reload_scores_path="${base_dir}/${job_dir_reload}/eval_invar_datasets.pkl.pkl"
                            path_final_scores="${base_dir}/${job_dir}/eval_invar_datasets.pkl.pkl"
                        else
                            echo "Unknown dataset ${dataset}"
                        fi
                    else
                        reload_scores_path="None"
                    fi
                    if test -f "${path_final_scores}"; then
                        echo "skipping ${path_final_scores}"
                        continue
                    else
                        echo "did not find ${path_final_scores}"
                        continue
                    fi
                    sbatch \
                        --job-name "${job_name}" \
                        -o "${model_dir}/log_output_%j.out" \
                        -e "${model_dir}/log_error_%j.err" \
                        --mem "256G" \
                        --cpus-per-task "32" \
                        --partition "cpu_p" \
                        --qos="cpu_normal" \
                        --nice="10000" \
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
                            "${reload_scores_path}"
                else
                    echo "Unknown hostname: $(hostname)"
                fi

            done
        done
    done
done
done
