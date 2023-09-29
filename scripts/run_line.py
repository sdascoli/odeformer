import subprocess
from subprocess import DEVNULL
import os
import re
from time import sleep
import itertools
from pathlib import Path
import shutil
from distutils import dir_util
user = os.getlogin()

exp_folder = 'scale'

#dump_path = f'/home/{user}/odeformer/experiments'
dump_path = f'/scratch/{user}/odeformer/experiments'
Path(dump_path).mkdir(exist_ok=True)

extra_args = {
    'use_wandb':True,
    'collate_queue_size': 1000,
    #'n_steps_per_epoch':1000,
    'print_freq': 30,
    'ode_integrator':'solve_ivp',
    'num_workers':1,
    'tokens_per_batch':10000,
    #'min_dimension':1,
    #'max_dimension':2,
    #'sign_as_token':True,
    'reload_data':dump_path + "/datagen/datagen_use_sympy_True",
    }

grid = {
    "tokens_per_batch":[10000,20000],
    "n_enc_layers":[4,8],
    "n_dec_layers":[8,16],
    "emb_emb_dim":[256,512],
    "n_emb_hidden_layers":[0,1],
    "attention_dropout":[0.0,0.1],
}

def get_free_gpus():
    output = subprocess.check_output("nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader", shell=True)
    free_memory = [int(x) for x in output.decode().strip().split('\n')]
    free_gpus = [i for i, memory in enumerate(free_memory) if memory > 15000]  # Change the threshold based on your needs
    free_gpus = sorted(free_gpus, key=lambda i: free_memory[i], reverse=True)
    return free_gpus

def get_most_free_gpu():
    output = subprocess.check_output("nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader", shell=True)
    free_memory = [int(x) for x in output.decode().strip().split('\n')]
    most_free = free_memory.index(max(free_memory))
    # set visible devices to the most free gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(most_free)
    return free_gpus

# Get the list of free GPUs
free_gpus = get_free_gpus()
print("Free GPUs: ",free_gpus)
if not free_gpus:
    print("No free GPUs available!")
    exit()

# Path to your PyTorch script
pytorch_script = "train.py"

# Function to run the PyTorch script with a specific learning rate on a specific GPU
def run_experiment(gpu_id, args, logfile):
    env_vars = os.environ.copy()
    env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    command = ["python", pytorch_script]
    for arg, value in args.items():
        command.append(f"--{arg}")
        command.append(str(value))
    with open(logfile, 'a') as f:
        subprocess.Popen(command, env=env_vars, stdout=DEVNULL, stderr=f)

def dict_product(d):
    for param_to_vary, values in d.items():
        # keep fixed parameters
        fixed_params = {p:v[0] for p,v in d.items() if p != param_to_vary}
        for value in values[1:]:
            yield {param_to_vary:value}, fixed_params

for (params, fixed_params) in dict_product(grid):
    if not free_gpus:
        break
    exp_id = 'exp_'+'_'.join(['{}_{}'.format(k,v) for k,v in params.items()])
    params['dump_path'] = dump_path
    params['exp_name'] = exp_folder
    params['exp_id'] = exp_id

    for arg, value in fixed_params.items():
        if arg not in params:
            params[arg] = value

    for arg, value in extra_args.items():
        if arg not in params:
            params[arg] = value

    job_dir = Path(os.path.join(dump_path, exp_folder, exp_id))
    job_dir.parent.mkdir(exist_ok=True)
    job_dir.mkdir(exist_ok=True)

    for f in os.listdir():
        if f.endswith('.py'):
            shutil.copy2(f, job_dir)
    dir_util.copy_tree('odeformer', os.path.join(job_dir,'odeformer'))
    dir_util.copy_tree('invar_datasets', os.path.join(job_dir,'invar_datasets'))
    os.chdir(job_dir)

    logfile = os.path.join(job_dir,'train.log')
    gpu_id = free_gpus.pop(0)
    print(f"Starting experiment {exp_id} on GPU: {gpu_id}")
    run_experiment(gpu_id, params, logfile)
    sleep(1)

print("All experiments started.")
