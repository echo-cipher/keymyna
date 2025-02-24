"""
Hyperparameter search automation with SLURM parallelization.
Generates and submits individual SLURM jobs for each hyperparameter combination.

All you really need to provide as a command-line argument to this function is the embeddings file.
"""

import itertools
import os
from parse_args import parse_args


def create_slurm_script(job_name, command):
    """Creates a SLURM script for a single job."""
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={RESULTS_DIR}/{job_name}.out
#SBATCH --error={RESULTS_DIR}/{job_name}.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=6G

source ~/.bashrc
conda activate musicvit

{command}
"""
    return script


def args_to_command(args, override_args):
    """Convert parsed arguments and overrides into a command string."""
    command = "python -u train.py"
    for arg, value in vars(args).items():
        # skip unset arguments or arguments overridden in the grid
        if value is not None and arg not in override_args:
            # boolean flags like --plot_confusion_matrix
            if isinstance(value, bool):
                if value:  
                    command += f" --{arg}"
            
            # list arguments
            elif isinstance(value, list): 
                command += f" --{arg} {' '.join(map(str, value))}"
            else:
                command += f" --{arg} {value}"

    # override arguments from the grid
    for key, value in override_args.items():
        if value is not None:
            # handle tuples (e.g., mixup)
            if isinstance(value, tuple):
                command += f" --{key} {' '.join(map(str, value))}"

            # handle lists (e.g., arch)
            elif isinstance(value, list):
                command += f" --{key} {' '.join(map(str, value))}"
            else:
                command += f" --{key} {value}"
    return command


def generate_hyperparameter_jobs(args):
    """Generate and submit SLURM jobs for each hyperparameter combination."""
    os.makedirs(SLURM_SCRIPTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    param_keys = list(GRID.keys())
    param_values = list(GRID.values())
    param_combinations = list(itertools.product(*param_values))

    for idx, param_set in enumerate(param_combinations, start=1):
        if idx < args.hp_start_idx:
            continue

        # Create unique identifiers and job paths
        param_dict = dict(zip(param_keys, param_set))
        job_name = f'hp_search_{idx}'
        result_file = os.path.join(RESULTS_DIR, f'result_{idx}.json')
        slurm_script_path = os.path.join(SLURM_SCRIPTS_DIR, f'{job_name}.slurm')

        # Skip if result already exists (resumable)
        if os.path.exists(result_file):
            print(f'Skipping job {idx}: Result already exists at {result_file}')
            continue

        # Add save_results to the parameter dictionary
        param_dict["save_results"] = result_file

        # Generate the training command dynamically
        command = args_to_command(args, param_dict)

        # Write the SLURM script
        slurm_script = create_slurm_script(job_name=job_name, command=command)
        with open(slurm_script_path, 'w') as f:
            f.write(slurm_script)

        # Submit the job
        os.system(f'sbatch {slurm_script_path}')
        print(f'Submitted job {idx}: {job_name}')


def get_architectures():
    ''' Returns a list of default architectures for hyperparameter search. '''
    dims = [1024, 2048, 4096, 8192]
    dropouts = [0.75, 0.9, 0.95, 0.99]

    architectures = []

    # single layer with dropout
    for dim in dims:
        for dropout in dropouts:
            architectures.append(f'{dim} d{dropout}')

    # two layers with the same dimension and dropout
    for dim in dims:
        if dim == 8192:
            break
        for dropout in dropouts:
            architectures.append(f'{dim} d{dropout} {dim}')

    '''
    # two layers with decreasing dimension
    for dim in dims:
        for dropout in dropouts:
            architectures.append(f'{dim} {dim // 2}')
    '''

    return architectures


if __name__ == '__main__':
    args = parse_args()

    # Paths to store SLURM scripts and results
    SLURM_SCRIPTS_DIR = 'slurm_scripts'
    RESULTS_DIR = args.results_path

    # hyperparameter grid
    GRID = {
        'batch_size': [64],
        'learning_rate': [3e-4],
        'weight_decay': [1e-4],
        'arch': get_architectures()
    }

    generate_hyperparameter_jobs(args)
    
    
