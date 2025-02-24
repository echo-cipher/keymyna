'''
Hyperparameter search automation for a single GPU machine.
If you have access to a HPC system that uses SLURM, you can use hp_slurm.py to parallelize 
this search on multiple GPUs.

Pass in default arguments as command-line arguments.
'''

import json
import os
import copy
import itertools

from train import parse_args, main

# hyperparameter grid
GRID = {
    'batch_size': [32, 64, 128, 256, 512],
    'learning_rate': [1e-4, 3e-4, 1e-3, 3e-3],
    'weight_decay': [1e-5, 1e-4, 1e-3, 1e-2],
    'mixup': [None, (2, 5)]
}

def run_grid_search():
    # parse the 'base' args from command line (like --embeds_file, etc.)
    base_args = parse_args()

    if base_args.checkpoint_path is not None:
        os.makedirs(base_args.checkpoint_path, exist_ok=True)

    param_keys = list(GRID.keys())
    param_values = list(GRID.values())
    param_combinations = list(itertools.product(*param_values))

    results = []
    total_runs = len(param_combinations)
    
    for run_idx, param_set in enumerate(param_combinations, start=1):
        param_dict = dict(zip(param_keys, param_set))

        print(f'\nStarting grid search run {run_idx}/{total_runs}')
        print(f'Hyperparams: {param_dict}\n')

        # make a fresh copy of the base arguments
        new_args = copy.deepcopy(base_args)
        for param, value in param_dict.items():
            setattr(new_args, param, value)

        best_metrics, best_primary = main(new_args)

        # store the run results
        result = {
            **param_dict,
            'best_primary': best_primary,
            'best_metrics': best_metrics
        }
        results.append(result)

        # save results to JSON
        with open('hp_results.json', 'w') as f:
            json.dump(results, f, indent=4)

        print(f'Finished run {run_idx}. Best Primary Metric: {best_primary:.4f}\n')

    print('Grid search complete! All results have been saved to hp_results.json.')
    return results


if __name__ == '__main__':
    run_grid_search()
