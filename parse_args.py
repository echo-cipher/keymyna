'''
Module for parsing command-line arguments. 
Made into a separate file to avoid long startup time for hp_slurm.py

TODO: return this to train.py
'''

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='KeyMyna training script')

    # dataset hyperparameters
    parser.add_argument('--embeds_file', type=str, required=True,
                        help='Path to the embeddings file.')
    
    # embedding-space augmentation hyperparameters
    parser.add_argument('--augmentations', type=str,
                        help='Path to embedding space augmentation checkpoint.')
    parser.add_argument('--augmentation_names', type=str, nargs='*', default=None,
                        help='Subset of augmentation names (key values from the augmentation file, if provided) to use.')
    parser.add_argument('--p_aug', type=float, default=0.8,
                        help='Probability value for applying augmentations')
    parser.add_argument('--compose_augs', type=int, default=1,
                        help='Number of random augmentations to compose')
    
    # training hyperparameters
    parser.add_argument('--checkpoint_path', type=str,
                        help='Path to save checkpoints. Default: None')
    parser.add_argument('--resume', type=str,
                        help='Path to load checkpoint. Default: None')
    parser.add_argument('--evaluate', action='store_true', 
                        help='Evaluation mode')
    parser.add_argument('--save_results', type=str, 
                        help='Path to save run results in a JSON file. Default: None')
    parser.add_argument('--arch', type=str, nargs='*', default=['256', 'd0.75', '256'],
                        help='Model architecture. Do not specify input or output size - they will automatically be set as --dim and --dim_out (24), respectively. Use d[0-1] for dropout (e.g., d0.75). ReLU activations will automatically be placed after linear layers. Default: [256 d0.75 256]')
    parser.add_argument('--plot_confusion_matrix', action='store_true',
                        help='Should the script plot confusion matrices after every epoch?')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training. Default: 64')
    parser.add_argument('--dim', type=int, default=384,
                        help='Dimension of embeddings. Default: 384')
    parser.add_argument('--dim_out', type=int, default=24,
                        help='Output dimension. Default: 24')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate for the optimizer. Default: 3e-4')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for the optimizer. Default: 1e-5')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for training. Default: 100')
    parser.add_argument('--mixup', type=float, nargs='*', default=None,
                        help="Mixup strategy as a list of integers (alpha/beta), e.g., '2 5'. Default: None.")
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility. Default: 42')
    
    # TEMPORARY
    parser.add_argument('--results_path', type=str, default=None,
                        help='Save result JSON files (for hp_slurm.py)')
    parser.add_argument('--hp_start_idx', type=int, default=1, 
                        help='Start hyperparameter search at a certain index')
    
    # EXPERIMENTAL
    '''
    parser.add_argument('--scheduler', type=str, default='none', choices=['step', 'cosine', 'none'],
                        help='Learning rate scheduler type. Use "step", "cosine", or "none". Default: none')
    parser.add_argument('--scheduler_step', type=int, default=10,
                        help='StepLR scheduler step size (epochs). Only used if --scheduler is "step".')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1,
                        help='StepLR scheduler decay factor. Only used if --scheduler is "step".')
    parser.add_argument('--scheduler_t_max', type=int, default=50,
                        help='T_max for cosine annealing scheduler. Only used if --scheduler is "cosine".')
    '''

    args = parser.parse_args()
    return args