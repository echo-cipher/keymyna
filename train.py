'''
Self-contained MLP training script. 
'''

import argparse
from collections import defaultdict
import json
import matplotlib.pyplot as plt
from mir_eval.key import weighted_score
import numpy as np
import os
import pickle
import random
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, top_k_accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# TEMPORARY
from parse_args import parse_args


def main(args: argparse.Namespace):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using {device}')
    seed_everything(args.seed)

    model = make_model(args).to(device)
    if args.resume:
        load_model(model, device, args.resume)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model head contains {n_params:,} parameters')

    with open(args.embeds_file, 'rb') as f:
        data = pickle.load(f)

    train_dataset = EmbeddingDataset(
        indices=data['train']['indices'],
        embeddings=data['train']['embeddings'],
        labels=data['train']['labels']
    )

    test_dataset = EmbeddingDataset(
        indices=data['test']['indices'],
        embeddings=data['test']['embeddings'],
        labels=data['test']['labels']
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    # scheduler = make_scheduler(args, optimizer)

    if args.evaluate:
        if not args.resume:
            print(f'Warning: evaluating without a loaded checkpoint!')
        test_metrics, test_primary = test(model, test_loader, criterion, device, 0, args.plot_confusion_matrix)
        print('\n' + '-' * 50)
        print(f'Evaluation results: \n{test_metrics["text"]}')
        print('-' * 50)
        return

    augmentations = None
    if args.augmentations:
        augmentations = get_augmentations(args.augmentations, args.dim, device)
        if args.augmentation_names:
            augmentations = {k: augmentations[k] for k in args.augmentation_names}

    best_metrics = {}
    best_primary = 0
    for epoch in range(args.epochs):
        print(f'Starting epoch {epoch + 1}/{args.epochs}')

        train_metrics, train_primary = train_epoch(model, train_loader, criterion, optimizer, device, args.mixup, augmentations, args.p_aug)
        test_metrics, test_primary = test(model, test_loader, criterion, device, epoch, args.plot_confusion_matrix)

        print(f'\tTrain Metrics: {train_metrics["text"]}, Primary Metric: {train_primary}\n')
        print(f'\tTest Metrics: {test_metrics["text"]}, Primary Metric: {test_primary}\n')

        '''
        if scheduler:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f'\tScheduler updated learning rate to: {current_lr:.6f}')
        '''

        if test_primary > best_primary:
            best_metrics = test_metrics
            best_primary = test_primary

        if args.checkpoint_path:
            os.makedirs(args.checkpoint_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, f'adapter_epoch_{epoch+1}.pth'))

    print(f'Best result: {best_primary}')

    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(best_metrics, f, indent=4)

    return best_metrics, best_primary

'''
def make_scheduler(args: argparse.Namespace, optimizer: optim.Optimizer):
    if args.scheduler == 'step':
        print(f'Using StepLR scheduler: step_size={args.scheduler_step}, gamma={args.scheduler_gamma}')
        return optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    
    elif args.scheduler == 'cosine':
        print(f'Using CosineAnnealingLR scheduler: T_max={args.scheduler_t_max}')
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.scheduler_t_max)
    
    else:
        print('No learning rate scheduler will be used.')
        return None
'''


def load_model(model: nn.Module, device: str, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    return model


def get_augmentations(checkpoint: str, dim: int, device: str):
    state_dicts = torch.load(checkpoint, weights_only=True, map_location=device)
    augmentations = {}
    for aug_name, state_dict in state_dicts.items():
        model = nn.Linear(dim, dim, bias=False)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        augmentations[aug_name] = model

    return augmentations


@torch.no_grad()
def augment(embeds: torch.Tensor, augmentations: dict, p: float):
    if random.random() < p:
        name = random.choice(list(augmentations.keys()))
        return augmentations[name](embeds)
    return embeds


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha=5.0, beta=2.0):
    ''' Compute the mixup data. Return mixed inputs, pairs of targets, and lambda '''
    lam = np.random.beta(alpha, beta) if alpha > 0 else 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = (y, y[index]) if y is not None else (None, None)
    return mixed_x, y_a, y_b, lam


class EmbeddingDataset(Dataset):
    ''' Dataset for precomputed embeddings. '''
    def __init__(self, indices: torch.LongTensor, embeddings: torch.Tensor, labels: list, standardize: bool = False):
        self.indices = indices
        self.embeddings = embeddings
        self.labels = labels
        self.standardize = standardize

        self.standardized_embeddings = torch.tensor(
            StandardScaler().fit_transform(embeddings.cpu().numpy()), dtype=torch.float32
        ).to(embeddings.device)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        embedding = self.embeddings[idx] if not self.standardize else self.standardized_embeddings[idx]
        label = self.labels[actual_idx]
        return actual_idx, embedding, label
    
    
def seed_everything(seed: int):
    random.seed(seed)    
    np.random.seed(seed)    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def update_predictions_targets(all_targets: defaultdict, all_predictions: defaultdict, indices: torch.LongTensor, labels: torch.Tensor, outputs: torch.Tensor):
    indices = indices.cpu().numpy()
    labels = labels.cpu().numpy()
    outputs = outputs.detach().cpu().numpy()

    # Accumulate predictions and store targets
    for idx, label, output in zip(indices, labels, outputs):
        all_predictions[idx].append(output)
        all_targets[idx] = label


def cat_predictions_targets(all_targets: defaultdict, all_predictions: defaultdict):
    all_targets = np.array([
        all_targets[i] for i in range(len(all_targets))
    ])
    all_predictions = np.array([
        np.mean(all_predictions[i], axis=0) for i in range(len(all_predictions))
    ])

    return all_targets, all_predictions


def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device, mixup: tuple, augmentations: dict, p_aug: float):
    ''' Train the classifier/MLP on embeddings for one epoch. '''
    model.train()
    running_loss = 0.0
    all_targets = defaultdict(list)
    all_predictions = defaultdict(list)

    for indices, inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device)

        if augmentations:
            for _ in range(args.compose_augs):
                inputs = augment(inputs, augmentations, p_aug)

        if mixup:
            inputs, y_a, y_b, lam = mixup_data(inputs, labels, mixup[0], mixup[1])

        optimizer.zero_grad()
        outputs = model(inputs)

        if mixup:
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        update_predictions_targets(all_targets, all_predictions, indices, labels, outputs)

    all_targets, all_predictions = cat_predictions_targets(all_targets, all_predictions)

    # Compute metrics
    train_metrics = compute_metrics(all_targets, all_predictions)
    primary_metric = train_metrics.get('weighted_accuracy', -np.inf)

    return train_metrics, primary_metric


def test(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: torch.device, epoch: int, plot_cm: bool):
    model.eval()
    running_loss = 0.0
    all_targets = defaultdict(list)
    all_predictions = defaultdict(list)

    with torch.no_grad():
        for indices, inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            update_predictions_targets(all_targets, all_predictions, indices, labels, outputs)

    all_targets, all_predictions = cat_predictions_targets(all_targets, all_predictions)

    # Compute metrics
    test_metrics = compute_metrics(all_targets, all_predictions)
    primary_metric = test_metrics.get('weighted_accuracy', -np.inf)

    if plot_cm:
        plot_confusion_matrix(all_targets, np.argmax(all_predictions, axis=1), epoch, mode='test', save_path=args.checkpoint_path)

    return test_metrics, primary_metric


def plot_confusion_matrix(y_true, y_pred, epoch, mode, save_path='./'):
    '''
    Plots and saves a confusion matrix as a styled PNG for a paper.
    '''
    labels = list(range(24))
    readable_labels = [
        'C', 'Cm', 'Db', 'Dbm', 'D', 
        'Db', 'Eb', 'Ebm', 'E', 'Em', 
        'F', 'Fm', 'Gb', 'Gbm', 'G', 
        'Gm', 'Ab', 'Abm', 'A', 'Am', 
        'Bb', 'Bbm', 'B', 'Bm'
    ]
    readable_labels = [l.capitalize() for l in [
        'c major', 'c minor', 'db major', 'db minor', 'd major', 
        'd minor', 'eb major', 'eb minor', 'e major', 'e minor', 
        'f major', 'f minor', 'gb major', 'gb minor', 'g major', 
        'g minor', 'ab major', 'ab minor', 'a major', 'a minor', 
        'bb major', 'bb minor', 'b major', 'b minor'
    ]]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=readable_labels, yticklabels=readable_labels)
    plt.title(f'Confusion Matrix - {mode.capitalize()} (Epoch {epoch + 1})', fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('True Labels', fontsize=14)
    plt.tight_layout()

    file_name = f'confusion-matrix-{epoch + 1}-{mode}.png'
    plt.savefig(os.path.join(save_path, file_name))
    plt.close()
    

def compute_metrics(targets: np.ndarray, predictions: np.ndarray):
        top_1_accuracy = accuracy_score(targets, np.argmax(predictions, axis=1))
        top_5_accuracy = top_k_accuracy_score(targets, predictions, k=5, labels=list(range(24)))

        metrics = {
            'top_1_accuracy': top_1_accuracy,
            'top_5_accuracy': top_5_accuracy,
            'text': f'Top-1 Accuracy: {top_1_accuracy:.4f}, Top-5 Accuracy: {top_5_accuracy:.4f}'
        }

        score, fifths, relative, parallel, other = key_detection_accuracy(targets, np.argmax(predictions, axis=1))
        metrics['weighted_accuracy'] = score
        metrics['text'] += f'\n\tWeighted: {score:.4f}\n\tFifth: {fifths:.4f}\n\tRelative: {relative:.4f}\n\tParallel: {parallel:.4f}\n\tOther: {other:.4f}'

        return metrics


def key_detection_accuracy(targets: list, predictions: list):
    keys = [
        'c major', 'c minor', 'db major', 'db minor', 'd major', 
        'd minor', 'eb major', 'eb minor', 'e major', 'e minor', 
        'f major', 'f minor', 'gb major', 'gb minor', 'g major', 
        'g minor', 'ab major', 'ab minor', 'a major', 'a minor', 
        'bb major', 'bb minor', 'b major', 'b minor'
    ]

    targets = [keys[i] for i in targets]
    predictions = [keys[i] for i in predictions]

    weighted_scores = [weighted_score(target, pred) for target, pred in zip(targets, predictions)]
    fifths = sum([score == 0.5 for score in weighted_scores]) / len(targets)
    relative = sum([score == 0.3 for score in weighted_scores]) / len(targets)
    parallel = sum([score == 0.2 for score in weighted_scores]) / len(targets)
    other = sum([score == 0 for score in weighted_scores]) / len(targets)

    score = np.mean(weighted_scores)
    return score, fifths, relative, parallel, other


def make_model(args):
    '''
    Returns a PyTorch Sequential model based on args.arch.
    - The first Linear layer uses input size = args.dim
    - The last Linear layer uses output size = 24
    - Any architecture item 'dX' (e.g. d0.75) is interpreted as Dropout with p=X
    - All other items are interpreted as the number of units in a Linear layer
      followed by a ReLU activation.
    '''
    layers = []
    in_dim = args.dim

    for item in args.arch:
        if item.startswith('d'):
            dropout_p = float(item[1:])
            layers.append(nn.Dropout(dropout_p))
        else:
            # Linear + ReLU
            out_dim = int(item)
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = out_dim

    # final linear layer to output dimension
    layers.append(nn.Linear(in_dim, args.dim_out))

    model = nn.Sequential(*layers)
    return model


if __name__ == '__main__':
    args = parse_args()
    main(args)