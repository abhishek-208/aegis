"""
Data loading and partitioning utilities.
"""
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import config as config

def load_data():
    """
    Downloads and transforms data.
    
    This function handles the downloading and preprocessing of MNIST or CIFAR10 datasets.
    It applies standard normalization transforms:
    - MNIST: Mean (0.1307,), Std (0.3081,)
    - CIFAR10: Mean (0.4914, 0.4822, 0.4465), Std (0.2023, 0.1994, 0.2010)
    
    Returns:
        train_dataset (torchvision.datasets): The training dataset object.
        test_dataset (torchvision.datasets): The testing dataset object.
    """
    print(f"\n[Data] Loading {config.DATASET_NAME} dataset...")
    
    if config.DATASET_NAME == 'MNIST':
        # Standard MNIST normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
    elif config.DATASET_NAME == 'CIFAR10':
        # Standard CIFAR-10 normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown DATASET_NAME: {config.DATASET_NAME}")
    
    return train_dataset, test_dataset

def partition_data(train_dataset):
    """
    Partitions data among clients based on the strategy defined in config.DATA_SPLIT_TYPE.
    
    Strategies:
    1. BALANCED_IID: Randomly assigns equal number of samples to all clients.
    2. UNBALANCED_IID: Uses Dirichlet distribution to create data imbalance (different sizes per client).
    3. NON_IID: Uses Label Sharding (Pathological Non-IID) where each client sees only a few classes.
    
    Args:
        train_dataset (torchvision.datasets): The full training dataset.
        
    Returns:
        list[DataLoader]: A list of DataLoaders, one per client.
    """
    num_samples = len(train_dataset)
    
    # Uses the specific config variable for thread count
    loader_args = {
        'batch_size': config.BATCH_SIZE,
        'shuffle': True,
        'num_workers': config.DATALOADER_WORKERS,
        'pin_memory': True
    }

    client_dataloaders = []

    if config.DATA_SPLIT_TYPE == 'BALANCED_IID':
        # --- Strategy 1: BALANCED IID ---
        # Simply shuffle all indices and slice them equally.
        print(f"  > Using {config.DATA_SPLIT_TYPE} split.")
        samples_per_client = num_samples // config.NUM_CLIENTS
        indices = list(range(num_samples))
        np.random.shuffle(indices)
        
        for i in range(config.NUM_CLIENTS):
            start = i * samples_per_client
            end = (i + 1) * samples_per_client
            subset = Subset(train_dataset, indices[start:end])
            client_dataloaders.append(DataLoader(subset, **loader_args))

    elif config.DATA_SPLIT_TYPE == 'UNBALANCED_IID':
        # --- Strategy 2: UNBALANCED IID (Dirichlet) ---
        # Generate random proportions using Dirichlet distribution.
        # Check config.DIRICHLET_ALPHA controls the degree of imbalance.
        print(f"  > Using {config.DATA_SPLIT_TYPE} split (Alpha={config.DIRICHLET_ALPHA}).")
        proportions = np.random.dirichlet(np.repeat(config.DIRICHLET_ALPHA, config.NUM_CLIENTS))
        client_sizes = (proportions * num_samples).astype(int)
        
        # Adjust last client to catch any rounding errors in total sum
        client_sizes[-1] = num_samples - np.sum(client_sizes[:-1])
        np.random.shuffle(client_sizes)
        
        indices = list(range(num_samples))
        np.random.shuffle(indices)
        
        current_idx = 0
        for size in client_sizes:
            if size == 0: continue # Skip empty clients if alpha is very extreme
            subset = Subset(train_dataset, indices[current_idx : current_idx + size])
            client_dataloaders.append(DataLoader(subset, **loader_args))
            current_idx += size

    elif config.DATA_SPLIT_TYPE == 'NON_IID':
        # --- Strategy 3: NON-IID (Label Sharding) ---
        # Sorts data by label, divides into shards, and assigns specific number of shards to each client.
        # This ensures clients only see a subset of classes (e.g., only 2 classes out of 10).
        print(f"  > Using {config.DATA_SPLIT_TYPE} split ({config.SHARDS_PER_CLIENT} shards/client).")
        
        # 1. Get targets/labels
        if hasattr(train_dataset, 'targets'):
            labels = torch.tensor(train_dataset.targets)
        elif hasattr(train_dataset, 'train_labels'):
            labels = torch.tensor(train_dataset.train_labels)
        else:
             labels = torch.tensor([y for _, y in train_dataset])

        # 2. Sort indices by label to group classes together
        sorted_indices = torch.argsort(labels).tolist()
        
        # 3. Create shards
        num_shards = config.NUM_CLIENTS * config.SHARDS_PER_CLIENT
        samples_per_shard = num_samples // num_shards
        
        shard_indices = list(range(num_shards))
        np.random.shuffle(shard_indices) # Randomly assign shards to clients
        
        # 4. Assign shards to clients
        for i in range(config.NUM_CLIENTS):
            # Each client gets 'SHARDS_PER_CLIENT' shards
            shards = shard_indices[i * config.SHARDS_PER_CLIENT : (i + 1) * config.SHARDS_PER_CLIENT]
            client_indices = []
            for shard_id in shards:
                start = shard_id * samples_per_shard
                end = start + samples_per_shard
                client_indices.extend(sorted_indices[start:end])
            
            subset = Subset(train_dataset, client_indices)
            client_dataloaders.append(DataLoader(subset, **loader_args))

    print(f"  > Created {len(client_dataloaders)} client partitions.")
    return client_dataloaders

def get_test_dataloader(test_dataset):
    return DataLoader(
        test_dataset, 
        batch_size=1024, 
        shuffle=False,
        num_workers=config.DATALOADER_WORKERS, 
        pin_memory=True
    )