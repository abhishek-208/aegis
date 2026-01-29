"""
Defines the Client class for the Federated Learning simulation.
"""

import torch
import torch.optim as optim
import torch.nn as nn
from collections import OrderedDict

import config
from model import get_model

# --- === Byzantine Attack Implementation === ---

def apply_attack(weights, attack_type):
    """Corrupts a set of model weights based on the specified attack type."""
    if attack_type == 'none' or attack_type == 'label_flip':
        return weights
    
    corrupted_weights = OrderedDict()
    
    if attack_type == 'sign_flip':
        for key, tensor in weights.items():
            corrupted_weights[key] = tensor * -1.0
        return corrupted_weights
        
    elif attack_type == 'additive_noise':
        # Mean Shift Attack
        for key, tensor in weights.items():
            
            # We use tensor.device to match the device of the weights (which is CPU).
            
            noise = torch.ones_like(tensor, device=tensor.device) * config.ATTACK_NOISE_STD
            corrupted_weights[key] = tensor + noise
        return corrupted_weights
        
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

# --- === Client Class Definition === ---

class Client:
    """Represents a single client in the FL system."""
    
    def __init__(self, client_id, dataloader):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = config.DEVICE

    def train(self, global_model_state_dict, is_byzantine=False, attack_type='none', force_device=None):
        """Performs one round of local training."""
        
        # Determine which device to use for this specific training run
        train_device = force_device if force_device else self.device
        
        # --- Step 1: Setup ---
        model = get_model().to(train_device)
        model.load_state_dict(global_model_state_dict)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(), 
            lr=config.LEARNING_RATE, 
            momentum=config.MOMENTUM
        )
        
        # --- Step 2: Local Training ---
        model.train()
        for _ in range(config.LOCAL_EPOCHS):
            for data, target in self.dataloader:
                data, target = data.to(train_device), target.to(train_device)
                
                # --- Label Flipping Attack (Stealth) ---
                if is_byzantine and attack_type == 'label_flip':
                    # Shift labels by 1 (target mod 10)
                    # This makes the model learn incorrect class mappings
                    target = (target + 1) % 10
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # Always return weights on CPU to avoid pickling/CUDA issues during aggregation
        local_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        
        # --- Step 3: Apply Attack (if Byzantine) ---
        if is_byzantine:
            corrupted_weights = apply_attack(local_weights, attack_type)
            return corrupted_weights, len(self.dataloader.dataset)
        else:
            return local_weights, len(self.dataloader.dataset)