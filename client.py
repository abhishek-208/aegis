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

def apply_attack(weights, global_weights, attack_type, scale_factor=1.0):
    """Corrupts a set of model weights based on the specified attack type."""
    if attack_type == 'none' or attack_type == 'label_flip':
        return weights
    
    corrupted_weights = OrderedDict()
    
    if attack_type == 'sign_flip':
        for key, tensor in weights.items():
            global_tensor = global_weights[key].to(tensor.device)
            delta = tensor - global_tensor
            # Flip the delta and re-apply back to the global weights
            corrupted_weights[key] = global_tensor - (scale_factor * delta)
        return corrupted_weights
        
    elif attack_type == 'additive_noise':
        # Mean Shift Attack
        for key, tensor in weights.items():
            
            # We use tensor.device to match the device of the weights (which is CPU).
            
            noise = torch.ones_like(tensor, device=tensor.device) * config.ATTACK_NOISE_STD
            corrupted_weights[key] = tensor + noise
        return corrupted_weights
        
    elif attack_type == 'orthogonal':
        # Orthogonal Scale Attack:
        # Add a noise vector that is orthogonal to the current weights.
        for key, tensor in weights.items():
            noise = torch.randn_like(tensor, device=tensor.device)
            
            # Gram-Schmidt Orthogonalization: v = n - proj_u(n)
            # proj = (n . u) / (u . u) * u
            dot = torch.sum(noise * tensor)
            norm_sq = torch.sum(tensor * tensor)
            proj = (dot / (norm_sq + 1e-9)) * tensor
            orth_noise = noise - proj
            
            # Scale to match the weight magnitude (Strong attack)
            current_norm = torch.norm(tensor)
            orth_norm = torch.norm(orth_noise)
            if orth_norm > 1e-9:
                orth_noise = orth_noise * (current_norm / orth_norm)

            corrupted_weights[key] = tensor + orth_noise
        return corrupted_weights

    else:
        # For volume_spam, we don't change weights here (except implicit label_flip in train)
        if attack_type == 'volume_spam':
            return weights
        # Only raise error if truly unknown
        # raise ValueError(f"Unknown attack type: {attack_type}")
        return weights

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
                
                # --- Label Flipping Logic ---
                # Used for 'label_flip' AND 'volume_spam' (Volume spam is just label flip + huge count)
                if is_byzantine and (attack_type == 'label_flip' or attack_type == 'volume_spam'):
                    # Shift labels by 1 (target mod 10)
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
            corrupted_weights = apply_attack(local_weights, global_model_state_dict, attack_type)
            
            # --- Volume Spam Logic ---
            if attack_type == 'volume_spam':
                # Report a massively inflated data size (1 Billion)
                # Aegis clips this to 2.0 * avg, maximizing our weight.
                return corrupted_weights, 1_000_000_000
            
            return corrupted_weights, len(self.dataloader.dataset)
        else:
            return local_weights, len(self.dataloader.dataset)