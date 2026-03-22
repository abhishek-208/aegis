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
        # Stealthy Gaussian Noise on Deltas
        for key, tensor in weights.items():
            global_tensor = global_weights[key].to(tensor.device)
            delta = tensor - global_tensor
            
            # Generate random gaussian noise
            noise = torch.randn_like(delta)
            
            # Scale the noise to be a multiple of the honest delta's norm to stay somewhat stealthy
            delta_norm = torch.norm(delta)
            noise_norm = torch.norm(noise)
            
            # Fallback to avoid division by zero if noise matrix is completely zero
            if noise_norm > 1e-9:
                 scaled_noise = noise * (delta_norm / noise_norm) * config.ATTACK_NOISE_STD
            else:
                 scaled_noise = noise
                 
            corrupted_weights[key] = global_tensor + delta + scaled_noise
            
        return corrupted_weights
        
    elif attack_type == 'orthogonal':
        # Delta-Aware Orthogonal Noise Attack
        # The goal is to add severe noise to the update, but ensure the noise is perfectly orthogonal (dot product = 0)
        # to the honest update. This bypasses structural defenses (like Cosine Similarity) that measure update angles.
        for key, tensor in weights.items():
            global_tensor = global_weights[key].to(tensor.device)   
            
            # Step 1: Calculate the honest update (delta) that standard local training produced
            delta = tensor - global_tensor
            
            # Step 2: Generate entirely random Gaussian noise
            noise = torch.randn_like(delta)
            
            # Step 3: Perform Gram-Schmidt Orthogonalization against the DELTA
            # We want to subtract the component of 'noise' that points in the direction of 'delta'.
            # Formula: v_orthogonal = noise - projection_of_noise_onto_delta
            # Projection formula = ((noise dot delta) / (delta dot delta)) * delta
            dot = torch.sum(noise * delta)
            norm_sq = torch.sum(delta * delta)
            
            # Fallback to avoid division by zero if delta happens to be a perfectly zero matrix
            if norm_sq > 1e-9:
                proj = (dot / norm_sq) * delta
            else:
                proj = torch.zeros_like(delta)
                
            # 'orth_noise' is now mathematically guaranteed to be orthogonal to 'delta'
            orth_noise = noise - proj
            
            # Step 4: Scale the orthogonal noise to match the honest delta's magnitude
            # This makes the attack stealthy against magnitude/variance filters (like Median Absolute Deviation)
            # because the total norm of the malicious update stays within expected statistical bounds.
            delta_norm = torch.norm(delta)
            orth_norm = torch.norm(orth_noise)
            
            if orth_norm > 1e-9:
                orth_noise = orth_noise * (delta_norm / orth_norm)

            # Step 5: Construct the final Corrupted Weights
            # W_corrupted = W_global + honest_delta + orthogonal_noise
            # Since our input 'tensor' already equals (W_global + honest_delta), we just add orth_noise.
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

    def train(self, global_model_state_dict, is_byzantine=False, attack_type='none', force_device=None, current_lr=None):
        """Performs one round of local training."""
        
        # Determine which device to use for this specific training run
        train_device = force_device if force_device else self.device
        
        # Fallback to config lr if none provided
        lr_to_use = current_lr if current_lr is not None else config.LEARNING_RATE
        
        # --- Step 1: Setup ---
        model = get_model().to(train_device)
        model.load_state_dict(global_model_state_dict)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(), 
            lr=lr_to_use, 
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
                return self.client_id, corrupted_weights, 1_000_000_000
            
            return self.client_id, corrupted_weights, len(self.dataloader.dataset)
        else:
            return self.client_id, local_weights, len(self.dataloader.dataset)