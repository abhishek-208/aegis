"""
Defines the Client class for the Federated Learning simulation.
"""

import torch
import torch.optim as optim
import torch.nn as nn
from collections import OrderedDict
import copy

import config
from model import get_model

# --- === Byzantine Attack Implementation === ---

def apply_attack(weights, global_weights, attack_type, scale_factor=1.0):
    """Corrupts a set of model weights based on the specified attack type."""
    if attack_type in ['none', 'label_flip', 'alie', 'ipm', 'volume_spam', 'sybil']:
        # none: honest client
        # label_flip: only labels are altered during train(), gradients flow normally
        # alie: orchestrated centrally by the server (uses cross-client stats)
        # ipm: orchestrated centrally by the server (negated scaled mean of honest deltas)
        # volume_spam: only report size is altered in Client.train(); weights are honest
        # sybil: poisoning happens via label-flip during train(); cloning happens in server.py
        return weights
    
    corrupted_weights = OrderedDict()
    
    if attack_type == 'sign_flip':
        # Flip the direction of the honest update
        for key, tensor in weights.items():
            global_tensor = global_weights[key].to(tensor.device)
            delta = tensor - global_tensor
            # Flip the delta and re-apply back to the global weights
            corrupted_weights[key] = global_tensor - (scale_factor * delta)
        return corrupted_weights
        
    elif attack_type in ['additive_noise', 'catastrophic_noise', 'pure_additive_noise']:
        # Gaussian Noise on Deltas
        for key, tensor in weights.items():
            global_tensor = global_weights[key].to(tensor.device)
            delta = tensor - global_tensor
            
            if attack_type == 'pure_additive_noise':
                # Standard formulation: fixed variance N(0, σ²I) independent of delta magnitude
                # Uses ATTACK_NOISE_STD as the fixed standard deviation
                fixed_sigma = getattr(config, 'ATTACK_NOISE_STD', 2.0)
                noise = torch.randn_like(delta) * fixed_sigma
                corrupted_weights[key] = global_tensor + delta + noise
            else:
                # Original stealth formulation: noise magnitude is proportional to honest delta norm
                noise = torch.randn_like(delta)
                
                # Determine the scaling factor
                if attack_type == 'catastrophic_noise':
                    multiplier = 1000.0  # Massive explosion
                else:
                    multiplier = getattr(config, 'ATTACK_NOISE_STD', 2.0)  # Stealth mode
                
                # Scale the noise to be a multiple of the honest delta's norm to stay somewhat stealthy
                delta_norm = torch.norm(delta)
                noise_norm = torch.norm(noise)
                
                if noise_norm > 1e-9:
                     scaled_noise = noise * (delta_norm / noise_norm) * multiplier
                else:
                     scaled_noise = noise
                     
                corrupted_weights[key] = global_tensor + delta + scaled_noise
            
        return corrupted_weights
        
    elif attack_type == 'orthogonal':
        # Delta-Aware Orthogonal Noise Attack
        # 'orthogonal': orthogonalizes against the client's own honest gradient (naive)
        for key, tensor in weights.items():
            global_tensor = global_weights[key].to(tensor.device)   
            
            # Step 1: Calculate the honest update (delta) that standard local training produced
            honest_delta = tensor - global_tensor
            
            # Step 2: Determine which delta to orthogonalize against
            target_delta = honest_delta
                
            # Step 3: Generate entirely random Gaussian noise
            noise = torch.randn_like(honest_delta)
            
            # Step 4: Perform Gram-Schmidt Orthogonalization against the TARGET DELTA
            # We want to subtract the component of 'noise' that points in the direction of 'target_delta'.
            dot = torch.sum(noise * target_delta)
            norm_sq = torch.sum(target_delta * target_delta)
            
            if norm_sq > 1e-9:
                proj = (dot / norm_sq) * target_delta
            else:
                proj = torch.zeros_like(target_delta)
                
            # 'orth_noise' is now mathematically guaranteed to be orthogonal to 'target_delta'
            orth_noise = noise - proj
            
            # Step 5: Scale the orthogonal noise to match the honest delta's magnitude
            # This makes the attack stealthy against magnitude/variance filters.
            delta_norm = torch.norm(honest_delta)
            orth_norm = torch.norm(orth_noise)
            
            if orth_norm > 1e-9:
                orth_noise = orth_noise * (delta_norm / orth_norm)

            # Step 6: Construct the final Corrupted Weights
            # W_corrupted = W_global + honest_delta + orthogonal_noise
            corrupted_weights[key] = tensor + orth_noise
            
        return corrupted_weights

    else:
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
                
                optimizer.zero_grad()
                output = model(data)
                
                # --- Label Flipping Logic ---
                # Used for 'label_flip', 'sybil', and 'volume_spam'. 
                # Volume spam needs poisoned gradients to test the synergy between 
                # volume-bounding and directional filters.
                if is_byzantine and attack_type in ('label_flip', 'sybil', 'volume_spam'):
                    num_classes = output.shape[1]  # Automated detection from logits
                    target = (num_classes - 1) - target
                
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