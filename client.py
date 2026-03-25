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

def apply_attack(weights, global_weights, attack_type, scale_factor=1.0, prev_global_weights=None):
    """Corrupts a set of model weights based on the specified attack type."""
    if attack_type == 'none' or attack_type == 'label_flip' or attack_type == 'alie':
        return weights
    
    corrupted_weights = OrderedDict()
    
    if attack_type == 'sign_flip' or attack_type == 'sybil':
        # Sybil attack applies model poisoning similarly to sign_flip
        for key, tensor in weights.items():
            global_tensor = global_weights[key].to(tensor.device)
            delta = tensor - global_tensor
            # Flip the delta and re-apply back to the global weights
            corrupted_weights[key] = global_tensor - (scale_factor * delta)
        return corrupted_weights
        
    elif attack_type == 'additive_noise' or attack_type == 'catastrophic_noise':
        # Gaussian Noise on Deltas
        for key, tensor in weights.items():
            global_tensor = global_weights[key].to(tensor.device)
            delta = tensor - global_tensor
            
            # Generate random gaussian noise
            noise = torch.randn_like(delta)
            
            # Determine the scaling factor
            if attack_type == 'catastrophic_noise':
                multiplier = 1000.0  # Massive explosion to stress-test clipping defenses
            else:
                multiplier = getattr(config, 'ATTACK_NOISE_STD', 2.0)  # Stealth mode
            
            # Scale the noise to be a multiple of the honest delta's norm to stay somewhat stealthy
            delta_norm = torch.norm(delta)
            noise_norm = torch.norm(noise)
            
            # Fallback to avoid division by zero if noise matrix is completely zero
            if noise_norm > 1e-9:
                 scaled_noise = noise * (delta_norm / noise_norm) * multiplier
            else:
                 scaled_noise = noise
                 
            corrupted_weights[key] = global_tensor + delta + scaled_noise
            
        return corrupted_weights
        
    elif attack_type == 'orthogonal' or attack_type == 'informed_orthogonal':
        # Delta-Aware Orthogonal Noise Attack
        # 'orthogonal': orthogonalizes against the client's own honest gradient (naive)
        # 'informed_orthogonal': orthogonalizes against the server's previous global delta (omniscient)
        for key, tensor in weights.items():
            global_tensor = global_weights[key].to(tensor.device)   
            
            # Step 1: Calculate the honest update (delta) that standard local training produced
            honest_delta = tensor - global_tensor
            
            # Step 2: Determine which delta to orthogonalize against
            if attack_type == 'informed_orthogonal' and prev_global_weights is not None:
                prev_global_tensor = prev_global_weights[key].to(tensor.device)
                # The server's step direction from the previous round
                target_delta = global_tensor - prev_global_tensor
                # If target_delta is zero (e.g. Round 1), fall back to honest_delta
                if torch.norm(target_delta) < 1e-9:
                    target_delta = honest_delta
            else:
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
        self.prev_global_model_state_dict = None

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
                # Used for 'label_flip' only now. (Volume spam sends honest weights to isolate threshold testing)
                if is_byzantine and attack_type == 'label_flip':
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
            corrupted_weights = apply_attack(local_weights, global_model_state_dict, attack_type, prev_global_weights=self.prev_global_model_state_dict)
            
            # Save the current rounded global model so we can use it to guess the median NEXT round
            self.prev_global_model_state_dict = copy.deepcopy(global_model_state_dict)
            
            # --- Volume Spam Logic ---
            if attack_type == 'volume_spam':
                # Report a massively inflated data size (1 Billion)
                # Aegis clips this to 2.0 * avg, maximizing our weight.
                return self.client_id, corrupted_weights, 1_000_000_000
            
            return self.client_id, corrupted_weights, len(self.dataloader.dataset)
        else:
            # Save the current rounded global model even if honest (just in case they turn Byzantine next round)
            self.prev_global_model_state_dict = copy.deepcopy(global_model_state_dict)
            return self.client_id, local_weights, len(self.dataloader.dataset)