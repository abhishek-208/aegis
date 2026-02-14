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

# --- === Differential Privacy Helpers === ---

def _flatten_weights(weights_dict):
    """Flattens a model's state_dict into a single 1D tensor."""
    return torch.cat([p.flatten() for p in weights_dict.values()])

def _unflatten_weights(flat_tensor, template_dict):
    """Un-flattens a 1D tensor back into a model's state_dict."""
    new_dict = OrderedDict()
    current_idx = 0
    for key, tensor in template_dict.items():
        num_elements = tensor.numel()
        shape = tensor.shape
        new_dict[key] = flat_tensor[current_idx : current_idx + num_elements].reshape(shape)
        current_idx += num_elements
    return new_dict

def clip_updates(new_weights, global_weights):
    """
    Clips the update (new_weights - global_weights) to a maximum L2 norm defined in config.
    Returns the clipped new_weights.
    """
    # 1. Flatten both
    new_flat = _flatten_weights(new_weights)
    global_flat = _flatten_weights(global_weights)
    
    # 2. Ensure they are on the same device (CPU recommended for this step)
    if new_flat.device != global_flat.device:
        global_flat = global_flat.to(new_flat.device)
        
    # 3. Calculate Update Vector
    update_vector = new_flat - global_flat
    
    # 4. Calculate Norm
    total_norm = torch.norm(update_vector)
    
    # 5. Clip if necessary
    if total_norm > config.DP_CLIP_NORM:
        scaling_factor = config.DP_CLIP_NORM / (total_norm + 1e-9)
        update_vector = update_vector * scaling_factor
        
        # 6. Reconstruct new weights
        clipped_flat = global_flat + update_vector
        return _unflatten_weights(clipped_flat, global_weights)
    
    return new_weights


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
            corrupted_weights = apply_attack(local_weights, attack_type)
            
            # --- Volume Spam Logic ---
            if attack_type == 'volume_spam':
                # Report a massively inflated data size (1 Billion)
                # Aegis clips this to 2.0 * avg, maximizing our weight.
                return corrupted_weights, 1_000_000_000
            
            final_weights = corrupted_weights
        else:
            final_weights = local_weights
            
        # --- Step 4: Differential Privacy Clipping ---
        if config.DP_ENABLED:
            # Clip the update (relative to global_model)
            # Note: We pass global_weights_cpu because local_weights are on CPU
            # global_model_state_dict passed to train() might be on GPU or CPU. 
            # We ensure consistency inside clip_updates.
            
            # Make sure global reference is on CPU for the math
            global_cpu = {k: v.cpu() for k, v in global_model_state_dict.items()}
            final_weights = clip_updates(final_weights, global_cpu)
            
            # --- Local DP: Add Noise HERE if enabled ---
            if getattr(config, 'DP_MODE', 'central') == 'local':
                 # Local DP Noise
                 # Sensitivity = 2 * C (since we clipped update to C, max distance is 2C? 
                 # Actually, commonly we just clip norm to C, so sensitivity is C for sum, but for local release...
                 # Standard Local DP (Gaussian): sigma = sqrt(2 * log(1.25/delta)) / epsilon * Sensitivity
                 # Here we use the config parameters. 
                 # For simplicity, we use the same DP_NOISE_MULTIPLIER as "sigma relative to C".
                 
                 noise_std = config.DP_CLIP_NORM * config.DP_NOISE_MULTIPLIER
                 
                 for key in final_weights:
                     noise = torch.normal(
                         mean=0.0,
                         std=noise_std,
                         size=final_weights[key].shape,
                         device=final_weights[key].device
                     )
                     final_weights[key] += noise

            
        return final_weights, len(self.dataloader.dataset)
