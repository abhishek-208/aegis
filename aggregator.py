"""
Implements the aggregation logic.

Contains:
1. `fed_avg(updates)`: Standard Federated Averaging.
2. `aegis(updates)`: Our Byzantine-resilient method (Aegis).
3. `cw_med(updates)`: Coordinate-wise Median (Corrected version).
4. `multi_krum(updates, ...)`: Multi-Krum (Corrected version).
"""

import torch
import copy
import math  # <-- ADDED for math.floor
from collections import OrderedDict
from config import DEVICE, RWA_MAD_THRESHOLD, RWA_EPSILON

# --- === HELPER FUNCTIONS FOR Aegis/Krum/CWMed === ---

def _flatten_weights(weights_dict):
    """
    Flattens a model's state_dict into a single 1D tensor.
    """
    return torch.cat([p.flatten() for p in weights_dict.values()]).to(DEVICE)

def _unflatten_weights(flat_tensor, template_dict):
    """
    Un-flattens a 1D tensor back into a model's state_dict.
    """
    new_dict = OrderedDict()
    current_idx = 0
    for key, tensor in template_dict.items():
        num_elements = tensor.numel()
        shape = tensor.shape
        new_dict[key] = flat_tensor[current_idx : current_idx + num_elements].reshape(shape)
        current_idx += num_elements
    return new_dict

# --- ================================== ---


def fed_avg(updates):
    """
    Performs standard Federated Averaging (weighted by data size).
    """
    total_data_points = sum(n_k for _, n_k in updates)
    if not updates:
        return OrderedDict()
        
    template_weights = updates[0][0]
    avg_weights = OrderedDict()
    
    for key in template_weights:
        avg_weights[key] = torch.zeros_like(template_weights[key], device=DEVICE)

    for client_weights, n_k in updates:
        weight = n_k / total_data_points
        for key in client_weights:
            avg_weights[key] += client_weights[key].to(DEVICE) * weight
            
    return avg_weights


def aegis(updates):
    """
    Performs our Byzantine-Resilient Aegis (Aegis).
    Upgraded to handle Sign Flip and Label Flip via Cosine Similarity.
    """
    print("    > Aggregator: Using Robust Weighted Avg (Aegis)...")
    
    if not updates:
        return OrderedDict()
        
    all_flat_weights = []
    data_sizes = []
    template_dict = updates[0][0]
    
    for weights_dict, n_k in updates:
        all_flat_weights.append(_flatten_weights(weights_dict))
        data_sizes.append(n_k)
        
    weights_matrix = torch.stack(all_flat_weights)
    all_data_sizes_tensor = torch.tensor(data_sizes, device=DEVICE, dtype=torch.float32)

    # --- Step 1: Volume Clipping ---
    # Clip n_k to 2.0 * average_data_size
    avg_data_size = torch.mean(all_data_sizes_tensor)
    clipped_data_sizes = torch.clamp(all_data_sizes_tensor, max=2.0 * avg_data_size.item())

    # --- Step 2: Calculate Robust Center and Euclidean Distances ---
    w_median = torch.median(weights_matrix, dim=0).values
    euclidean_distances = torch.norm(weights_matrix - w_median, dim=1)
    
    # --- Step 3: Calculate Cosine Similarity & Penalty ---
    # Cosine Sim between each client update and the median vector
    # epsilon added to denominator to avoid division by zero if median is all zeros
    cos_sim = torch.nn.functional.cosine_similarity(weights_matrix, w_median.unsqueeze(0), dim=1)
    
    # Penalty: 0.0 (perfect alignment) to 2.0 (opposite direction)
    cosine_penalty = 1.0 - cos_sim

    # --- Step 4: Hard Filtering (MAD + Directional) ---
    # A. Euclidean Stats
    s_median = torch.median(euclidean_distances)
    s_mad = torch.median(torch.abs(euclidean_distances - s_median))
    mad_threshold = s_median + (RWA_MAD_THRESHOLD * s_mad)
    
    # B. Filter Logic
    # Reject if Distance > Threshold OR Cosine Similarity < 0 (Opposite direction)
    # Using indices logic
    
    pass_euclidean = euclidean_distances <= mad_threshold
    pass_direction = cos_sim >= 0.0 # Reject negative cosine similarity
    
    approved_mask = pass_euclidean & pass_direction
    approved_indices = torch.where(approved_mask)[0]
    
    if len(approved_indices) == 0:
        print("    > Aegis: All clients discarded as outliers! Skipping round.")
        return None
        
    print(f"    > Aegis: Approved {len(approved_indices)}/{len(updates)} clients (Rejected {len(updates) - len(approved_indices)}).")

    # --- Step 5: Calculate Enhanced Credit Scores ---
    approved_clipped_sizes = clipped_data_sizes[approved_indices]
    approved_distances = euclidean_distances[approved_indices]
    approved_cosine_penalties = cosine_penalty[approved_indices]
    approved_weights_matrix = weights_matrix[approved_indices]

    # Formula: Score = Clipped_Volume / (Euclidean_Distance + (Cosine_Penalty * 10.0) + Epsilon)
    denominator = approved_distances + (approved_cosine_penalties * 10.0) + RWA_EPSILON
    raw_scores = approved_clipped_sizes / denominator
    
    total_score = torch.sum(raw_scores)
    final_scores = (raw_scores / total_score).unsqueeze(1)
    
    # --- Step 6: Aggregate ---
    new_flat_global_model = torch.sum(approved_weights_matrix * final_scores, dim=0)
    new_global_model_dict = _unflatten_weights(new_flat_global_model, template_dict)
    
    return new_global_model_dict

def cw_med(updates):
    """
    Coordinate-wise median aggregator.
    
    print("    > Aggregator: Using Coordinate-wise Median (CWMed)...")

    if not updates:
        return OrderedDict()

    template_dict = updates[0][0]

    # Build matrix: (num_clients, dim)
    all_flat = []
    for weights_dict, n_k in updates:
        flat = _flatten_weights(weights_dict)
        all_flat.append(flat)

    weights_matrix = torch.stack(all_flat)  # shape: (n_clients, dim)

    # Take median along dim=0 -> returns median for each coordinate.
    w_median = torch.median(weights_matrix, dim=0).values

    # Unflatten and return
    new_global_model_dict = _unflatten_weights(w_median, template_dict)
    return new_global_model_dict
    """

def multi_krum(updates, fraction_byzantine, m_selected=None, weighted=False):
    """
    Mutlti Krum
    """
    print("    > Aggregator: Using Multi-Krum...")

    if not updates:
        return OrderedDict()

    n = len(updates)
    # conservative integer number of Byzantines
    f = int(math.floor(n * fraction_byzantine))

    # The number of neighbors to sum in score: r = n - f - 2 (must be >= 1)
    r = max(1, n - f - 2)

    # default m_selected: choose 1 (classic Krum) or up to (n - f - 2)
    if m_selected is None:
        m_selected = max(1, n - f - 2)

    if m_selected > n:
        raise ValueError("m_selected cannot be greater than number of clients")
    if r < 1 or m_selected < 1:
        print(f"    > Krum: Insufficient clients (n={n}, f={f}) to select. Skipping round.")
        return None # Not enough clients to run Krum

    print(f"    > Krum: n={n}, assumed f={f}, sum r={r} neighbors, selecting m={m_selected} clients")

    template_dict = updates[0][0]

    # Build matrix
    all_flat = []
    sample_sizes = []
    for weights_dict, n_k in updates:
        all_flat.append(_flatten_weights(weights_dict))
        sample_sizes.append(n_k)

    weights_matrix = torch.stack(all_flat)  # shape: (n, dim)

    # Compute pairwise squared Euclidean distances.
    distances = torch.cdist(weights_matrix, weights_matrix, p=2.0)
    distances = distances ** 2  # squared distances

    # Exclude self-distance by setting diagonal to infinity
    distances.fill_diagonal_(float('inf'))

    # For each client i, sort distances to others and sum the smallest r of them.
    sorted_distances, _ = torch.sort(distances, dim=1)  # ascending
    top_r = sorted_distances[:, :r]
    scores = torch.sum(top_r, dim=1)  # shape: (n,)

    # Choose m_selected clients with smallest scores
    _, best_indices = torch.topk(scores, k=m_selected, largest=False)
    best_indices = best_indices.tolist()

    print(f"    > Krum: Selected indices (lowest scores): {best_indices}")

    # Aggregate selected weights
    selected_updates = weights_matrix[best_indices]  # shape: (m_selected, dim)

    if weighted:
        # Use provided sample sizes to form weighted average
        print("    > Krum: Using WEIGHTED average.")
        weights = torch.tensor([sample_sizes[i] for i in best_indices], dtype=torch.float32, device=selected_updates.device)
        weights = weights / torch.sum(weights)
        # (m_selected, 1) * (m_selected, dim) -> weighted sum
        new_flat = torch.sum(selected_updates * weights.unsqueeze(1), dim=0)
    else:
        # Unweighted average
        print("    > Krum: Using UNWEIGHTED average.")
        new_flat = torch.mean(selected_updates, dim=0)

    new_global_model_dict = _unflatten_weights(new_flat, template_dict)
    return new_global_model_dict