"""
Implements the aggregation logic.

Contains:
1. `fed_avg(updates)`: Standard Federated Averaging.
2. `aegis(updates)`: Our Byzantine-resilient method (Aegis).
3. `cw_med(updates)`: Coordinate-wise Median (Corrected version).
4. `multi_krum(updates, ...)`: Multi-Krum (Corrected version).
5. `fools_gold(updates, global_model)`: FoolsGold (Non-IID baseline).
"""

import torch
import copy
import math  # <-- ADDED for math.floor
from collections import OrderedDict
from config import (DEVICE, OUTLIER_SENSITIVITY, RWA_EPSILON,
                     ADAPTIVE_THRESHOLD_ENABLED, K_MAX, K_MIN,
                     WARMUP_ROUNDS, VARIANCE_SENSITIVITY, K_SAFE_FLOOR)

# --- === FOOLSGOLD STATE === ---
# Persistent per-client gradient history across rounds.
# Key: client_id, Value: running sum of flattened deltas (1D tensor).
FOOLSGOLD_HISTORY = {}

def reset_foolsgold_history():
    """Clears FoolsGold history. Call before each experiment."""
    global FOOLSGOLD_HISTORY
    FOOLSGOLD_HISTORY = {}

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


def fed_avg(updates, **kwargs):
    """
    Performs standard Federated Averaging (weighted by data size).
    """
    if not updates:
        return OrderedDict(), None

    total_data_points = sum(n_k for _, _, n_k in updates)
    template_weights = updates[0][1]
    avg_weights = OrderedDict()
    
    for key in template_weights:
        avg_weights[key] = torch.zeros_like(template_weights[key], device=DEVICE, dtype=torch.float32)

    for _, client_weights, n_k in updates:
        weight = n_k / total_data_points
        for key in client_weights:
            avg_weights[key] += client_weights[key].to(DEVICE).float() * weight
    
    # Cast back to original dtypes (e.g., BatchNorm's num_batches_tracked needs Long)
    for key in template_weights:
        avg_weights[key] = avg_weights[key].to(template_weights[key].dtype)
            
    return avg_weights, None # No stats for FedAvg


def aegis(updates, current_round=None):
    """
    Performs our Byzantine-Resilient Aegis (Aegis).
    Upgraded to handle Sign Flip and Label Flip via Cosine Similarity.
    Uses adaptive thresholding (Strategy A+C) when enabled.
    """
    print("    > Aggregator: Aegis...")
    
    if not updates:
        return OrderedDict(), None
        
    all_flat_weights = []   #list that will hold each client's entire model weights flattened into a single 1D tensor
    data_sizes = []         #list that will hold the number of data points each client has (n_k)
    template_dict = updates[0][1]   #template dictionary to store the shape and type of the model weights
    
    for _, weights_dict, n_k in updates:
        all_flat_weights.append(_flatten_weights(weights_dict))
        data_sizes.append(n_k)
        
    weights_matrix = torch.stack(all_flat_weights)
    all_data_sizes_tensor = torch.tensor(data_sizes, device=DEVICE, dtype=torch.float32)

    # --- Step 1: Volume Clipping ---
    # Clip n_k to 2.0 * average_data_size
    avg_data_size = torch.mean(all_data_sizes_tensor)
    clipped_data_sizes = torch.clamp(all_data_sizes_tensor, max=2.0 * avg_data_size.item())

    # --- NEW Step 1.5: Compute Deltas (Updates) ---
    # Aegis works best on gradient updates, not raw weights (especially with DP clipping).
    # We subtract the template_weights (global model from start of round) from each client's weights.
    
    # 1. Flatten the template/global model
    flat_global = _flatten_weights(template_dict)
    
    # 2. Compute Deltas: delta_i = w_i - w_global
    # shape: (n_clients, dim)
    deltas_matrix = weights_matrix - flat_global.unsqueeze(0)

    # --- Step 2: Calculate Robust Center and Euclidean Distances (ON DELTAS) ---
    # Use deltas for median computation
    d_median = torch.median(deltas_matrix, dim=0).values   # Coordinate wise median of updates
    
    # Euclidean distance of each client's UPDATE from the median UPDATE
    euclidean_distances = torch.norm(deltas_matrix - d_median, dim=1)
    
    # --- Step 3: Calculate Cosine Similarity & Penalty (ON DELTAS) ---
    # Cosine Sim between each client UPDATE and the median UPDATE
    cos_sim = torch.nn.functional.cosine_similarity(deltas_matrix, d_median.unsqueeze(0), dim=1)
    


    # Penalty: 0.0 (perfect alignment) to 2.0 (opposite direction)
    cosine_penalty = 1.0 - cos_sim

    # --- Step 4: Hard Filtering (MAD + Directional) ---
    # A. Euclidean Stats
    median_distance = torch.median(euclidean_distances)    # median distance
    distance_mad = torch.median(torch.abs(euclidean_distances - median_distance))    # median absolute deviation

    # Compute adaptive k (Strategy A + C) or use fixed value
    if ADAPTIVE_THRESHOLD_ENABLED and current_round is not None:
        # Strategy A: Round-Based Decay — k_phase decays linearly from K_MAX to K_MIN
        progress = min(current_round / WARMUP_ROUNDS, 1.0)
        k_phase = K_MAX - (K_MAX - K_MIN) * progress

        # Strategy C: Variance-Normalized — self-calibrate based on coefficient of variation
        coeff_of_variation = distance_mad / (median_distance + RWA_EPSILON)
        k = k_phase * (1.0 + VARIANCE_SENSITIVITY * coeff_of_variation.item())
        k = max(k, K_SAFE_FLOOR) # Safety clamp

        print(f"    > Adaptive k: {k:.3f} (phase={k_phase:.2f}, CV={coeff_of_variation:.4f}, round={current_round + 1})")
    else:
        k = OUTLIER_SENSITIVITY
        if ADAPTIVE_THRESHOLD_ENABLED: 
            # If enabled but current_round is None (e.g. first call), we still want to track k
            # But usually it is passed. If not passed, we can't adapt.
             pass 

    rejection_cutoff = median_distance + (k * distance_mad)
    


    # B. Filter Logic
    # Reject if Distance > Threshold OR Cosine Similarity < 0 (Opposite direction)
    # Using indices logic
    
    pass_euclidean = euclidean_distances <= rejection_cutoff
    pass_direction = cos_sim >= 0.0 # Reject negative cosine similarity
    
    approved_mask = pass_euclidean & pass_direction
    approved_indices = torch.where(approved_mask)[0]
    
    if len(approved_indices) == 0:
        print("    > Aegis: All clients discarded as outliers! Skipping round.")
        return None, None
        
    print(f"    > Aegis: Approved {len(approved_indices)}/{len(updates)} clients (Rejected {len(updates) - len(approved_indices)}).")

    # --- Step 5: Calculate Enhanced Credit Scores ---
    approved_clipped_sizes = clipped_data_sizes[approved_indices]
    approved_distances = euclidean_distances[approved_indices]
    approved_cosine_penalties = cosine_penalty[approved_indices]
    # NOTE: We still avg the WEIGHTS, not the deltas, to reconstruct the model.
    # But we use scores derived from deltas.
    approved_weights_matrix = weights_matrix[approved_indices]

    # Formula: Score = Clipped_Volume / (Euclidean_Distance + (Cosine_Penalty * 10.0) + Epsilon)
    denominator = approved_distances + (approved_cosine_penalties * 10.0) + RWA_EPSILON
    raw_scores = approved_clipped_sizes / denominator
    
    total_score = torch.sum(raw_scores)
    final_scores = (raw_scores / total_score).unsqueeze(1)
    
    # --- Step 6: Aggregate ---
    # We aggregate the WEIGHTS using the scores computed from DELTAS.
    new_flat_global_model = torch.sum(approved_weights_matrix * final_scores, dim=0)
    new_global_model_dict = _unflatten_weights(new_flat_global_model, template_dict)
    
    # --- Step 7: Return Stats for Visualization ---
    # We return the original weights_matrix (on CPU if possible to save GPU mem) and approved indices
    stats = {
        "weights_matrix": weights_matrix.cpu().numpy(),
        "approved_indices": approved_indices.cpu().numpy(),
        "adaptive_k": k if ADAPTIVE_THRESHOLD_ENABLED else None # Return k for plotting
    }
    
    return new_global_model_dict, stats

def cw_med(updates, **kwargs):
    print("    > Aggregator: Using Coordinate-wise Median (CWMed)...")

    if not updates:
        return OrderedDict(), None

    template_dict = updates[0][1]

    # Build matrix: (num_clients, dim)
    all_flat = []
    for _, weights_dict, n_k in updates:
        flat = _flatten_weights(weights_dict)
        all_flat.append(flat)

    weights_matrix = torch.stack(all_flat)  # shape: (n_clients, dim)

    # Take median along dim=0 -> returns median for each coordinate.
    w_median = torch.median(weights_matrix, dim=0).values

    # Unflatten and return
    new_global_model_dict = _unflatten_weights(w_median, template_dict)
    
    # CWMed does not return stats for viz currently
    return new_global_model_dict, None 


def multi_krum(updates, fraction_byzantine, m_selected=None, weighted=False):
    """
    Mutlti Krum
    """
    print("    > Aggregator: Using Multi-Krum...")

    if not updates:
        return OrderedDict(), None

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
        return None, None # Not enough clients to run Krum

    print(f"    > Krum: n={n}, assumed f={f}, sum r={r} neighbors, selecting m={m_selected} clients")

    template_dict = updates[0][1]

    # Build matrix
    all_flat = []
    sample_sizes = []
    for _, weights_dict, n_k in updates:
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
    return new_global_model_dict, None


# --- === FOOLSGOLD AGGREGATOR === ---

def fools_gold(updates, global_model=None, **kwargs):
    """
    FoolsGold aggregation (Fung et al., 2020).
    
    Penalizes Sybil/colluding attackers by tracking *historical* gradient
    contributions and computing pairwise cosine similarity. Clients whose
    historical updates look very similar to others receive near-zero weight.
    
    Designed as a baseline for Non-IID scenarios where honest clients
    naturally have diverse gradients.
    
    Args:
        updates: list of (client_id, weights_dict, num_samples).
        global_model: the current global model state_dict (OrderedDict).
    
    Returns:
        (new_global_model_dict, None)
    """
    global FOOLSGOLD_HISTORY
    
    print("    > Aggregator: FoolsGold...")
    
    if not updates:
        return OrderedDict(), None
    
    if global_model is None:
        raise ValueError("FoolsGold requires `global_model` keyword argument.")
    
    n_clients = len(updates)
    template_dict = updates[0][1]
    
    # --- Step 1: Flatten the global model ---
    flat_global = _flatten_weights(global_model)
    
    # --- Step 2: Compute deltas and update history ---
    client_ids = []
    flat_deltas = []       # Current round deltas (for reference)
    weight_dicts = []      # Current round state_dicts (for final aggregation)
    
    for client_id, weights_dict, n_k in updates:
        client_ids.append(client_id)
        weight_dicts.append(weights_dict)
        
        # Delta: client_weights - global_weights
        flat_client = _flatten_weights(weights_dict)
        delta = flat_client - flat_global  # 1D tensor on DEVICE
        flat_deltas.append(delta)
        
        # Accumulate into history
        if client_id not in FOOLSGOLD_HISTORY:
            FOOLSGOLD_HISTORY[client_id] = delta.clone()
        else:
            FOOLSGOLD_HISTORY[client_id] = FOOLSGOLD_HISTORY[client_id].to(DEVICE) + delta
    
    # --- Step 3: Build history matrix for participating clients ---
    # Shape: (n_clients, dim)
    history_matrix = torch.stack([FOOLSGOLD_HISTORY[cid].to(DEVICE) for cid in client_ids])
    
    # --- Step 4: Pairwise cosine similarity ---
    # Normalize each row to unit length
    norms = torch.norm(history_matrix, dim=1, keepdim=True).clamp(min=1e-12)
    normed = history_matrix / norms
    
    # Cosine similarity matrix: (n_clients, n_clients)
    cs_matrix = torch.mm(normed, normed.t())  # O(n^2) pairwise
    
    # --- Step 5: FoolsGold Scoring ---
    # For each client i, find max similarity to any OTHER client j
    # Set diagonal to -inf so self-similarity is ignored
    cs_matrix.fill_diagonal_(-float('inf'))
    v, _ = torch.max(cs_matrix, dim=1)  # v_i = max_{j != i} cos_sim(h_i, h_j)
    
    # Clamp v to [0, 1] — negative similarities mean very different, no penalty
    v = torch.clamp(v, min=0.0, max=1.0)
    
    # Raw penalty: score = 1 - v  (high similarity → low score)
    scores = 1.0 - v
    
    # Log-penalty transformation (standard FoolsGold formula)
    # Amplifies the gap: honest diverse clients keep high scores,
    # while near-identical Sybils get penalized exponentially.
    epsilon = 1e-9
    scores = scores * (torch.log(torch.clamp(scores, min=epsilon) + 1.0) / math.log(2.0))
    
    # --- Step 6: Normalize to sum to 1 ---
    total_score = torch.sum(scores)
    if total_score < epsilon:
        # All clients look identical — fall back to uniform weights
        print("    > FoolsGold: All scores ~0. Using uniform weights.")
        final_weights = torch.ones(n_clients, device=DEVICE) / n_clients
    else:
        final_weights = scores / total_score
    
    # --- Step 7: Weighted aggregation of CURRENT round state_dicts ---
    new_global_dict = OrderedDict()
    for key in template_dict:
        new_global_dict[key] = torch.zeros_like(template_dict[key], device=DEVICE, dtype=torch.float32)
    
    for i, weights_dict in enumerate(weight_dicts):
        w_i = final_weights[i].item()
        for key in weights_dict:
            new_global_dict[key] += weights_dict[key].to(DEVICE).float() * w_i
    
    # Cast back to original dtypes
    for key in template_dict:
        new_global_dict[key] = new_global_dict[key].to(template_dict[key].dtype)
    
    print(f"    > FoolsGold: Weights = {[f'{w:.4f}' for w in final_weights.cpu().tolist()]}")
    
    return new_global_dict, None