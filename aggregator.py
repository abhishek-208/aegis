"""
Implements the aggregation logic.

Contains:
1. `fed_avg(updates)`: Standard Federated Averaging.
2. `aegis(updates)`: Our Byzantine-resilient method (Aegis).
3. `cw_med(updates)`: Coordinate-wise Median (Corrected version).
4. `multi_krum(updates, ...)`: Multi-Krum (Corrected version).
5. `bulyan(updates, fraction_byzantine)`: Bulyan (El Mhamdi et al., 2018).
6. `fools_gold(updates, global_model)`: FoolsGold (Non-IID baseline).
"""

import torch
import copy
import math  # <-- ADDED for math.floor
from collections import OrderedDict
from config import (DEVICE, OUTLIER_SENSITIVITY, RWA_EPSILON,
                     ADAPTIVE_THRESHOLD_ENABLED, K_MAX, K_MIN,
                     WARMUP_ROUNDS, VARIANCE_SENSITIVITY, K_SAFE_FLOOR,
                     FOOLSGOLD_KAPPA)

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


def aegis(updates, current_round=None, 
          ablate_volume_clipping=False, 
          ablate_directional=False, 
          ablate_cosine_penalty=False, 
          ablate_adaptive=False,
          ablate_euclidean_filter=False,
          **kwargs):
    """
    Performs our Byzantine-Resilient Aegis (Aegis).
    Upgraded to handle Sign Flip and Label Flip via Cosine Similarity.
    Uses adaptive thresholding (Strategy A+C) when enabled.
    Ablation parameters allow disabling specific defenses.
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

    # --- Step 1: Volume Bounding / Clipping ---
    # Moved to Step 5 to ensure clipping is robustly calculated over approved clients only.

    # --- NEW Step 1.5: Compute Deltas (Updates) ---
    # Aegis works best on gradient updates, not raw weights (especially with DP clipping).
    # We subtract the template_weights (global model from start of round) from each client's weights.
    
    # 1. Flatten the template/global model
    # Use the actual global_model passed from the server for the start of the round.
    global_model_state = kwargs.get('global_model', None)
    if global_model_state is not None:
        flat_global = _flatten_weights(global_model_state)
    else:
        # Fallback to the first client as a template (structure only, values might be slightly biased)
        # Note: This fallback should ideally not be hit in the main simulation.
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
    if ADAPTIVE_THRESHOLD_ENABLED and current_round is not None and not ablate_adaptive:
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
    
    if not ablate_euclidean_filter:
        pass_euclidean = euclidean_distances <= rejection_cutoff
    else:
        pass_euclidean = torch.ones_like(euclidean_distances, dtype=torch.bool)
    
    if not ablate_directional:
        pass_direction = cos_sim >= 0.0 # Reject negative cosine similarity
    else:
        # Ignore direction
        pass_direction = torch.ones_like(cos_sim, dtype=torch.bool)
    
    approved_mask = pass_euclidean & pass_direction
    approved_indices = torch.where(approved_mask)[0]
    
    if len(approved_indices) == 0:
        print("    > Aegis: All clients discarded as outliers! Skipping round.")
        return None, None
        
    print(f"    > Aegis: Approved {len(approved_indices)}/{len(updates)} clients (Rejected {len(updates) - len(approved_indices)}).")

    # --- Step 5: Calculate Enhanced Credit Scores ---
    approved_data_sizes = all_data_sizes_tensor[approved_indices]
    
    if not ablate_volume_clipping:
        # Robust Volume Clipping: bounding sizes against the MEDIAN of only the approved clients
        robust_median_size = torch.median(approved_data_sizes)
        approved_clipped_sizes = torch.clamp(approved_data_sizes, max=2.0 * robust_median_size.item())
    else:
        approved_clipped_sizes = approved_data_sizes
        
    approved_distances = euclidean_distances[approved_indices]
    approved_cosine_penalties = cosine_penalty[approved_indices]
    # NOTE: We still avg the WEIGHTS, not the deltas, to reconstruct the model.
    # But we use scores derived from deltas.
    approved_weights_matrix = weights_matrix[approved_indices]

    # Formula: Score = Clipped_Volume / (Euclidean_Distance + (Cosine_Penalty * 10.0) + Epsilon)
    if not ablate_cosine_penalty:
        denominator = approved_distances + (approved_cosine_penalties * 10.0) + RWA_EPSILON
    else:
        denominator = approved_distances + RWA_EPSILON
        
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


# --- === BULYAN AGGREGATOR === ---

def bulyan(updates, fraction_byzantine, **kwargs):
    """
    Bulyan aggregation — El Mhamdi et al., 2018.
    "The Hidden Vulnerability of Distributed Learning in Byzantium"

    Algorithm:
      Phase 1 — Iterative Krum:
        Repeat θ = n - 2f times:
          1. Compute pairwise squared Euclidean distances among remaining candidates.
          2. Score each candidate by summing its (n_remaining - f - 2) smallest distances.
          3. Select the candidate with the lowest score → add to selection_set.
          4. Remove it from the candidate pool.

      Phase 2 — Coordinate-wise Trimmed Mean:
        Stack the θ selected gradients.
        For each coordinate d:
          1. Compute the median across the θ clients.
          2. Find the β = θ - 2f clients closest to the median.
          3. Average those β values → final model coordinate.

    Constraint: n >= 4f + 3

    Args:
        updates: list of (client_id, weights_dict, num_samples).
        fraction_byzantine: float in [0, 1), fraction of clients assumed Byzantine.

    Returns:
        (new_global_model_dict, None)
    """
    print("    > Aggregator: Bulyan...")

    if not updates:
        return OrderedDict(), None

    n = len(updates)
    f = int(math.floor(n * fraction_byzantine))

    assert n >= 4 * f + 3, (
        f"Bulyan requires n >= 4f + 3, but got n={n}, f={f} (need n >= {4*f+3})."
    )

    # θ: number of Krum selections (iterations of Phase 1)
    theta = n - 2 * f
    # β: number of clients retained per coordinate in trimmed mean (Phase 2)
    beta = theta - 2 * f

    print(f"    > Bulyan: n={n}, f={f}, θ={theta} (Krum selections), β={beta} (trimmed mean size)")

    template_dict = updates[0][1]

    # --- Build flat weight matrix: shape (n, dim) ---
    all_flat = []
    for _, weights_dict, _ in updates:
        all_flat.append(_flatten_weights(weights_dict))
    weights_matrix = torch.stack(all_flat)  # (n, dim)

    # ---------------------------------------------------------------
    # PHASE 1: Iterative Krum — select θ candidates
    # ---------------------------------------------------------------
    # We maintain indices into the ORIGINAL weights_matrix.
    remaining_indices = list(range(n))   # pool of candidate original indices
    selection_set = []                   # list of 1D tensors (selected gradients)

    for iteration in range(theta):
        n_remaining = len(remaining_indices)
        # Number of nearest neighbours to sum per client in this pool
        # Paper: n - f - 2, but here the pool shrinks; we use pool_size - f - 2
        r = max(1, n_remaining - f - 2)

        # Build sub-matrix from the current pool
        pool_matrix = weights_matrix[remaining_indices]  # (n_remaining, dim)

        # Pairwise squared Euclidean distances (n_remaining × n_remaining)
        dists = torch.cdist(pool_matrix, pool_matrix, p=2.0) ** 2
        dists.fill_diagonal_(float('inf'))

        # Score = sum of r nearest neighbours' distances
        sorted_dists, _ = torch.sort(dists, dim=1)   # ascending
        scores = sorted_dists[:, :r].sum(dim=1)       # (n_remaining,)

        # Pick the local index with the lowest score
        local_best = torch.argmin(scores).item()

        # Map back to original index and record
        original_best = remaining_indices[local_best]
        selection_set.append(weights_matrix[original_best])          # 1D tensor
        remaining_indices.pop(local_best)

    # ---------------------------------------------------------------
    # PHASE 2: Coordinate-wise Trimmed Mean over the θ selected gradients
    # ---------------------------------------------------------------
    # Stack: shape (θ, dim)
    selected_matrix = torch.stack(selection_set)  # (theta, dim)

    # 1. Coordinate-wise median across the θ selected clients: shape (dim,)
    coord_median = torch.median(selected_matrix, dim=0).values  # (dim,)

    # 2. Absolute distance of each client from the median: shape (θ, dim)
    abs_deviations = torch.abs(selected_matrix - coord_median.unsqueeze(0))  # (theta, dim)

    # 3. For each coordinate, find the β clients closest to the median.
    #    Sort deviation along the client dimension (dim=0), take indices of β smallest.
    #    sorted_idx shape: (theta, dim) — each column gives the rank of clients for that coordinate.
    _, sorted_idx = torch.sort(abs_deviations, dim=0)  # ascending along client axis
    beta_idx = sorted_idx[:beta, :]  # (beta, dim) — top-β closest per coordinate

    # 4. Gather the actual values for those β clients per coordinate.
    #    For coordinate d: select selected_matrix[beta_idx[:, d], d]
    #    Fully vectorized via advanced indexing — no loop over parameters.
    beta_values = selected_matrix[beta_idx, torch.arange(selected_matrix.shape[1], device=selected_matrix.device).unsqueeze(0)]
    # beta_values shape: (beta, dim)

    # 5. Mean of the β selected values per coordinate
    new_flat = beta_values.mean(dim=0)  # (dim,)

    new_global_model_dict = _unflatten_weights(new_flat, template_dict)

    print(f"    > Bulyan: Done. Selected {theta} via Krum, trimmed to β={beta} per coordinate.")
    return new_global_model_dict, None


# --- === FOOLSGOLD AGGREGATOR === ---

def fools_gold(updates, global_model=None, **kwargs):
    """
    FoolsGold aggregation — Fung et al., 2020.
    
    Penalizes Sybil/colluding attackers by tracking *historical* gradient
    contributions on **indicative features** (last FC layer) and computing
    pairwise cosine similarity with per-class normalization.
    
    Paper algorithm:
      1. Track history H_i on the last FC layer (indicative features).
      2. Normalize H_i per class (row-wise L2 normalization).
      3. Compute pairwise cosine similarity on normalized, flattened H.
      4. v_i = max_{j≠i} cos_sim(H_i, H_j).
      5. α_i = 1 - v_i.
      6. Logit transform: α_i = κ · (ln(α_i / (1 - α_i)) + 0.5), clamp to [0,1].
      7. Update: w_{t+1} = w_t + (1/n) · Σ α_i · Δ_i  (no sum-to-1 normalization).
    
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
    
    # --- Step 1: Identify the indicative feature layer (last FC layer) ---
    # The paper uses only the output layer for similarity computation.
    # Find the last key ending in '.weight' with a 2D shape (Linear layer).
    indicator_key = None
    for key in template_dict.keys():
        if key.endswith('.weight') and template_dict[key].dim() == 2:
            indicator_key = key
    
    if indicator_key is None:
        raise ValueError("FoolsGold: Could not find a Linear layer for indicative features.")
    
    # --- Step 2: Compute full deltas and update indicative feature history ---
    client_ids = []
    client_deltas = []  # Full model deltas for final aggregation
    
    for client_id, weights_dict, n_k in updates:
        client_ids.append(client_id)
        
        # Full model delta: Δ_i = w_i - w_global (for every layer)
        full_delta = OrderedDict()
        for key in weights_dict:
            full_delta[key] = (weights_dict[key].to(DEVICE).float() -
                               global_model[key].to(DEVICE).float())
        client_deltas.append(full_delta)
        
        # Indicative feature delta: only the last FC layer
        indicator_delta = full_delta[indicator_key]  # Shape: (num_classes, num_features)
        
        # Accumulate into history: H_i += Δ_indicator_i
        if client_id not in FOOLSGOLD_HISTORY:
            FOOLSGOLD_HISTORY[client_id] = indicator_delta.clone()
        else:
            FOOLSGOLD_HISTORY[client_id] = FOOLSGOLD_HISTORY[client_id].to(DEVICE) + indicator_delta
    
    # --- Step 3: Per-class (row-wise) L2 normalization of history ---
    # Each H_i has shape (num_classes, num_features).
    # Normalize each row (class) to unit norm before similarity computation.
    normalized_histories = []
    for cid in client_ids:
        h = FOOLSGOLD_HISTORY[cid].to(DEVICE)  # (num_classes, num_features)
        row_norms = torch.norm(h, dim=1, keepdim=True).clamp(min=1e-12)
        h_normed = h / row_norms
        normalized_histories.append(h_normed.flatten())  # Flatten for cosine sim
    
    history_matrix = torch.stack(normalized_histories)  # (n_clients, num_classes * num_features)
    
    # --- Step 4: Pairwise cosine similarity ---
    # Note: feature selection is a close approximation of the paper's
    # ST-weighted indicative feature procedure (see paper §4.2).
    norms = torch.norm(history_matrix, dim=1, keepdim=True).clamp(min=1e-12)
    normed = history_matrix / norms
    cs_matrix = torch.mm(normed, normed.t())  # (n_clients, n_clients), in [-1, 1]
    cs_matrix = torch.clamp(cs_matrix, min=0.0, max=1.0)  # Keep only positive similarity
    
    # --- Step 5: Pardoning (paper Algorithm 1, lines 8-12) ---
    # For each pair (i, j): if v_i < v_j, rescale cs[i,j] by v_i/v_j.
    # This "pardons" a client that is similar to another but less similar overall —
    # preventing honest clients from being penalized for coincidental resemblance.
    cs_pardoned = cs_matrix.clone()
    # v_i = max_{j≠i} cs[i,j] BEFORE pardoning (used only for the pardon ratio)
    cs_no_diag = cs_matrix.clone()
    cs_no_diag.fill_diagonal_(0.0)
    v_pre = cs_no_diag.max(dim=1).values  # (n_clients,)
    
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if v_pre[i] < v_pre[j]:
                # client i is "less guilty" → pardon its similarity to j
                ratio = (v_pre[i] / (v_pre[j] + 1e-12)).item()
                cs_pardoned[i, j] = cs_pardoned[i, j] * ratio
    
    # --- Step 6: Compute v_i after pardoning ---
    cs_pardoned.fill_diagonal_(-float('inf'))
    v, _ = torch.max(cs_pardoned, dim=1)
    v = torch.clamp(v, min=0.0, max=1.0)
    
    # Raw score: α_i = 1 - v_i
    alpha = 1.0 - v
    
    # --- Step 7: Normalize by max score (paper's explicit step before logit) ---
    alpha_max = alpha.max()
    if alpha_max > 1e-9:
        alpha = alpha / alpha_max  # Most honest client gets α = 1.0
    else:
        # All clients look identical → no one gets credit
        print("    > FoolsGold: All scores ~0. Using uniform weights.")
        alpha = torch.ones(n_clients, device=DEVICE)
    
    # --- Step 8: Logit transform with κ (paper's §3.1) ---
    # α = κ · (ln(α / (1-α)) + 0.5), clamped to [0, 1]
    eps = 1e-6
    alpha_safe = torch.clamp(alpha, min=eps, max=1.0 - eps)
    logit_val = torch.log(alpha_safe / (1.0 - alpha_safe))
    alpha = FOOLSGOLD_KAPPA * (logit_val + 0.5)
    alpha = torch.clamp(alpha, min=0.0, max=1.0)
    
    # --- Step 9: Weighted delta aggregation ---
    # Paper's rule: w_t = w_{t-1} + Σ_i α_i · Δ_i
    # In our pseudo-gradient FL framework, Δ_i = w_i - w_global already contains
    # accumulated local SGD steps (not raw per-sample gradients). Without 1/n the
    # update is n× too large and explodes. Dividing by n_clients is the correct
    # translation: when all α_i = 1 (all honest) this equals FedAvg exactly.
    new_global_dict = OrderedDict()
    for key in global_model:
        new_global_dict[key] = global_model[key].to(DEVICE).float().clone()
    
    for i, delta in enumerate(client_deltas):
        a_i = alpha[i].item() / n_clients  # 1/n scaling required in pseudo-gradient FL
        for key in delta:
            new_global_dict[key] += a_i * delta[key]
    
    # Cast back to original dtypes (e.g., BatchNorm num_batches_tracked → Long)
    for key in template_dict:
        new_global_dict[key] = new_global_dict[key].to(template_dict[key].dtype)
    
    print(f"    > FoolsGold: α = {[f'{a:.4f}' for a in alpha.cpu().tolist()]}")
    print(f"    > FoolsGold: Indicator layer = '{indicator_key}'")
    
    return new_global_dict, None
