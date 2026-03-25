"""
Defines the Server class for the Federated Learning simulation.
"""

import torch
import torch.nn as nn
import random
import math
import time
import concurrent.futures
from collections import OrderedDict
from scipy.stats import norm as scipy_norm

import config
from model import get_model

# --- === HELPER FOR MULTIPROCESSING === ---
def client_training_wrapper(args):
    """
    Standalone function for multiprocessing.
    args: (client_instance, global_weights, is_byzantine, attack_type, device_to_use, current_lr)
    """
    # Unpack the 6 arguments
    client, global_weights, is_byzantine, attack_type, device, current_lr = args
    
    # CRITICAL: Prevent oversubscription. Each worker gets 1 thread.
    if device == 'cpu':
        torch.set_num_threads(1)
        
    return client.train(global_weights, is_byzantine, attack_type, force_device=device, current_lr=current_lr)

# --- ================================== ---

class Server:
    def __init__(self, aggregator_func, test_loader):
        self.device = config.DEVICE
        self.global_model = get_model().to(self.device)
        self.aggregator_func = aggregator_func
        self.test_loader = test_loader
        
        # --- velocity buffer for Global Momentum (FedAvgM) ---
        # Stores the "moving average" of the server updates (V_t)
        self.server_momentum_buffer = None
        
        # --- Learning Rate Management ---
        self.current_lr = config.LEARNING_RATE
        
        # --- Fixed Byzantine Clients ---
        # Select a fixed set of traitors once at the start.
        # This simulates a real-world scenario where specific devices are compromised.
        num_byzantine = math.floor(config.NUM_CLIENTS * config.FRACTION_BYZANTINE)
        all_indices = list(range(config.NUM_CLIENTS))
        self.fixed_byzantine_indices = set(random.sample(all_indices, num_byzantine))
        
        print(f"    [Server] Fixed Byzantine Clients (Count: {num_byzantine}): {sorted(list(self.fixed_byzantine_indices))}")
        
        # --- Random Attack Schedules ---
        # Generate a start and end round for each traitor
        self.byzantine_schedules = {}
        for client_id in self.fixed_byzantine_indices:
            # Ensure valid range
            limit_from_total = max(0, config.NUM_ROUNDS - config.ATTACK_WINDOW_MIN_DURATION)
            actual_max_start = min(limit_from_total, config.ATTACK_DEADLINE_ROUND)
            
            # Beta distribution biases ~80% of start rounds into early rounds
            # betavariate(2, 5) has mean ~0.28, so most values cluster near 0
            start_round = int(random.betavariate(2, 5) * actual_max_start)
            
            duration = random.randint(config.ATTACK_WINDOW_MIN_DURATION, config.ATTACK_WINDOW_MAX_DURATION)
            end_round = start_round + duration
            
            self.byzantine_schedules[client_id] = (start_round, end_round)
            print(f"    [Server] Traitor {client_id} will attack from Round {start_round} to {end_round}")

    def select_clients(self, all_clients):
        num_to_select = random.randint(
            config.MIN_CLIENTS_PER_ROUND,
            config.MAX_CLIENTS_PER_ROUND
        )
        num_to_select = min(num_to_select, len(all_clients))
        return random.sample(all_clients, num_to_select)

    def run_round(self, all_clients, attack_type, fraction_byzantine, current_round=None):
        """Orchestrates one complete round of federated learning."""
        
        # --- Step 1: Client Selection ---
        selected_clients = self.select_clients(all_clients)
        
        # --- Step 2: Determine Attackers for this Round ---
        # Logic: A client is an attacker IF:
        # 1. They are in the fixed byzantine list AND
        # 2. The current round is within their attack window AND
        # 3. They pass the probabilistic check (ATTACK_PROBABILITY)
        
        actual_attackers = []
        
        # Only select attackers if the attack_type is not 'none'
        if attack_type != 'none':
            for client in selected_clients:
                if client.client_id in self.fixed_byzantine_indices:
                    # This client is a Traitor. Are they in their attack window?
                    is_active = False
                    if current_round is not None:
                        start, end = self.byzantine_schedules.get(client.client_id, (0, 0))
                        if start <= current_round <= end:
                            is_active = True
                    else:
                        # Fallback if current_round not provided
                        is_active = True
                    
                    if is_active:
                         if random.random() < config.ATTACK_PROBABILITY:
                             actual_attackers.append(client)
        
        byzantine_client_set = set(c.client_id for c in actual_attackers)
        num_byzantine = len(actual_attackers)
        
        # Always print round info
        print(f"    > Round Info: {len(selected_clients)} Participants, {num_byzantine} Active Attackers ({attack_type})")

        # --- Step 3: Local Training ---
        t_start_train = time.time()
        
        # global weights to CPU for safe state_dict loading
        global_weights_cpu = {k: v.cpu() for k, v in self.global_model.state_dict().items()}
        
        # DECISION: Parallel GPU or Serial GPU?
        max_parallel = config.MAX_PARALLEL_CLIENTS
        use_parallel = (max_parallel is not None) and (max_parallel > 1)
        
        # Always train on the GPU (server device)
        training_device = self.device
        
        if use_parallel:
            # GPU-PARALLEL MODE: Multiple threads share the GPU
            mp_args = []
            for client in selected_clients:
                is_byz = client.client_id in byzantine_client_set
                mp_args.append((client, global_weights_cpu, is_byz, attack_type, training_device, self.current_lr))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
                updates = list(executor.map(client_training_wrapper, mp_args))
                
        else:
            # SERIAL MODE: One client at a time on GPU
            updates = []
            for client in selected_clients:
                is_byz = client.client_id in byzantine_client_set
                res = client.train(global_weights_cpu, is_byz, attack_type, force_device=training_device, current_lr=self.current_lr)
                updates.append(res)

        if config.DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        t_end_train = time.time()
        print(f"    > Training Time: {t_end_train - t_start_train:.2f}s")
        
        # --- SYBIL ATTACK SYNCHRONIZATION ---
        # If this is a Sybil attack, all attackers must submit the exact same update.
        if attack_type == 'sybil' and len(byzantine_client_set) > 0:
            sybil_base_update = None
            for update in updates:
                c_id, w, n_samp = update
                if c_id in byzantine_client_set:
                    # Make a deepcopy or just reference the weights
                    # Since we only read them during aggregation, reference is fine
                    sybil_base_update = (w, n_samp)
                    break
            
            if sybil_base_update is not None:
                base_w, base_n = sybil_base_update
                for i in range(len(updates)):
                    c_id, _, _ = updates[i]
                    if c_id in byzantine_client_set:
                        updates[i] = (c_id, base_w, base_n)

        # --- ALIE ATTACK COORDINATION ---
        # "A Little Is Enough" (Baruch et al., NeurIPS 2019)
        # All Byzantines first trained honestly. Now we replace their updates
        # with a coordinated poisoned gradient crafted from cross-client statistics.
        if attack_type == 'alie' and len(byzantine_client_set) > 0:
            
            n_total = len(updates)
            m_byz = len(byzantine_client_set)
            
            # --- Step A: Collect the gradients (deltas) to compute statistics over ---
            # Flatten the global model once for delta computation.
            flat_global = torch.cat([p.flatten() for p in global_weights_cpu.values()])
            
            # Decide which clients' gradients to use for μ, σ estimation.
            if config.ALIE_USE_OMNISCIENT:
                # Omniscient: use ALL clients' deltas (honest + Byzantine)
                stat_deltas = []
                for c_id, w, n_samp in updates:
                    flat_w = torch.cat([v.flatten() for v in w.values()])
                    stat_deltas.append(flat_w - flat_global)
            else:
                # Colluding: use only Byzantine clients' deltas
                stat_deltas = []
                for c_id, w, n_samp in updates:
                    if c_id in byzantine_client_set:
                        flat_w = torch.cat([v.flatten() for v in w.values()])
                        stat_deltas.append(flat_w - flat_global)
            
            if len(stat_deltas) >= 2:
                # Stack into matrix: (num_stat_clients, model_dim)
                stat_matrix = torch.stack(stat_deltas)
                
                # --- Step B: Compute coordinate-wise mean and std ---
                mu = stat_matrix.mean(dim=0)    # (model_dim,)
                sigma = stat_matrix.std(dim=0)  # (model_dim,)
                
                # --- Step C: Determine z ---
                if config.ALIE_Z is not None:
                    # Use the fixed z from config
                    z = config.ALIE_Z
                else:
                    # Paper formula: z = Φ^{-1}((n - 2m) / (n - m))
                    # This is the maximum z such that the poisoned value still
                    # looks like it could be an honest gradient to a majority filter.
                    ratio = (n_total - 2 * m_byz) / max(n_total - m_byz, 1)
                    ratio = max(0.001, min(ratio, 0.999))  # Clamp for numerical safety
                    z = scipy_norm.ppf(ratio)
                    z = max(z, 0.0)  # Floor at 0 — negative z has negligible impact
                
                # --- Step D: Craft the poisoned delta ---
                # g_malicious = μ - z * σ  (pushes every coordinate downward)
                poisoned_delta = mu - z * sigma
                
                # Convert back to weight space: W_malicious = W_global + poisoned_delta
                poisoned_flat_weights = flat_global + poisoned_delta
                
                # Unflatten back into a state_dict using the template structure
                template_dict = updates[0][1]
                poisoned_weights = OrderedDict()
                idx = 0
                for key, tensor in template_dict.items():
                    numel = tensor.numel()
                    poisoned_weights[key] = poisoned_flat_weights[idx:idx + numel].reshape(tensor.shape)
                    idx += numel
                
                # --- Step E: Replace all Byzantine clients' updates ---
                for i in range(len(updates)):
                    c_id, w, n_samp = updates[i]
                    if c_id in byzantine_client_set:
                        # Keep the honest n_samp (no volume inflation — ALIE is stealth-only)
                        updates[i] = (c_id, poisoned_weights, n_samp)
                
                print(f"    > [ALIE] Crafted poisoned gradient: z={z:.4f}, "
                      f"using {'ALL' if config.ALIE_USE_OMNISCIENT else 'Byzantine-only'} "
                      f"gradients ({len(stat_deltas)} clients)")
            else:
                print(f"    > [ALIE] Not enough Byzantine clients for stats ({len(stat_deltas)}). "
                      f"Skipping coordination this round.")

        # --- Step 4: Aggregation ---
        t_start_agg = time.time()
        
        # Aggregator now returns (weights, stats) tuple
        # Pass current_round for adaptive thresholding (aegis uses it, others ignore it)
        try:
            new_global_weights, agg_stats = self.aggregator_func(updates, current_round=current_round, global_model=global_weights_cpu)
        except TypeError:
            # Aggregator doesn't accept current_round (e.g., fed_avg, cw_med, multi_krum)
            new_global_weights, agg_stats = self.aggregator_func(updates)
        
        if config.DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        t_end_agg = time.time()
        
        # --- Visualization: PCA Projection ---
        viz_data = None
        if agg_stats:
            try:
                from sklearn.decomposition import PCA
                
                # agg_stats contains: weights_matrix (ndarray), approved_indices, original_indices
                weights = agg_stats['weights_matrix']
                n_samples = weights.shape[0]
                
                if n_samples >= 2:
                    pca = PCA(n_components=2)
                    coords = pca.fit_transform(weights) # Shape (N, 2)
                    
                    viz_data = {
                        "coords": coords,
                        "approved_indices": agg_stats['approved_indices'],
                        "original_indices": [c.client_id for c in selected_clients],
                        "byzantine_set": list(byzantine_client_set)
                    }
                else:
                    # Not enough samples for PCA
                    viz_data = None
            except ImportError:
                print("    > [Visualizer] Sklearn not found. Skipping PCA plot.")
                pass
        
        # --- Step 5: Update Global Model ---
        if new_global_weights:
            # Server Momentum is an Aegis-specific enhancement.
            # FedAvg, CW-Med, and Krum should use their standard weight update
            # without any server-side memory, to ensure a fair baseline comparison.
            agg_name = getattr(self.aggregator_func, '__name__', '') or getattr(self.aggregator_func, 'func', lambda: None).__name__
            is_aegis = (agg_name == 'aegis')
            if config.SERVER_MOMENTUM_ENABLED and is_aegis:  # In case of Global Momentum (Aegis only)
                current_global_weights = self.global_model.state_dict()
                
                # 1. Initialize velocity buffer if it's the first round
                if self.server_momentum_buffer is None:
                    self.server_momentum_buffer = {}
                    for k, v in current_global_weights.items():
                        self.server_momentum_buffer[k] = torch.zeros_like(v, device=self.device)
                
                # 2. Compute Pseudo-Gradient and Apply Momentum
                # IMPORTANT: Skip momentum for BatchNorm running stats (running_mean,
                # running_var, num_batches_tracked). These are NOT learnable parameters —
                # they are exponential moving averages maintained during training.
                # Applying momentum can push running_var negative → sqrt(negative) → NaN.
                _BN_STATS = {'running_mean', 'running_var', 'num_batches_tracked'}
                
                final_weights = {}
                for k in new_global_weights.keys():
                    # Check if this key ends with a BatchNorm stat name
                    is_bn_stat = any(k.endswith(s) for s in _BN_STATS)
                    
                    if is_bn_stat:
                        # Directly use the aggregated value (no momentum)
                        final_weights[k] = new_global_weights[k].to(self.device)
                    else:
                        v_current = current_global_weights[k].to(self.device)
                        v_new = new_global_weights[k].to(self.device)
                        
                        # Pseudo-Gradient: The direction the clients *want* the server to move.
                        pseudo_grad = v_new - v_current
                        
                        # Velocity Update: V_t = (Momentum * V_t-1) + Pseudo-Gradient
                        velocity = (config.SERVER_MOMENTUM * self.server_momentum_buffer[k]) + pseudo_grad
                        self.server_momentum_buffer[k] = velocity
                        
                        # Final Model Update: W_t = W_t-1 + (Learning_Rate * V_t)
                        final_weights[k] = v_current + (config.SERVER_LEARNING_RATE * velocity)
                    
                self.global_model.load_state_dict(final_weights)
            else:
                # Standard FL: Just load the newly averaged weights directly (No memory and no global momentum)
                self.global_model.load_state_dict(new_global_weights)

        # Extract stats (e.g., adaptive k)
        adaptive_k = None
        if agg_stats and "adaptive_k" in agg_stats:
            adaptive_k = agg_stats["adaptive_k"]

        # --- Step 6: Learning Rate Decay ---
        if config.LR_DECAY_ENABLED:
            self.current_lr = max(config.MIN_LR, self.current_lr * config.LR_DECAY_RATE)

        return {
            "train_time": t_end_train - t_start_train,
            "agg_time": t_end_agg - t_start_agg,
            "viz_data": viz_data,
            "num_byzantine": num_byzantine,
            "num_selected": len(selected_clients),
            "adaptive_k": adaptive_k # Pass k to main.py
        }

    def evaluate(self):
        self.global_model.eval()
        test_loss = 0
        correct = 0
        criterion = nn.CrossEntropyLoss(reduction='sum')
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        return test_loss, accuracy