"""
Defines the Server class for the Federated Learning simulation.
"""

import torch
import torch.nn as nn
import random
import math
import time
import concurrent.futures

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
        
        # --- Step 4: Aggregation ---
        t_start_agg = time.time()
        
        # Aggregator now returns (weights, stats) tuple
        # Pass current_round for adaptive thresholding (aegis uses it, others ignore it)
        try:
            new_global_weights, agg_stats = self.aggregator_func(updates, current_round=current_round)
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
                final_weights = {}
                for k in new_global_weights.keys():
                    v_current = current_global_weights[k].to(self.device)
                    v_new = new_global_weights[k].to(self.device)
                    
                    # Pseudo-Gradient: The direction the clients *want* the server to move.
                    pseudo_grad = v_new - v_current
                    
                    # Velocity Update: V_t = (Momentum * V_t-1) + Pseudo-Gradient
                    # We keep a fraction of the previous velocity (inertia) and add the current push.
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