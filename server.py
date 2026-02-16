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
from reputation_manager import ReputationManager

# --- === HELPER FOR MULTIPROCESSING === ---
def client_training_wrapper(args):
    """
    Standalone function for multiprocessing.
    args: (client_instance, global_weights, is_byzantine, attack_type, device_to_use)
    """
    # Unpack the   5th argument
    client, global_weights, is_byzantine, attack_type, device = args
    
    # CRITICAL: Prevent oversubscription. Each worker gets 1 thread.
    if device == 'cpu':
        torch.set_num_threads(1)
        
    return client.train(global_weights, is_byzantine, attack_type, force_device=device)

# --- ================================== ---

class Server:
    def __init__(self, aggregator_func, test_loader):
        self.device = config.DEVICE
        self.global_model = get_model().to(self.device)
        self.aggregator_func = aggregator_func
        self.aggregator_func = aggregator_func
        self.test_loader = test_loader
        self.reputation_manager = ReputationManager()
        
        # ---Fixed Byzantine Clients ---
        # Select a fixed set of traitors once at the start.
        # This simulates a real-world scenario where specific devices are compromised.
        num_byzantine = math.floor(config.NUM_CLIENTS * config.FRACTION_BYZANTINE)
        all_indices = list(range(config.NUM_CLIENTS))
        self.fixed_byzantine_indices = set(random.sample(all_indices, num_byzantine))
        
        print(f"    [Server] Fixed Byzantine Clients (Count: {num_byzantine}): {sorted(list(self.fixed_byzantine_indices))}")
        
        # --- NEW: Random Attack Schedules ---
        # Generate a start and end round for each traitor
        self.byzantine_schedules = {}
        for client_id in self.fixed_byzantine_indices:
            # Ensure valid range
            limit_from_total = max(0, config.NUM_ROUNDS - config.ATTACK_WINDOW_MIN_DURATION)
            actual_max_start = min(limit_from_total, config.ATTACK_WINDOW_MAX_START_ROUND)
            
            # Beta distribution biases ~80% of start rounds into early rounds
            # betavariate(2, 5) has mean ~0.28, so most values cluster near 0
            start_round = int(random.betavariate(2, 5) * actual_max_start)
            
            duration = random.randint(config.ATTACK_WINDOW_MIN_DURATION, config.ATTACK_WINDOW_MAX_DURATION)
            end_round = start_round + duration
            
            self.byzantine_schedules[client_id] = (start_round, end_round)
            print(f"    [Server] Traitor {client_id} will attack from Round {start_round} to {end_round}")

    def select_clients(self, all_clients):
        # Filter out banned clients
        banned_ids = set(self.reputation_manager.get_banned_clients())
        candidates = [c for c in all_clients if c.client_id not in banned_ids]
        
        if not candidates:
            print("    [Server] CRITICAL: All clients are banned! Cannot evaluate.")
            return []

        num_to_select = random.randint(
            config.MIN_CLIENTS_PER_ROUND,
            config.MAX_CLIENTS_PER_ROUND
        )
        num_to_select = min(num_to_select, len(candidates))
        return random.sample(candidates, num_to_select)

    def run_round(self, all_clients, attack_type, fraction_byzantine, round_num=None):
        """Orchestrates one complete round of federated learning."""
        
        # --- Step 1: Client Selection ---
        selected_clients = self.select_clients(all_clients)
        
        # --- Step 2: Determine Attackers for this Round ---
        # Logic: A client is an attacker IF:
        # 1. They are in the fixed byzantine list AND
        # 2. They pass the probabilistic check (ATTACK_PROBABILITY)
        
        actual_attackers = []
        for client in selected_clients:
            if client.client_id in self.fixed_byzantine_indices:
                # This client is a Traitor. Are they in their attack window?
                is_active = False
                if round_num is not None:
                    start, end = self.byzantine_schedules.get(client.client_id, (0, 0))
                    if start <= round_num <= end:
                        is_active = True
                else:
                    # Fallback if round_num not provided (shouldn't happen in updated main)
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
        
        # global weights to CPU for pickling
        global_weights_cpu = {k: v.cpu() for k, v in self.global_model.state_dict().items()}
        
        mp_args = []
        
        # DECISION: Parallel CPU or Serial GPU?
        # If MAX_PARALLEL_CLIENTS is set, we use multiprocessing on CPU.
        use_parallel = (config.MAX_PARALLEL_CLIENTS is not None) and (config.MAX_PARALLEL_CLIENTS > 1)
        
        if use_parallel:
            # PARALLEL MODE: Use the device defined in config (likely CUDA)
            # CAUTION: This requires sufficient VRAM.
            training_device = self.device
            for client in selected_clients:
                is_byz = client.client_id in byzantine_client_set
                mp_args.append((client, global_weights_cpu, is_byz, attack_type, training_device))
            
            # Run in ThreadPool
            with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_CLIENTS) as executor:
                updates = list(executor.map(client_training_wrapper, mp_args))
                
        else:
            # SERIAL MODE: Use the config device (likely GPU)
            # This is often faster for small models!
            updates = []
            training_device = self.device # Use Server's GPU
            for client in selected_clients:
                is_byz = client.client_id in byzantine_client_set
                # We call the wrapper directly or client.train directly
                # Note: We must pass 'None' for force_device to let client use its default
                res = client.train(global_weights_cpu, is_byz, attack_type, force_device=None)
                updates.append(res)

        if config.DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        t_end_train = time.time()
        print(f"    > Training Time: {t_end_train - t_start_train:.2f}s")
        
        # --- Step 4: Aggregation ---
        t_start_agg = time.time()
        # --- Step 4: Aggregation ---
        t_start_agg = time.time()
        
        # Aggregator now returns (weights, stats) tuple
        new_global_weights, agg_stats = self.aggregator_func(updates)
        
        # --- Update Reputations ---
        if agg_stats and 'raw_scores' in agg_stats:
            raw_scores = agg_stats['raw_scores']
            participating_ids = [c.client_id for c in selected_clients]
            self.reputation_manager.update_reputations(participating_ids, raw_scores, current_round=round_num)
        
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
                        "original_indices": [c.client_id for c in selected_clients], # Correctly derived from local variable
                        "byzantine_set": list(byzantine_client_set)        # To know who is ACTUALLY bad
                    }
                else:
                    # Not enough samples for PCA
                    viz_data = None
            except ImportError:
                print("    > [Visualizer] Sklearn not found. Skipping PCA plot.")
                pass
        if config.DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        t_end_agg = time.time()
        
        # --- Step 5: Update Global Model ---
        if new_global_weights:
            self.global_model.load_state_dict(new_global_weights)

        return {
            "train_time": t_end_train - t_start_train,
            "agg_time": t_end_agg - t_start_agg,
            "viz_data": viz_data
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