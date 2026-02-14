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
from privacy_accountant import PrivacyAccountant

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
        self.test_loader = test_loader
        
        # --- Differential Privacy Accountant ---
        if config.DP_ENABLED:
            # Conservative sample rate (q) = Max possible clients / Total clients
            # This accounts for the worst-case privacy loss per step.
            q = config.MAX_CLIENTS_PER_ROUND / config.NUM_CLIENTS
            self.accountant = PrivacyAccountant(
                noise_multiplier=config.DP_NOISE_MULTIPLIER,
                sample_rate=q,
                delta=config.DP_DELTA
            )
        else:
            self.accountant = None

    def select_clients(self, all_clients):
        num_to_select = random.randint(
            config.MIN_CLIENTS_PER_ROUND,
            config.MAX_CLIENTS_PER_ROUND
        )
        num_to_select = min(num_to_select, len(all_clients))
        return random.sample(all_clients, num_to_select)

    def run_round(self, all_clients, attack_type, fraction_byzantine):
        """Orchestrates one complete round of federated learning."""
        
        # --- Step 1: Client Selection ---
        selected_clients = self.select_clients(all_clients)
        
        # --- Step 2: Designate Byzantine Clients ---
        num_byzantine = math.floor(len(selected_clients) * fraction_byzantine)
        byzantine_clients = random.sample(selected_clients, num_byzantine)
        byzantine_client_set = set(c.client_id for c in byzantine_clients)
        
        # Always print round info
        print(f"    > Round Info: {len(selected_clients)} Participants, {num_byzantine} Byzantine ({attack_type})")

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
        result = self.aggregator_func(updates)
        
        if result is None:
            new_global_weights, agg_stats = None, None
            print("    > Server: Aggregator returned None (Round Skipped).")
        else:
            new_global_weights, agg_stats = result

        
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
            # --- Differential Privacy: Add Noise to Aggregated Update ---
            if config.DP_ENABLED and getattr(config, 'DP_MODE', 'central') == 'central':
                # noise_std = (C * sigma) / N
                # We normalize by N because we are adding noise to the AVERAGE, not the SUM.
                # Use len(selected_clients) as the sensitivity denominator (assuming all participated)
                n_participants = len(selected_clients)
                noise_std = (config.DP_CLIP_NORM * config.DP_NOISE_MULTIPLIER) / n_participants
                
                print(f"    > DP (Central): Injecting noise (std={noise_std:.6f}) to global model...")
                
                for key in new_global_weights:
                    # Generate noise on the same device as the weights
                    noise = torch.normal(
                        mean=0.0, 
                        std=noise_std, 
                        size=new_global_weights[key].shape, 
                        device=new_global_weights[key].device
                    )
                    new_global_weights[key] += noise
            
            # If DP_MODE is 'local', noise was already added by clients.


            self.global_model.load_state_dict(new_global_weights)

            # --- DP Accounting ---
            if config.DP_ENABLED and self.accountant:
                self.accountant.step()
                eps = self.accountant.get_epsilon()
                print(f"    > DP: Spent Privacy Budget (Epsilon): {eps:.4f} / {config.DP_TARGET_EPSILON}")
                
                if eps >= config.DP_TARGET_EPSILON:
                    print("    > DP: Privacy Budget Exhausted! Stopping training.")
                    # Return a signal to stop? Or raise StopIteration?
                    # For now, we will rely on main loop checking or just let it continue but warn (soft stop)
                    # Ideally, we should modify the return dict to signal 'stop'
                    return {
                        "train_time": t_end_train - t_start_train,
                        "agg_time": t_end_agg - t_start_agg,
                        "viz_data": viz_data,
                        "stop_training": True # New flag
                    }
        
        return {
            "train_time": t_end_train - t_start_train,
            "agg_time": t_end_agg - t_start_agg,
            "viz_data": viz_data,
            "stop_training": False
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