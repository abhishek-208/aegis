"""
Defines the Server class for the Federated Learning simulation.
"""

import torch
import torch.nn as nn
import random
import math
import time
import torch.multiprocessing as mp

import config
from model import get_model

# --- === HELPER FOR MULTIPROCESSING === ---
def client_training_wrapper(args):
    """
    Standalone function for multiprocessing.
    args: (client_instance, global_weights, is_byzantine, attack_type, device_to_use)
    """
    # Unpack the   5th argument
    client, global_weights, is_byzantine, attack_type, device = args
    return client.train(global_weights, is_byzantine, attack_type, force_device=device)

# --- ================================== ---

class Server:
    def __init__(self, aggregator_func, test_loader):
        self.device = config.DEVICE
        self.global_model = get_model().to(self.device)
        self.aggregator_func = aggregator_func
        self.test_loader = test_loader

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
        
        if num_byzantine > 0:
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
            # PARALLEL MODE: Force 'cpu' to avoid GPU thrashing/context errors
            training_device = 'cpu'
            for client in selected_clients:
                is_byz = client.client_id in byzantine_client_set
                mp_args.append((client, global_weights_cpu, is_byz, attack_type, training_device))
            
            # Run in Pool
            with mp.Pool(processes=config.MAX_PARALLEL_CLIENTS) as pool:
                updates = pool.map(client_training_wrapper, mp_args)
                
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
        
        # --- Step 4: Aggregation ---
        t_start_agg = time.time()
        new_global_weights = self.aggregator_func(updates)
        if config.DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        t_end_agg = time.time()
        
        # --- Step 5: Update Global Model ---
        if new_global_weights:
            self.global_model.load_state_dict(new_global_weights)

        return {
            "train_time": t_end_train - t_start_train,
            "agg_time": t_end_agg - t_start_agg
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