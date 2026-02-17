import time
import torch
import random
import config
from server import Server
from client import Client
from aggregator import aegis
from data_utils import load_data, partition_data, get_test_dataloader

# --- Override Config for fast test ---
config.NUM_ROUNDS = 20 # Need more rounds to see windows
config.NUM_CLIENTS = 5
config.MIN_CLIENTS_PER_ROUND = 5
config.MAX_CLIENTS_PER_ROUND = 5
config.FRACTION_BYZANTINE = 0.4 
config.ATTACK_TYPE = 'sign_flip' 
config.ATTACK_PROBABILITY = 1.0  

# Verify Windows
config.ATTACK_WINDOW_MIN_DURATION = 3
config.ATTACK_WINDOW_MAX_DURATION = 8
config.ATTACK_DEADLINE_ROUND = 5 # Force early attacks
config.REPUTATION_GRACE_PERIOD = 5       # Short grace period for testing

def verify_reputation():
    print("--- Starting Reputation System Verification (Optimization Check) ---")
    
    # 1. Setup Data & Clients
    print("Loading data...")
    train_dataset, test_dataset = load_data()
    # Use small subset or just partition normally (cifar is fast enough for 5 rounds if we limit epochs)
    config.LOCAL_EPOCHS = 1 # ensure fast
    
    client_dataloaders = partition_data(train_dataset, split_type='BALANCED_IID')
    # Limit to 10 clients
    client_dataloaders = client_dataloaders[:config.NUM_CLIENTS]
    
    test_loader = get_test_dataloader(test_dataset)
    all_clients = [Client(cid, loader) for cid, loader in enumerate(client_dataloaders)]
    
    # 2. Setup Server
    server = Server(aggregator_func=aegis, test_loader=test_loader)
    
    # 3. Run Rounds
    fixed_traitors = server.fixed_byzantine_indices
    print(f"Server initialized with Traitors: {fixed_traitors}")
    
    for round_num in range(config.NUM_ROUNDS):
        print(f"\n--- Round {round_num + 1} ---")
        # Note: fraction_byzantine arg in run_round is now effectively ignored for SELECTION, 
        # but kept for API compatibility or used for attack scaling if needed. 
        # The logic now uses fixed_byzantine_indices.
        server.run_round(all_clients, config.ATTACK_TYPE, config.FRACTION_BYZANTINE, round_num=round_num)
        
        # Check reputations
        reps = server.reputation_manager.reputations
        print(f"Current Reputations: {reps}")
        banned = server.reputation_manager.get_banned_clients()
        print(f"Banned Clients: {banned}")
        
    # 4. Final Verification
    print("\n--- Final Results ---")
    banned = server.reputation_manager.get_banned_clients()
    if banned:
        print(f"SUCCESS: Banned clients detected: {banned}")
    else:
        print("WARNING: No clients were banned. Check attack strength or threshold.")
        
    # Check if reputations changed from 0.5
    changed = any(r != 0.5 for r in server.reputation_manager.reputations.values())
    if changed:
         print("SUCCESS: Reputations updated.")
    else:
         print("FAILURE: Reputations did not change.")

if __name__ == "__main__":
    verify_reputation()
