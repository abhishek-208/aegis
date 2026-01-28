"""
Central configuration file for the Federated Learning simulation.

"""

import torch
import os 

# --- === Simulation Parameters === ---
NUM_ROUNDS = 80           # Total number of federated learning rounds
NUM_CLIENTS = 30          # Total number of clients in the pool
MIN_CLIENTS_PER_ROUND = 1   # Minimum clients to select each round
MAX_CLIENTS_PER_ROUND = 26  # Maximum clients to select each round
FRACTION_BYZANTINE = 0.4

# --- === Model & Data Parameters === ---
MODEL_TYPE = 'CNN'        # 'MLP' for MNIST, 'CNN' for CIFAR10
DATASET_NAME = 'CIFAR10'    
# DATA_SPLIT_TYPE can be: 'BALANCED_IID', 'UNBALANCED_IID', or 'NON_IID'
DATA_SPLIT_TYPE = 'BALANCED_IID'

# For NON_IID: Number of classes/shards per client
SHARDS_PER_CLIENT = 2 
DIRICHLET_ALPHA = 0.5
BATCH_SIZE = 32           

# --- === Client Training Parameters === ---
LOCAL_EPOCHS = 1
LEARNING_RATE = 0.001      
MOMENTUM = 0.8            

# --- === Aegis Parameters === ---
# T = S_median + (Threshold * MAD)
RWA_MAD_THRESHOLD = 3.0
RWA_EPSILON = 1e-9

# --- === Attack Parameters === ---
# ATTACK_NOISE_STD:
# For Mean Shift (additive_noise), this is the magnitude of the shift.
# Since inputs are normalized (~0-1 range), a shift of 2.0 is MASSIVE.
ATTACK_NOISE_STD = 2.0  

# ATTACK_TYPE:
# Options: 'none', 'sign_flip', 'additive_noise'
ATTACK_TYPE = 'additive_noise' 

# --- === Performance Optimizations === ---
EVALUATE_EVERY_N_ROUNDS = 1

# --- MULTIPROCESSING CONTROL (Decoupled) ---

# 1. MAX_PARALLEL_CLIENTS:
# How many clients train simultaneously. 
#.
# If None, it uses all available cores.
MAX_PARALLEL_CLIENTS = 6 

# 2. DATALOADER_WORKERS:
# How many subprocesses each DataLoader uses to load data.
# RECOMMENDATION: Keep at 0 when using Multiprocessing for clients.
# If you set this > 0, you get (MAX_PARALLEL_CLIENTS * DATALOADER_WORKERS) total threads.
DATALOADER_WORKERS = 0

# --- === System Parameters === ---
# SERVER_DEVICE: The server (aggregation/eval) uses GPU.
# Clients will train on CPU (forced in server.py) to allow parallelism.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- === Results Directory === ---
RESULTS_DIR = 'Results'