"""
Central configuration file for the Federated Learning simulation.

"""

import torch
import os 

# --- === Simulation Parameters === ---
NUM_ROUNDS = 1000          # Total number of federated learning rounds
NUM_CLIENTS = 40          # Total number of clients in the pool
MIN_CLIENTS_PER_ROUND = 1   # Minimum clients to select each round
MAX_CLIENTS_PER_ROUND = 35  # Maximum clients to select each round
FRACTION_BYZANTINE = 0.3

# --- === Experiment Mode === ---
# Set to True to run the "Aegis vs Self" comparison across data splits/attacks.
# Set to False to run the standard manual experiments list in main.py.
COMPARE_AEGIS_SCENARIOS = False

# --- === Model & Data Parameters === ---
MODEL_TYPE = 'CNN'        # 'MLP' for MNIST, 'CNN' for CIFAR10
DATASET_NAME = 'CIFAR10'    
# DATA_SPLIT_TYPE can be: 'BALANCED_IID', 'UNBALANCED_IID', or 'NON_IID'
DATA_SPLIT_TYPE = 'BALANCED_IID'

# For NON_IID: Number of classes/shards per client
SHARDS_PER_CLIENT = 2 
DIRICHLET_ALPHA = 0.5
BATCH_SIZE = 128           

# --- === Client Training Parameters === ---
LOCAL_EPOCHS = 1
LEARNING_RATE = 0.001      
MOMENTUM = 0.8            

# --- === Aegis Parameters === ---
# T = S_median + (Threshold * MAD)
RWA_MAD_THRESHOLD = 3.0
RWA_EPSILON = 1e-9

REPUTATION_ALPHA = 0.7       # Weight for old reputation (0.7 * old + 0.3 * new)
REPUTATION_BAN_THRESHOLD = 0.15 # Score below which a client is banned forever
REPUTATION_GRACE_PERIOD = 10    # No bans in the first N rounds
REJECTION_PENALTY_SCORE = 0.1   # Score given to Aegis-rejected clients (not 0.0)



# --- === Attack Parameters === ---
# ATTACK_NOISE_STD:
# For Mean Shift (additive_noise), this is the magnitude of the shift.
# Since inputs are normalized (~0-1 range), a shift of 2.0 is MASSIVE.
ATTACK_NOISE_STD = 2.0  

# ATTACK_TYPE:
# Options: 'none', 'sign_flip', 'additive_noise', 'label_flip', 'orthogonal', 'volume_spam'
ATTACK_TYPE = 'additive_noise' 
ATTACK_PROBABILITY = 1    # Probability that a traitor attacks in a given round

# --- === Attack Timeline Parameters === ---
# Random Window Logic:
# Each traitor picks a start round [0, NUM_ROUNDS - MIN_DURATION]
# And a duration [MIN_DURATION, MAX_DURATION]
ATTACK_WINDOW_MIN_DURATION = 10
ATTACK_WINDOW_MAX_DURATION = 50
ATTACK_WINDOW_MAX_START_ROUND = 40 # 80% of attacks in first ~20% of convergence rounds




# --- === Performance Optimizations === ---
EVALUATE_EVERY_N_ROUNDS = 1

# --- MULTIPROCESSING CONTROL (Decoupled) ---

# 1. MAX_PARALLEL_CLIENTS:
# How many clients train simultaneously. 
#.
# If None, it uses all available cores.
MAX_PARALLEL_CLIENTS = 20 

# 2. DATALOADER_WORKERS:
# How many subprocesses each DataLoader uses to load data.
# RECOMMENDATION: Keep at 0 when using Multiprocessing for clients.
# If you set this > 0, you get (MAX_PARALLEL_CLIENTS * DATALOADER_WORKERS) total threads.
DATALOADER_WORKERS = 0

# --- === System Parameters === ---
# SERVER_DEVICE: The server (aggregation/eval) uses GPU.
# Clients will train on CPU (forced in server.py) to allow parallelism.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- === Early Stopping Parameters === ---
EARLY_STOPPING_ENABLED = True
PATIENCE = 10           # Number of rounds to wait for improvement
MIN_DELTA = 0.001       # Minimum change in loss to qualify as improvement

# --- === Results Directory === ---
RESULTS_DIR = 'Results'

# --- === Visualization Parameters === ---
VISUALIZE_GRADIENTS = False       # Master toggle for scatter plots
VISUALIZE_EVERY_N_ROUNDS = 10    # Save scatter plot every N rounds

