"""
Central configuration file for the Federated Learning simulation.

"""

import torch
import os 

# --- === 1. Simulation Core Parameters === ---
NUM_ROUNDS = 1000            # Total number of federated learning rounds
NUM_CLIENTS = 30             # Total number of clients in the pool
MIN_CLIENTS_PER_ROUND = 15   # Minimum clients to select each round
MAX_CLIENTS_PER_ROUND = 25   # Maximum clients to select each round
RANDOM_SEED = 42             # Fixed seed for reproducibility control

# --- === 2. Experiment Toggles === ---
# Only ONE of the following should be set to True (or both False for manual experiments).
COMPARE_AEGIS_SCENARIOS = False    # Set True to run "Aegis vs Baselines" comparison
RUN_ABLATION_STUDY = True         # Set True to run Aegis Ablation Study across components

# --- === 3. Data & Model Parameters === ---
MODEL_TYPE = 'ImprovedCNN'  # 'MLP' (MNIST), 'CNN' (LeNet-5), 'ImprovedCNN' (CIFAR10)
DATASET_NAME = 'CIFAR10'    
DATA_SPLIT_TYPE = 'NON_IID' # Options: 'BALANCED_IID', 'UNBALANCED_IID', or 'NON_IID'
SHARDS_PER_CLIENT = 4       # For NON_IID: Number of classes/shards per client
DIRICHLET_ALPHA = 0.5       # For NON_IID Dirichlet splits (if implemented)
BATCH_SIZE = 256           

# --- === 4. Local Training & LR Decay Parameters === ---
LOCAL_EPOCHS = 1
LEARNING_RATE = 0.01      
MOMENTUM = 0.8            
LR_DECAY_ENABLED = True
LR_DECAY_RATE = 0.99       # Decay multiplier per round
MIN_LR = 1e-4              # Minimum learning rate limit

# --- === 5. Server Aggregation & Momentum Parameters === ---
# Server Momentum acts as an inertia buff against massive late-stage accuracy drops.
SERVER_MOMENTUM_ENABLED = True      # If False, works as standard FL.
SERVER_MOMENTUM = 0.7               # FedAvgM momentum parameter (velocity to keep).
SERVER_LEARNING_RATE = 1.0          # Server step size.

# --- === 6. Aegis Adaptive Thresholding Parameters === ---
# The rejection cutoff: T = median_distance + (k * MAD)
ADAPTIVE_THRESHOLD_ENABLED = True   # Set False to use fixed OUTLIER_SENSITIVITY
OUTLIER_SENSITIVITY = 3.0           # Fixed fallback when adaptive is disabled
# Strategy A: Round-Based Decay
K_MAX = 6.0                         # Initial (loose) threshold multiplier
K_MIN = 2.0                         # Final (strict) threshold multiplier (for IID)
WARMUP_ROUNDS = 300                 # Rounds over which k linearly decays
K_SAFE_FLOOR = 4.5                  # Absolute minimum k (vital for Non-IID cases)
# Strategy C: Variance-Normalized
VARIANCE_SENSITIVITY = 3.0          # Relaxation scale when updates are highly dispersed
RWA_EPSILON = 1e-9                  # Numerical stability constant

# --- === 7. FoolsGold Defense Parameters === ---
FOOLSGOLD_KAPPA = 1.0               # Higher bounds -> more aggressive suppression

# --- === 8. Byzantine Attack Parameters === ---
FRACTION_BYZANTINE = 0.40           # Ratio of Byzantine clients in the total pool

# Options: 'none', 'sign_flip', 'additive_noise', 'label_flip', 'orthogonal', 'volume_spam', 'sybil', 'catastrophic_noise', 'informed_orthogonal', 'alie'
ATTACK_TYPE = 'sign_flip' 
ATTACK_PROBABILITY = 0.9            # Probability of a traitor attacking whilst in their window
ATTACK_NOISE_STD = 2.0              # Multiplier for additive_noise magnitudes
NUM_SYBILS_PER_ATTACKER = 3         # Fake identities per traitor during a Sybil attack

# ALIE Specifics ("A Little Is Enough")
ALIE_Z = 1.0                        # Set to None for algorithmic paper formula.
ALIE_USE_OMNISCIENT = False         # True = use all gradients, False = only traitor gradients

# Attack Window Timeline
ATTACK_WINDOW_MIN_DURATION = 10     # Minimum active attack window length
ATTACK_WINDOW_MAX_DURATION = 200     # Maximum active attack window length
EXPECTED_CONVERGENCE_ROUNDS = 400   # Used to calculate the deadline
ATTACK_DEADLINE_PERCENT = 0.50      # Attacks must start prior to this % of convergence
ATTACK_DEADLINE_ROUND = int(EXPECTED_CONVERGENCE_ROUNDS * ATTACK_DEADLINE_PERCENT)

# --- === 9. Environment & System Limits === ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_PARALLEL_CLIENTS = 1            # 1 for Serial/GPU, None for maximum CPU threads
DATALOADER_WORKERS = 0              # MUST stay 0 if Multiprocessing clients > 0

# --- === 10. Evaluation & Early Stopping === ---
EVALUATE_EVERY_N_ROUNDS = 2
EARLY_STOPPING_ENABLED = True
PATIENCE = 40                       # Wait time for loss improvement
MIN_DELTA = 0.001                   # Substantial enough loss drop to reset patience

# --- === 11. Output, Visualization & System === ---
RESULTS_DIR = './saved_models'      # './saved_models' for Modal, 'C:/...' for local
VISUALIZE_GRADIENTS = False         # Master toggle for distance scatters
VISUALIZE_EVERY_N_ROUNDS = 10       
AUTO_SHUTDOWN = False               # Automatically turn off machine post-run
