"""
Central configuration file for the Federated Learning simulation.

"""

import torch
import os 

# --- === Simulation Parameters === ---
NUM_ROUNDS = 1000        # Total number of federated learning rounds
NUM_CLIENTS = 30          # Total number of clients in the pool
MIN_CLIENTS_PER_ROUND = 15   # Minimum clients to select each round
MAX_CLIENTS_PER_ROUND = 25  # Maximum clients to select each round
FRACTION_BYZANTINE = 0.40

# --- === Experiment Mode === ---
# Only ONE of the following should be set to True (or both False for standard manual experiments).

# Set to True to run the "Aegis vs Self" comparison across data splits/attacks.
COMPARE_AEGIS_SCENARIOS = False

# Set to True to run the Aegis Ablation Study across different components.
RUN_ABLATION_STUDY = True

# --- === Model & Data Parameters === ---
MODEL_TYPE = 'ImprovedCNN'  # 'MLP' for MNIST, 'CNN' (LeNet-5) or 'ImprovedCNN' (BatchNorm+Dropout) for CIFAR10
DATASET_NAME = 'CIFAR10'    
# DATA_SPLIT_TYPE can be: 'BALANCED_IID', 'UNBALANCED_IID', or 'NON_IID'
DATA_SPLIT_TYPE = 'NON_IID'

# For NON_IID: Number of classes/shards per client
SHARDS_PER_CLIENT = 4 
DIRICHLET_ALPHA = 0.5
BATCH_SIZE = 256           

# --- === Client Training Parameters === ---
LOCAL_EPOCHS = 1
LEARNING_RATE = 0.01      
MOMENTUM = 0.8            

# --- === Learning Rate Decay parameters === ---
LR_DECAY_ENABLED = True
LR_DECAY_RATE = 0.99       # Decay multiplier per round
MIN_LR = 1e-4              # Minimum learning rate limit

# --- === Server-Side Global Momentum Parameters === ---
# By default, standard FedAvg/Aegis has no server memory (it just averages). 
# Enabling Server Momentum acts like an inertia buff, preventing massive late-stage accuracy drops 
# when a randomly selected batch of clients is highly skewed.
SERVER_MOMENTUM_ENABLED = True     # Toggle to turn Global Momentum ON or OFF. If off, works as standard FL.
SERVER_MOMENTUM = 0.7               # FedAvgM momentum parameter (How much past velocity to keep).
SERVER_LEARNING_RATE = 1.0          # Server step size. 1.0 means take the full updated averaged step.
                                    # Higher values overshoot, lower values slow down global convergence.

# --- === Adaptive Thresholding Parameters === ---
# Adaptive Thresholding (Strategy A + C)
# The rejection cutoff: T = median_distance + (k * MAD)
# where k adapts each round based on training phase and current variance.
ADAPTIVE_THRESHOLD_ENABLED = True   # Set False to use fixed OUTLIER_SENSITIVITY
OUTLIER_SENSITIVITY = 3.0           # Fixed fallback when adaptive is disabled

# Strategy A: Round-Based Decay — k decays from K_MAX to K_MIN over WARMUP_ROUNDS
K_MAX = 6.0                         # Initial (loose) threshold multiplier
K_MIN = 2.0                         # Final (strict) threshold multiplier, in case of IID data
WARMUP_ROUNDS = 300                 # Rounds over which k linearly decays
K_SAFE_FLOOR = 4.5                 # Absolute minimum k, even with low variance (For Non IID cases)

# Strategy C: Variance-Normalized — k self-calibrates based on current spread
VARIANCE_SENSITIVITY = 3.0          # How much to relax k when updates are spread out
RWA_EPSILON = 1e-9

# --- === FoolsGold Parameters === ---
# κ (kappa): Confidence parameter for the logit transform.
# Higher values → more aggressive penalty for similar clients.
# Paper default: 1.0
FOOLSGOLD_KAPPA = 1.0

# --- === Attack Parameters === ---
# ATTACK_NOISE_STD:
# For Mean Shift (additive_noise), this is the magnitude of the shift.
# Since inputs are normalized (~0-1 range), a shift of 2.0 is MASSIVE.
ATTACK_NOISE_STD = 2.0  

# --- === ALIE Attack Parameters === ---
# "A Little Is Enough" (Baruch et al., NeurIPS 2019)
# z controls how many standard deviations below the mean the attack targets.
# Higher z = more aggressive but easier to detect.
# Lower z = more stealthy but less impactful per round.
ALIE_Z = 1.0                # Fixed z override. Set to None to use the paper's formula.
ALIE_USE_OMNISCIENT = False  # True = use all clients' grads for stats (strongest attack).
                             # False = use only Byzantine clients' grads (realistic).

# ATTACK_TYPE:
# Options: 'none', 'sign_flip', 'additive_noise', 'label_flip', 'orthogonal', 'volume_spam', 'sybil', 'catastrophic_noise', 'informed_orthogonal'
ATTACK_TYPE = 'sybil' 
ATTACK_PROBABILITY = 0.9    # Probability that a traitor attacks in a given round

# NUM_SYBILS_PER_ATTACKER: How many fake identities each attacker creates during a Sybil attack.
NUM_SYBILS_PER_ATTACKER = 3

# --- === Attack Timeline Parameters === ---
# Random Window Logic:
# Each traitor picks a start round [0, NUM_ROUNDS - MIN_DURATION]
# And a duration [MIN_DURATION, MAX_DURATION]
ATTACK_WINDOW_MIN_DURATION = 10     # Minimum attack window length
ATTACK_WINDOW_MAX_DURATION = 50     # Maximum attack window length

EXPECTED_CONVERGENCE_ROUNDS = 400   # Estimated rounds for model to converge
ATTACK_DEADLINE_PERCENT = 0.50     # Attacks must START within this % of convergence
ATTACK_DEADLINE_ROUND = int(EXPECTED_CONVERGENCE_ROUNDS * ATTACK_DEADLINE_PERCENT)

# --- === Performance Optimizations === ---
EVALUATE_EVERY_N_ROUNDS = 1

# --- MULTIPROCESSING CONTROL (Decoupled) ---

# 1. MAX_PARALLEL_CLIENTS:
# How many clients train simultaneously. 
#.
# If None, it uses all available cores. 
#MAX_PARALLEL_CLIENTS = 12  #For local machine
MAX_PARALLEL_CLIENTS = 1   #for running on Modal and using only GPU

# 2. DATALOADER_WORKERS:
# How many subprocesses each DataLoader uses to load data.
# RECOMMENDATION: Keep at 0 when using Multiprocessing for clients.
# If you set this > 0, you get (MAX_PARALLEL_CLIENTS * DATALOADER_WORKERS) total threads.
DATALOADER_WORKERS = 0  # MUST stay 0 — setting >0 with parallel clients causes RAM deadlock

# --- === System Parameters === ---
# SERVER_DEVICE: The server (aggregation/eval) uses GPU.
# Clients will train on CPU (forced in server.py) to allow parallelism.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- === Early Stopping Parameters === ---
EARLY_STOPPING_ENABLED = True
PATIENCE = 40           # Number of rounds to wait for improvement
MIN_DELTA = 0.001       # Minimum change in loss to qualify as improvement

# --- === Results Directory === ---
#RESULTS_DIR = r'D:\IITD\MTP 2\Results' # For running on local machine
RESULTS_DIR = './saved_models' #For running on Modal

# --- === Auto Shutdown Feature === ---
# If True, the computer will automatically shut down after the entire script finishes.
# Useful for leaving long simulations running overnight.
AUTO_SHUTDOWN = False

# --- === Visualization Parameters === ---
VISUALIZE_GRADIENTS = False       # Master toggle for scatter plots
VISUALIZE_EVERY_N_ROUNDS = 10    # Save scatter plot every N rounds

