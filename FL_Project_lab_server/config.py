"""
Central configuration file for the Federated Learning simulation.

"""

import torch
import os 

# --- === 1. Simulation Core Parameters === ---
NUM_ROUNDS = 1000           # Total number of federated learning rounds
NUM_CLIENTS = 30             # Total number of clients in the pool
MIN_CLIENTS_PER_ROUND = 15   # Minimum clients to select each round
MAX_CLIENTS_PER_ROUND = 28   # Maximum clients to select each round
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
BATCH_SIZE = 32           

# --- === 4. Local Training & LR Decay Parameters === ---
LOCAL_EPOCHS = 1
LEARNING_RATE = 0.01      
MOMENTUM = 0.8            
LR_DECAY_ENABLED = True
LR_DECAY_RATE = 0.99       # Decay multiplier per round
MIN_LR = 1e-4              # Minimum learning rate limit

# --- === 5. Server Aggregation & Momentum Parameters === ---
# Server Momentum acts as an inertia buff against massive late-stage accuracy drops.
SERVER_MOMENTUM_ENABLED = False      # If False, works as standard FL.
SERVER_MOMENTUM = 0.7               # FedAvgM momentum parameter (velocity to keep).
SERVER_LEARNING_RATE = 1.0          # Server step size.

# --- === 6. Aegis Adaptive Thresholding Parameters === ---
# The rejection cutoff: T = median_distance + (k * MAD)
ADAPTIVE_THRESHOLD_ENABLED = True   # Set False to use fixed OUTLIER_SENSITIVITY
OUTLIER_SENSITIVITY = 3.0           # Fixed fallback when adaptive is disabled
# Strategy A: Round-Based Decay
K_MAX = 6.0                         # Initial (loose) threshold multiplier
K_MIN = 2.0                         # Final (strict) threshold multiplier (for IID)
WARMUP_ROUNDS = 300                 # Rounds over which k linearly decays and then stops decaying
K_SAFE_FLOOR = 4                  # Absolute minimum k (vital for Non-IID cases)
# Strategy C: Variance-Normalized
VARIANCE_SENSITIVITY = 3.0          # Relaxation scale when updates are highly dispersed
PASS1_COS_THRESHOLD = -0.3           # Pass 1 directional screen threshold (Aegis)
                                    # 0.0 = aggressive (catches honest non-IID clients)
                                    # -0.3 = moderate (catches sign-flip but preserves borderline honest)
                                    # -0.5 = lenient (only catches strongly adversarial)
COSINE_PENALTY_WEIGHT = 10.0         # Multiplier α for single-round cosine penalty P_k in credit score
RWA_EPSILON = 1e-9                  # Numerical stability constant

# --- === 7. FoolsGold Defense Parameters === ---
FOOLSGOLD_KAPPA = 1.0               # Higher bounds -> more aggressive suppression

# --- === 7b. Aegis Cross-Round Reputation Parameters (ALIE Defense) === ---
REPUTATION_ENABLED = True           # Toggle cross-round reputation tracking
REPUTATION_DECAY = 0.95             # EMA decay factor γ (0.95 ≈ 20-round memory)
REPUTATION_WEIGHT = 20.0            # Scaling factor λ for reputation term in credit score denominator

# --- === 8. Byzantine Attack Parameters === ---
FRACTION_BYZANTINE = 0.30           # Ratio of Byzantine clients in the total pool

# Options: 'none', 'sign_flip', 'additive_noise', 'pure_additive_noise', 'catastrophic_noise', 'label_flip', 'orthogonal', 'volume_spam', 'sybil', 'alie', 'ipm'
ATTACK_TYPE = 'label_flip'
ATTACK_PROBABILITY = 1.0            # Probability of a traitor attacking whilst in their window
ATTACK_NOISE_STD = 2.0              # Multiplier for additive_noise magnitudes
NUM_SYBILS_PER_ATTACKER = 2         # Fake identities per traitor during a Sybil attack

# ALIE Specifics ("A Little Is Enough")
ALIE_Z = 1.0                        # Set to None for algorithmic paper formula.
ALIE_USE_OMNISCIENT = False         # True = use all gradients, False = only traitor gradients

# IPM Specifics ("Fall of Empires" — Xie, Koyejo, Gupta, UAI 2020)
# ε controls the stealth-vs-impact tradeoff:
#   ε = 1.0  → poisoned delta = -1.0 × μ̂ (full negation, cos ≈ -1, easily caught by cosine filter)
#   ε = 0.5  → poisoned delta = -0.5 × μ̂ (half magnitude, cos ≈ -1, still caught but smaller norm)
#   ε = 0.1  → poisoned delta = -0.1 × μ̂ (very small norm, cos ≈ -1, tiny per-round impact)
#   ε = 0.01 → poisoned delta = -0.01 × μ̂ (near-zero norm, direction ambiguous, ALIE-like stealth)
IPM_EPSILON = 0.5                   # Scaling factor τ: each Byzantine submits -τ × μ. ByzFL reference default: 2.0.
IPM_USE_OMNISCIENT = True           # True = compute μ over ALL clients' deltas; False = Byzantine clients' own deltas only

# --- === 9. Environment & System Limits === ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_PARALLEL_CLIENTS = 1            # 1 for Serial/GPU, None for maximum CPU threads
DATALOADER_WORKERS = 0              # MUST stay 0 if Multiprocessing clients > 0

# --- === 10. Evaluation & Early Stopping === ---
EVALUATE_EVERY_N_ROUNDS = 1
EARLY_STOPPING_ENABLED = True
PATIENCE = 70                       # Wait time for loss improvement 
MIN_DELTA = 0.001                   # Substantial enough loss drop to reset patience

# --- === 11. Output, Visualization & System === ---
if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or os.path.exists('/kaggle/working'):
    RESULTS_DIR = '/kaggle/working/saved_models' # Kaggle Linux
elif 'MODAL_ENVIRONMENT' in os.environ or 'MODAL_IMAGE_ID' in os.environ:
    RESULTS_DIR = './saved_models'               # Modal
elif os.name == 'nt':
    RESULTS_DIR = r'D:\IITD\MTP 2\Results'       # Local Windows PC
else:
    RESULTS_DIR = './results'                    # Lab Server (Linux)


VISUALIZE_GRADIENTS = False         # Master toggle for distance scatters
VISUALIZE_EVERY_N_ROUNDS = 10       
PLOT_SMOOTHING_WEIGHT = 0.85        # Exponential Moving Average weight (0.0 to 1.0)
AUTO_SHUTDOWN = False               # Automatically turn off machine post-run
