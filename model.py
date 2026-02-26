"""
Defines the machine learning model architecture.
We use a simple Multi-Layer Perceptron (MLP) for MNIST.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config

# --- === 1. MLP (for MNIST) === ---
class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) model for MNIST classification.
    It has two hidden layers.
    """
    def __init__(self):
        super(MLP, self).__init__()
        # MNIST images are 28x28 = 784 pixels
        self.fc1 = nn.Linear(784, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 64)   # Second hidden layer
        self.fc3 = nn.Linear(64, 10)    # Output layer (10 classes for MNIST)

    def forward(self, x):
        """
        Forward pass of the network.
        """
        # Flatten the 28x28 image into a 784-dim vector
        x = x.view(-1, 784)
        # Apply ReLU activation function after first hidden layer
        x = F.relu(self.fc1(x))
        # Apply ReLU activation function after second hidden layer
        x = F.relu(self.fc2(x))
        # No activation on the final layer (LogSoftmax will be applied by the loss function)
        x = self.fc3(x)
        return x

# --- === 2. CNN (for CIFAR-10) [ ] === ---
class CNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) based on LeNet-5
    for CIFAR-10 classification.
    Input shape: (Batch_size, 3, 32, 32)
    """
    def __init__(self):
        super(CNN, self).__init__()
        # --- Convolutional Layers ---
        # 1. Input: 3 channels (RGB), Output: 6 channels, Kernel: 5x5
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        # 2. Max pooling 2x2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 3. Input: 6 channels, Output: 16 channels, Kernel: 5x5
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        # --- Fully Connected (Linear) Layers ---
        # We need to calculate the flattened size
        # Input: (3, 32, 32)
        # conv1 -> (6, 28, 28)
        # pool1 -> (6, 14, 14)
        # conv2 -> (16, 10, 10)
        # pool2 -> (16, 5, 5)
        # Flattened size = 16 * 5 * 5 = 400
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 10 output classes

    def forward(self, x):
        """
        Forward pass of the network.
        """
        # --- Convolutional path ---
        # x starts as (Batch, 3, 32, 32)
        x = self.pool(F.relu(self.conv1(x))) # -> (Batch, 6, 14, 14)
        x = self.pool(F.relu(self.conv2(x))) # -> (Batch, 16, 5, 5)
        
        # --- Flattening ---
        # Flatten all dimensions except batch
        x = x.view(-1, 16 * 5 * 5) # -> (Batch, 400)
        
        # --- Fully Connected path ---
        x = F.relu(self.fc1(x)) # -> (Batch, 120)
        x = F.relu(self.fc2(x)) # -> (Batch, 84)
        x = self.fc3(x)         # -> (Batch, 10)
        return x
    

# --- === 3. Improved CNN (for CIFAR-10) === ---
class ImprovedCNN(nn.Module):
    """
    An improved CNN for CIFAR-10 with BatchNorm and Dropout.
    ~290K parameters — roughly 5x larger than LeNet-5 but still lightweight.
    
    Architecture: Two double-conv blocks with BatchNorm, followed by
    a fully connected classifier with Dropout for regularization.
    
    Input shape: (Batch_size, 3, 32, 32)
    """
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        
        # --- Block 1: 3 → 32 channels ---
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)    # (B, 32, 32, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)   # (B, 32, 32, 32)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)                            # (B, 32, 16, 16)
        
        # --- Block 2: 32 → 64 channels ---
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)   # (B, 64, 16, 16)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)   # (B, 64, 16, 16)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)                            # (B, 64, 8, 8)
        
        # --- Classifier ---
        # Flattened: 64 * 8 * 8 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))    # (B, 32, 32, 32)
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))  # (B, 32, 16, 16)
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))    # (B, 64, 16, 16)
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))  # (B, 64, 8, 8)
        
        # Classifier
        x = x.view(-1, 64 * 8 * 8)            # (B, 4096)
        x = F.relu(self.fc1(x))                # (B, 256)
        x = self.dropout(x)
        x = self.fc2(x)                        # (B, 10)
        return x


def get_model():
    """
    Helper factory function to instantiate the correct model
    based on the config file.
    
    Supported MODEL_TYPE values:
    - 'MLP': Simple MLP for MNIST (2 hidden layers)
    - 'CNN': LeNet-5 for CIFAR-10 (~62K params)
    - 'ImprovedCNN': Improved CNN for CIFAR-10 (~290K params, BatchNorm + Dropout)
    """
    if config.MODEL_TYPE == 'MLP':
        if config.DATASET_NAME != 'MNIST':
            print(f"Warning: Using an MLP model for {config.DATASET_NAME}. This may perform poorly.")
        return MLP()
        
    elif config.MODEL_TYPE == 'CNN':
        if config.DATASET_NAME != 'CIFAR10':
            print(f"Warning: Using a CNN model for {config.DATASET_NAME}. Check input dimensions.")
        return CNN()
    
    elif config.MODEL_TYPE == 'ImprovedCNN':
        if config.DATASET_NAME != 'CIFAR10':
            print(f"Warning: Using ImprovedCNN model for {config.DATASET_NAME}. Check input dimensions.")
        return ImprovedCNN()
        
    else:
        raise ValueError(f"Unknown MODEL_TYPE in config: {config.MODEL_TYPE}")