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
    


def get_model():
    """
    Helper factory function to instantiate the correct model
    based on the config file.
    """
    if config.MODEL_TYPE == 'MLP':
        # Check if we are using the wrong dataset
        if config.DATASET_NAME != 'MNIST':
            print(f"Warning: Using an MLP model for {config.DATASET_NAME}. This may perform poorly.")
        return MLP()
        
    elif config.MODEL_TYPE == 'CNN':
        # Check if we are using the wrong dataset
        if config.DATASET_NAME != 'CIFAR10':
            print(f"Warning: Using a CNN model for {config.DATASET_NAME}. Check input dimensions.")
        return CNN()
        
    else:
        raise ValueError(f"Unknown MODEL_TYPE in config: {config.MODEL_TYPE}")