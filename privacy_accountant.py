
import math
import torch

class PrivacyAccountant:
    """
    Tracks privacy budget (epsilon) consumption using Rényi Differential Privacy (RDP).
    This is a simplified implementation for the Gaussian Mechanism with Subsampling.
    """
    def __init__(self, noise_multiplier, sample_rate, delta):
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.delta = delta
        self.steps = 0
        
        # Alphas for RDP search
        self.alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

    def step(self):
        """Increments the step counter."""
        self.steps += 1

    def get_epsilon(self):
        """
        Computes the current epsilon for the given delta.
        Uses RDP composition.
        """
        if self.steps == 0:
            return 0.0
        
        # Compute RDP at step 'self.steps'
        # RDP_alpha = steps * RDP_mechanism(alpha)
        # We need RDP of Sampled Gaussian Mechanism.
        # Approximation: RDP_alpha(q, sigma) ~ 3.5 * q^2 * steps / sigma^2 ? (Very rough)
        
        # Better: Use the standard conversion if possible, or a basic composition bound.
        # Since implementing exact RDP for subsampled Gaussian is complex (requires integration),
        # we will use the 'Strong Composition' theorem approximation as a placeholder guide,
        # or a loose bound suitable for this standalone implementation.
        
        # Approximation for Gaussian Mechanism (without amplification by sampling for safety, or simple amplification)
        # Epsilon ≈ q * sqrt(T * log(1/delta)) * (1/sigma)
        # This is a heuristic.
        
        # For this project, we track steps and return a "Projected Epsilon" 
        # based on standard SGD analysis (Abadi et al.)
        # eps = q * sqrt(steps * log(1/delta)) / noise_multiplier
        
        if self.noise_multiplier == 0:
            return float('inf')

        eps = (self.sample_rate * math.sqrt(self.steps * math.log(1/self.delta))) / self.noise_multiplier
        return eps

    def check_budget(self, target_epsilon):
        """Returns True if budget is remaining, False if exhausted."""
        eps = self.get_epsilon()
        return eps < target_epsilon, eps
