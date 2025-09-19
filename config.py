"""Configuration settings for the VAE project."""

import os
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class VAEConfig:
    """Configuration class for VAE hyperparameters and settings."""
    
    # Model architecture
    input_dim: int = 784
    latent_dim: int = 32
    hidden_dims: Tuple[int, ...] = (512, 256, 128)
    
    # Training parameters
    batch_size: int = 128
    epochs: int = 100
    learning_rate: float = 1e-3
    beta: float = 1.0  # KL divergence weight
    
    # Data parameters
    image_size: Tuple[int, int] = (28, 28)
    num_channels: int = 1
    
    # Paths
    data_dir: str = "data"
    model_dir: str = "models"
    results_dir: str = "results"
    logs_dir: str = "logs"
    
    # Training options
    use_gpu: bool = True
    mixed_precision: bool = True
    gradient_clipping: float = 1.0
    
    # Evaluation
    num_samples: int = 16
    reconstruction_samples: int = 10
    
    # Wandb logging
    use_wandb: bool = False
    project_name: str = "vae-mnist"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for dir_path in [self.data_dir, self.model_dir, self.results_dir, self.logs_dir]:
            os.makedirs(dir_path, exist_ok=True)

# Global config instance
config = VAEConfig()
