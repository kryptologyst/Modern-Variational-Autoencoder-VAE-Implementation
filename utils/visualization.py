"""Visualization utilities for VAE results."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional
import seaborn as sns
from config import config

class VAEVisualizer:
    """Handles visualization of VAE training and results."""
    
    def __init__(self, vae_model, data_loader):
        self.vae = vae_model
        self.data_loader = data_loader
        
    def plot_latent_space(self, test_data: np.ndarray, test_labels: np.ndarray = None, 
                         save_path: str = None) -> None:
        """Plot the latent space representation."""
        # Encode test data
        z_mean, _, _ = self.vae.encoder(test_data)
        z_mean = z_mean.numpy()
        
        plt.figure(figsize=(12, 10))
        
        if test_labels is not None and z_mean.shape[1] == 2:
            # Color by class labels
            scatter = plt.scatter(z_mean[:, 0], z_mean[:, 1], 
                                c=test_labels, cmap='tab10', alpha=0.6)
            plt.colorbar(scatter)
        else:
            plt.scatter(z_mean[:, 0], z_mean[:, 1], alpha=0.6)
        
        plt.title('VAE Latent Space Representation')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_reconstructions(self, test_data: np.ndarray, num_samples: int = 10,
                           save_path: str = None) -> None:
        """Plot original vs reconstructed images."""
        # Get random samples
        indices = np.random.choice(len(test_data), num_samples, replace=False)
        samples = test_data[indices]
        
        # Get reconstructions
        reconstructions = self.vae(samples).numpy()
        
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
        
        for i in range(num_samples):
            # Original
            if len(samples[i]) == 784:  # MNIST
                img_orig = samples[i].reshape(28, 28)
                img_recon = reconstructions[i].reshape(28, 28)
            else:  # CIFAR-10
                img_orig = samples[i].reshape(32, 32, 3)
                img_recon = reconstructions[i].reshape(32, 32, 3)
            
            axes[0, i].imshow(img_orig, cmap='gray' if len(samples[i]) == 784 else None)
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(img_recon, cmap='gray' if len(samples[i]) == 784 else None)
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_samples(self, num_samples: int = 16, save_path: str = None) -> None:
        """Generate new samples from the latent space."""
        # Sample from standard normal distribution
        z_sample = tf.random.normal(shape=(num_samples, self.vae.encoder.latent_dim))
        
        # Decode samples
        generated = self.vae.decoder(z_sample).numpy()
        
        # Plot generated samples
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        axes = axes.ravel()
        
        for i in range(num_samples):
            if len(generated[i]) == 784:  # MNIST
                img = generated[i].reshape(28, 28)
                axes[i].imshow(img, cmap='gray')
            else:  # CIFAR-10
                img = generated[i].reshape(32, 32, 3)
                axes[i].imshow(img)
            
            axes[i].axis('off')
        
        plt.suptitle('Generated Samples')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self, history: dict, save_path: str = None) -> None:
        """Plot training history."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Total loss
        axes[0].plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_title('Total Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Reconstruction loss
        axes[1].plot(history['reconstruction_loss'], label='Training Recon Loss')
        if 'val_reconstruction_loss' in history:
            axes[1].plot(history['val_reconstruction_loss'], label='Validation Recon Loss')
        axes[1].set_title('Reconstruction Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # KL loss
        axes[2].plot(history['kl_loss'], label='Training KL Loss')
        if 'val_kl_loss' in history:
            axes[2].plot(history['val_kl_loss'], label='Validation KL Loss')
        axes[2].set_title('KL Divergence Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_latent_interpolation(self, start_img: np.ndarray, end_img: np.ndarray,
                                 num_steps: int = 10, save_path: str = None) -> None:
        """Plot interpolation between two images in latent space."""
        # Encode start and end images
        start_z, _, _ = self.vae.encoder(start_img.reshape(1, -1))
        end_z, _, _ = self.vae.encoder(end_img.reshape(1, -1))
        
        # Create interpolation
        alphas = np.linspace(0, 1, num_steps)
        interpolated_images = []
        
        for alpha in alphas:
            z_interp = (1 - alpha) * start_z + alpha * end_z
            img_interp = self.vae.decoder(z_interp).numpy()
            interpolated_images.append(img_interp[0])
        
        # Plot interpolation
        fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 2, 2))
        
        for i, img in enumerate(interpolated_images):
            if len(img) == 784:  # MNIST
                img_reshaped = img.reshape(28, 28)
                axes[i].imshow(img_reshaped, cmap='gray')
            else:  # CIFAR-10
                img_reshaped = img.reshape(32, 32, 3)
                axes[i].imshow(img_reshaped)
            
            axes[i].axis('off')
            axes[i].set_title(f'α={alphas[i]:.1f}')
        
        plt.suptitle('Latent Space Interpolation')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
