"""Main training script for the VAE model."""

import tensorflow as tf
import numpy as np
import os
import argparse
from datetime import datetime
import json

from config import config
from models.vae import create_vae
from data.data_loader import DataLoader, mock_db
from utils.visualization import VAEVisualizer

# Enable mixed precision training
if config.mixed_precision:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

def setup_gpu():
    """Configure GPU settings."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus and config.use_gpu:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU: {len(gpus)} device(s) available")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("Using CPU")

class VAETrainer:
    """Handles VAE model training and evaluation."""
    
    def __init__(self, dataset_name: str = "mnist"):
        self.dataset_name = dataset_name
        self.data_loader = DataLoader(dataset_name)
        self.vae = None
        self.visualizer = None
        self.experiment_id = None
        
    def create_model(self):
        """Create and compile the VAE model."""
        # Adjust input dimension based on dataset
        if self.dataset_name in ["mnist", "fashion_mnist"]:
            input_dim = 784
        else:  # CIFAR-10
            input_dim = 3072
            
        self.vae = create_vae(
            input_dim=input_dim,
            latent_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
            beta=config.beta
        )
        
        # Use AdamW optimizer with learning rate scheduling
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=config.learning_rate,
            decay_steps=config.epochs * 1000,  # Approximate steps per epoch
            alpha=0.1
        )
        
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=1e-4
        )
        
        self.vae.compile(optimizer=optimizer)
        self.visualizer = VAEVisualizer(self.vae, self.data_loader)
        
    def setup_callbacks(self):
        """Setup training callbacks."""
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.model_dir, 'best_vae.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(lr_reducer)
        
        # TensorBoard logging
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(config.logs_dir, f'vae_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callbacks.append(tensorboard_callback)
        
        return callbacks
    
    def train(self):
        """Train the VAE model."""
        print("Loading data...")
        x_train, x_test, y_train, y_test = self.data_loader.load_data()
        train_dataset, test_dataset = self.data_loader.create_datasets()
        
        print("Creating model...")
        self.create_model()
        
        # Print model summary
        print("\nModel Architecture:")
        self.vae.encoder.build((None, x_train.shape[1]))
        self.vae.decoder.build((None, config.latent_dim))
        
        print(f"Encoder parameters: {self.vae.encoder.count_params():,}")
        print(f"Decoder parameters: {self.vae.decoder.count_params():,}")
        print(f"Total parameters: {self.vae.count_params():,}")
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Generate experiment ID
        self.experiment_id = f"vae_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\nStarting training for experiment: {self.experiment_id}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Training samples: {len(x_train):,}")
        print(f"Test samples: {len(x_test):,}")
        print(f"Latent dimension: {config.latent_dim}")
        print(f"Beta (KL weight): {config.beta}")
        
        # Train the model
        history = self.vae.fit(
            train_dataset,
            epochs=config.epochs,
            validation_data=test_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save experiment to mock database
        config_dict = {
            'dataset': self.dataset_name,
            'latent_dim': config.latent_dim,
            'hidden_dims': config.hidden_dims,
            'beta': config.beta,
            'learning_rate': config.learning_rate,
            'epochs': config.epochs,
            'batch_size': config.batch_size
        }
        
        final_results = {
            'final_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'best_val_loss': float(min(history.history['val_loss'])),
            'total_epochs': len(history.history['loss'])
        }
        
        mock_db.save_experiment(self.experiment_id, config_dict, final_results)
        
        # Save training history
        history_path = os.path.join(config.results_dir, f'{self.experiment_id}_history.json')
        with open(history_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            history_dict = {}
            for key, values in history.history.items():
                history_dict[key] = [float(v) for v in values]
            json.dump(history_dict, f, indent=2)
        
        print(f"\nTraining completed!")
        print(f"Final validation loss: {final_results['final_val_loss']:.4f}")
        print(f"Best validation loss: {final_results['best_val_loss']:.4f}")
        
        return history
    
    def evaluate_and_visualize(self):
        """Evaluate the model and create visualizations."""
        if self.vae is None:
            print("No trained model found. Please train first.")
            return
        
        print("Creating visualizations...")
        x_train, x_test, y_train, y_test = self.data_loader.load_data()
        
        # Create results directory for this experiment
        exp_results_dir = os.path.join(config.results_dir, self.experiment_id)
        os.makedirs(exp_results_dir, exist_ok=True)
        
        # Plot latent space
        self.visualizer.plot_latent_space(
            x_test[:1000], y_test[:1000],
            save_path=os.path.join(exp_results_dir, 'latent_space.png')
        )
        
        # Plot reconstructions
        self.visualizer.plot_reconstructions(
            x_test, num_samples=10,
            save_path=os.path.join(exp_results_dir, 'reconstructions.png')
        )
        
        # Generate new samples
        self.visualizer.generate_samples(
            num_samples=16,
            save_path=os.path.join(exp_results_dir, 'generated_samples.png')
        )
        
        # Plot interpolation between two random samples
        sample_indices = np.random.choice(len(x_test), 2, replace=False)
        self.visualizer.plot_latent_interpolation(
            x_test[sample_indices[0]], x_test[sample_indices[1]],
            save_path=os.path.join(exp_results_dir, 'interpolation.png')
        )
        
        print(f"Visualizations saved to: {exp_results_dir}")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train VAE model')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'fashion_mnist', 'cifar10'],
                       help='Dataset to use for training')
    parser.add_argument('--latent_dim', type=int, default=32,
                       help='Latent space dimension')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Beta parameter for KL divergence weight')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config.latent_dim = args.latent_dim
    config.beta = args.beta
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    
    # Setup GPU
    setup_gpu()
    
    # Create trainer and train
    trainer = VAETrainer(args.dataset)
    history = trainer.train()
    
    # Create visualizations
    trainer.evaluate_and_visualize()
    
    # Plot training history
    trainer.visualizer.plot_training_history(
        history.history,
        save_path=os.path.join(config.results_dir, trainer.experiment_id, 'training_history.png')
    )

if __name__ == "__main__":
    main()
