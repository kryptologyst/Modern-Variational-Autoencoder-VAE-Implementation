# Modern Variational Autoencoder (VAE) Implementation

A comprehensive, modern implementation of Variational Autoencoders using TensorFlow 2.x with an interactive Streamlit web interface for training, generation, and analysis.

## Features

- **Modern Architecture**: Built with TensorFlow 2.x and latest best practices
- **Multiple Datasets**: Support for MNIST, Fashion-MNIST, and CIFAR-10
- **Interactive UI**: Beautiful Streamlit web interface for model interaction
- **Advanced Techniques**: 
  - Batch normalization and dropout for stability
  - Mixed precision training for efficiency
  - Learning rate scheduling and gradient clipping
  - β-VAE support for disentangled representations
- **Comprehensive Visualization**: Latent space exploration, reconstructions, and generation
- **Experiment Tracking**: Built-in experiment management and comparison
- **Production Ready**: Proper project structure, configuration, and documentation

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd 0079_Variational_autoencoder

# Install dependencies
pip install -r requirements.txt
```

### Training via Command Line

```bash
# Train on MNIST with default settings
python train.py

# Train on Fashion-MNIST with custom parameters
python train.py --dataset fashion_mnist --latent_dim 64 --beta 0.5 --epochs 100

# Train on CIFAR-10
python train.py --dataset cifar10 --latent_dim 128 --epochs 150
```

### Interactive Web Interface

```bash
# Launch the Streamlit app
streamlit run app.py
```

Then open your browser to `http://localhost:8501` to access the interactive interface.

## Web Interface Features

### Overview Tab
- Project introduction and current configuration
- Recent experiment summaries
- Architecture overview

### Training Tab
- Interactive model training with real-time progress
- Hyperparameter configuration
- Training tips and guidelines

### Generation Tab
- Random sample generation from latent space
- Image reconstruction visualization
- Interactive latent space exploration (for 2D latent spaces)

### Analysis Tab
- Latent space visualization with interactive plots
- Model performance metrics
- Architecture information

### Experiments Tab
- Experiment tracking and comparison
- Performance visualization across runs
- Best model identification

## Project Structure

```
0079_Variational_autoencoder/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.py                # Configuration settings
├── train.py                 # Command-line training script
├── app.py                   # Streamlit web interface
├── 0079.py                  # Original implementation (legacy)
├── models/
│   └── vae.py              # Modern VAE implementation
├── data/
│   └── data_loader.py      # Data loading and mock database
├── utils/
│   └── visualization.py    # Visualization utilities
├── data/                   # Dataset storage (created automatically)
├── models/                 # Saved model checkpoints
├── results/                # Training results and visualizations
└── logs/                   # TensorBoard logs
```

## 🔧 Configuration

The `config.py` file contains all configurable parameters:

```python
@dataclass
class VAEConfig:
    # Model architecture
    input_dim: int = 784
    latent_dim: int = 32
    hidden_dims: Tuple[int, ...] = (512, 256, 128)
    
    # Training parameters
    batch_size: int = 128
    epochs: int = 100
    learning_rate: float = 1e-3
    beta: float = 1.0  # KL divergence weight
    
    # ... more parameters
```

## Model Architecture

The VAE consists of:

1. **Encoder**: Maps input data to latent space parameters (μ, σ)
2. **Sampling Layer**: Samples from the latent distribution using reparameterization trick
3. **Decoder**: Reconstructs data from latent samples

### Key Improvements Over Basic VAE:

- **Batch Normalization**: Stabilizes training and improves convergence
- **Dropout**: Prevents overfitting and improves generalization
- **Gradient Clipping**: Prevents exploding gradients
- **Learning Rate Scheduling**: Cosine decay for better convergence
- **Mixed Precision**: Faster training with maintained accuracy
- **β-VAE**: Controllable disentanglement via β parameter

## Training Options

### Command Line Arguments:

- `--dataset`: Choose from 'mnist', 'fashion_mnist', 'cifar10'
- `--latent_dim`: Latent space dimension (default: 32)
- `--beta`: KL divergence weight for β-VAE (default: 1.0)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Training batch size (default: 128)

### Example Training Commands:

```bash
# Standard VAE on MNIST
python train.py --dataset mnist --latent_dim 32 --epochs 100

# β-VAE for disentangled representations
python train.py --dataset fashion_mnist --beta 0.5 --latent_dim 64

# High-capacity model for CIFAR-10
python train.py --dataset cifar10 --latent_dim 128 --epochs 200
```

## Visualization Features

- **Latent Space Plots**: 2D visualization of encoded data points
- **Reconstructions**: Original vs. reconstructed image comparisons
- **Generated Samples**: New samples from random latent vectors
- **Interpolation**: Smooth transitions between images in latent space
- **Training History**: Loss curves and metric tracking

## Experiment Tracking

The built-in mock database tracks:
- Model configurations and hyperparameters
- Training metrics and final performance
- Best performing models across experiments
- Experiment comparison and visualization

## Supported Datasets

### MNIST
- **Size**: 28×28 grayscale images
- **Classes**: 10 (digits 0-9)
- **Samples**: 60K training, 10K test

### Fashion-MNIST
- **Size**: 28×28 grayscale images
- **Classes**: 10 (clothing items)
- **Samples**: 60K training, 10K test

### CIFAR-10
- **Size**: 32×32 color images
- **Classes**: 10 (objects/animals)
- **Samples**: 50K training, 10K test

## Advanced Features

### β-VAE Implementation
Control the trade-off between reconstruction quality and latent space structure:
- `β = 1.0`: Standard VAE
- `β < 1.0`: Emphasizes reconstruction
- `β > 1.0`: Emphasizes disentanglement

### Mixed Precision Training
Automatically enabled for faster training on compatible GPUs while maintaining model accuracy.

### Learning Rate Scheduling
Cosine decay schedule for optimal convergence:
```python
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=config.learning_rate,
    decay_steps=config.epochs * steps_per_epoch,
    alpha=0.1
)
```

## Performance Tips

1. **Latent Dimension**: Start with 32-64 for images, adjust based on complexity
2. **β Parameter**: Use 0.5-2.0 for β-VAE experiments
3. **Learning Rate**: 1e-3 works well for most cases
4. **Batch Size**: Larger batches (128-256) generally improve stability
5. **Architecture**: Deeper networks capture more complex patterns

## Example Results

After training, you'll find in the `results/` directory:
- `latent_space.png`: Visualization of the learned latent space
- `reconstructions.png`: Original vs. reconstructed images
- `generated_samples.png`: New samples from the model
- `interpolation.png`: Smooth transitions between images
- `training_history.png`: Loss curves and metrics

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TensorFlow team for the excellent deep learning framework
- Streamlit team for the amazing web app framework
- Original VAE paper: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- β-VAE paper: [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)

## Support

If you encounter any issues or have questions:
1. Check the [Issues](../../issues) page
2. Create a new issue with detailed description
3. Include your environment details and error messages



# Modern-Variational-Autoencoder-VAE-Implementation
