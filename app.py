"""Streamlit web application for VAE model interaction."""

import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import os
import json
from datetime import datetime

from config import config
from models.vae import create_vae
from data.data_loader import DataLoader, mock_db
from utils.visualization import VAEVisualizer

# Configure Streamlit page
st.set_page_config(
    page_title="VAE Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_data(dataset_name, latent_dim):
    """Load model and data with caching."""
    # Determine input dimension
    if dataset_name in ["mnist", "fashion_mnist"]:
        input_dim = 784
    else:
        input_dim = 3072
    
    # Create model
    vae = create_vae(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=config.hidden_dims,
        beta=config.beta
    )
    
    # Load data
    data_loader = DataLoader(dataset_name)
    x_train, x_test, y_train, y_test = data_loader.load_data()
    
    return vae, data_loader, (x_train, x_test, y_train, y_test)

def plot_to_streamlit(fig):
    """Convert matplotlib figure to streamlit."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    st.image(buf, use_column_width=True)
    plt.close(fig)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Variational Autoencoder Explorer</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    dataset_name = st.sidebar.selectbox(
        "Dataset",
        ["mnist", "fashion_mnist", "cifar10"],
        help="Choose the dataset for training/inference"
    )
    
    latent_dim = st.sidebar.slider(
        "Latent Dimension",
        min_value=2, max_value=128, value=32,
        help="Dimension of the latent space"
    )
    
    beta = st.sidebar.slider(
        "Beta (KL Weight)",
        min_value=0.1, max_value=10.0, value=1.0, step=0.1,
        help="Weight for KL divergence loss"
    )
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Overview", "🚀 Training", "🎨 Generation", "📊 Analysis", "📈 Experiments"
    ])
    
    with tab1:
        st.header("Welcome to VAE Explorer!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 What is a Variational Autoencoder?
            
            A Variational Autoencoder (VAE) is a powerful generative model that:
            
            - **Encodes** input data into a probabilistic latent space
            - **Decodes** latent representations back to data space
            - **Generates** new samples by sampling from the latent space
            - **Learns** meaningful representations for dimensionality reduction
            
            ### 🔧 Features of this Implementation:
            
            - Modern TensorFlow 2.x architecture
            - Multiple dataset support (MNIST, Fashion-MNIST, CIFAR-10)
            - Interactive visualization and generation
            - Experiment tracking and comparison
            - Real-time training monitoring
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Current Configuration:
            """)
            
            st.markdown(f"""
            <div class="metric-card">
                <strong>Dataset:</strong> {dataset_name.upper()}<br>
                <strong>Latent Dimension:</strong> {latent_dim}<br>
                <strong>Beta Parameter:</strong> {beta}<br>
                <strong>Architecture:</strong> {' → '.join(map(str, config.hidden_dims))}<br>
                <strong>Batch Size:</strong> {config.batch_size}
            </div>
            """, unsafe_allow_html=True)
            
            # Show recent experiments
            experiments = mock_db.list_experiments()
            if experiments:
                st.markdown("### 🧪 Recent Experiments:")
                for exp in experiments[-3:]:  # Show last 3
                    st.markdown(f"- **{exp['id']}**: Loss = {exp['results'].get('final_val_loss', 'N/A'):.4f}")
    
    with tab2:
        st.header("🚀 Model Training")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Training Configuration")
            
            epochs = st.number_input("Epochs", min_value=1, max_value=200, value=50)
            batch_size = st.number_input("Batch Size", min_value=16, max_value=512, value=128)
            learning_rate = st.number_input("Learning Rate", min_value=1e-5, max_value=1e-2, 
                                          value=1e-3, format="%.2e")
            
            if st.button("🚀 Start Training", key="train_button"):
                with st.spinner("Training in progress..."):
                    try:
                        # Update config
                        config.epochs = epochs
                        config.batch_size = batch_size
                        config.learning_rate = learning_rate
                        config.latent_dim = latent_dim
                        config.beta = beta
                        
                        # Load model and data
                        vae, data_loader, (x_train, x_test, y_train, y_test) = load_model_and_data(
                            dataset_name, latent_dim
                        )
                        
                        # Create datasets
                        train_dataset, test_dataset = data_loader.create_datasets(batch_size)
                        
                        # Compile model
                        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                        vae.compile(optimizer=optimizer)
                        
                        # Training progress
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Simple training loop for demo
                        history = {'loss': [], 'val_loss': [], 'reconstruction_loss': [], 'kl_loss': []}
                        
                        for epoch in range(min(epochs, 10)):  # Limit for demo
                            # Train for one epoch
                            epoch_loss = []
                            for batch in train_dataset.take(10):  # Limit batches for demo
                                with tf.GradientTape() as tape:
                                    z_mean, z_log_var, z = vae.encoder(batch)
                                    reconstruction = vae.decoder(z)
                                    
                                    recon_loss = tf.reduce_mean(
                                        tf.reduce_sum(
                                            tf.keras.losses.binary_crossentropy(batch, reconstruction),
                                            axis=1
                                        )
                                    )
                                    
                                    kl_loss = -0.5 * tf.reduce_mean(
                                        tf.reduce_sum(
                                            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                                            axis=1
                                        )
                                    )
                                    
                                    total_loss = recon_loss + beta * kl_loss
                                
                                grads = tape.gradient(total_loss, vae.trainable_weights)
                                vae.optimizer.apply_gradients(zip(grads, vae.trainable_weights))
                                epoch_loss.append(total_loss.numpy())
                            
                            avg_loss = np.mean(epoch_loss)
                            history['loss'].append(avg_loss)
                            
                            progress_bar.progress((epoch + 1) / min(epochs, 10))
                            status_text.text(f"Epoch {epoch + 1}/{min(epochs, 10)} - Loss: {avg_loss:.4f}")
                        
                        st.success("Training completed!")
                        
                        # Save experiment
                        experiment_id = f"streamlit_{dataset_name}_{datetime.now().strftime('%H%M%S')}"
                        mock_db.save_experiment(
                            experiment_id,
                            {'dataset': dataset_name, 'latent_dim': latent_dim, 'beta': beta},
                            {'final_loss': history['loss'][-1]}
                        )
                        
                        # Store in session state
                        st.session_state['trained_vae'] = vae
                        st.session_state['data_loader'] = data_loader
                        st.session_state['training_data'] = (x_train, x_test, y_train, y_test)
                        
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
        
        with col2:
            st.markdown("### 📊 Training Tips")
            st.info("""
            **Hyperparameter Guidelines:**
            
            - **Latent Dim**: Start with 32-64 for images
            - **Beta**: 1.0 for standard VAE, <1.0 for β-VAE
            - **Learning Rate**: 1e-3 is usually good
            - **Epochs**: 50-100 for good results
            
            **Architecture Notes:**
            - Deeper networks capture more complex patterns
            - Batch normalization helps training stability
            - Dropout prevents overfitting
            """)
    
    with tab3:
        st.header("🎨 Sample Generation & Manipulation")
        
        if 'trained_vae' in st.session_state:
            vae = st.session_state['trained_vae']
            data_loader = st.session_state['data_loader']
            x_train, x_test, y_train, y_test = st.session_state['training_data']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎲 Random Generation")
                
                num_samples = st.slider("Number of samples", 4, 16, 9)
                
                if st.button("Generate Random Samples"):
                    # Generate random samples
                    z_sample = tf.random.normal(shape=(num_samples, latent_dim))
                    generated = vae.decoder(z_sample).numpy()
                    
                    # Display samples
                    cols = st.columns(3)
                    for i in range(num_samples):
                        col_idx = i % 3
                        with cols[col_idx]:
                            if dataset_name in ["mnist", "fashion_mnist"]:
                                img = generated[i].reshape(28, 28)
                                fig, ax = plt.subplots(figsize=(3, 3))
                                ax.imshow(img, cmap='gray')
                                ax.axis('off')
                                plot_to_streamlit(fig)
                            else:
                                img = generated[i].reshape(32, 32, 3)
                                img = np.clip(img, 0, 1)
                                st.image(img, width=100)
            
            with col2:
                st.subheader("🔄 Reconstructions")
                
                if st.button("Show Reconstructions"):
                    # Get random test samples
                    indices = np.random.choice(len(x_test), 6, replace=False)
                    samples = x_test[indices]
                    reconstructions = vae(samples).numpy()
                    
                    # Display original vs reconstructed
                    for i in range(6):
                        col_orig, col_recon = st.columns(2)
                        
                        with col_orig:
                            if i == 0:
                                st.markdown("**Original**")
                            if dataset_name in ["mnist", "fashion_mnist"]:
                                img = samples[i].reshape(28, 28)
                                fig, ax = plt.subplots(figsize=(2, 2))
                                ax.imshow(img, cmap='gray')
                                ax.axis('off')
                                plot_to_streamlit(fig)
                        
                        with col_recon:
                            if i == 0:
                                st.markdown("**Reconstructed**")
                            if dataset_name in ["mnist", "fashion_mnist"]:
                                img = reconstructions[i].reshape(28, 28)
                                fig, ax = plt.subplots(figsize=(2, 2))
                                ax.imshow(img, cmap='gray')
                                ax.axis('off')
                                plot_to_streamlit(fig)
            
            # Latent space exploration (only for 2D)
            if latent_dim == 2:
                st.subheader("🗺️ Latent Space Exploration")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    z1 = st.slider("Latent Dimension 1", -3.0, 3.0, 0.0, 0.1)
                    z2 = st.slider("Latent Dimension 2", -3.0, 3.0, 0.0, 0.1)
                
                with col2:
                    z_point = tf.constant([[z1, z2]], dtype=tf.float32)
                    generated = vae.decoder(z_point).numpy()
                    
                    if dataset_name in ["mnist", "fashion_mnist"]:
                        img = generated[0].reshape(28, 28)
                        fig, ax = plt.subplots(figsize=(4, 4))
                        ax.imshow(img, cmap='gray')
                        ax.set_title(f"Generated at z=({z1:.1f}, {z2:.1f})")
                        ax.axis('off')
                        plot_to_streamlit(fig)
        
        else:
            st.warning("Please train a model first in the Training tab!")
    
    with tab4:
        st.header("📊 Model Analysis")
        
        if 'trained_vae' in st.session_state:
            vae = st.session_state['trained_vae']
            x_train, x_test, y_train, y_test = st.session_state['training_data']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Latent Space Visualization")
                
                if latent_dim == 2:
                    # Encode test data
                    z_mean, _, _ = vae.encoder(x_test[:1000])
                    z_mean = z_mean.numpy()
                    
                    # Create interactive plot
                    fig = px.scatter(
                        x=z_mean[:, 0], y=z_mean[:, 1],
                        color=y_test[:1000].flatten(),
                        title="Latent Space Representation",
                        labels={'x': 'Latent Dim 1', 'y': 'Latent Dim 2'},
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Latent space visualization is only available for 2D latent spaces.")
            
            with col2:
                st.subheader("📈 Model Metrics")
                
                # Calculate reconstruction error
                sample_indices = np.random.choice(len(x_test), 100, replace=False)
                samples = x_test[sample_indices]
                reconstructions = vae(samples).numpy()
                
                mse = np.mean((samples - reconstructions) ** 2)
                mae = np.mean(np.abs(samples - reconstructions))
                
                st.metric("Mean Squared Error", f"{mse:.6f}")
                st.metric("Mean Absolute Error", f"{mae:.6f}")
                
                # Model info
                st.markdown("### 🏗️ Model Architecture")
                total_params = vae.count_params()
                encoder_params = vae.encoder.count_params()
                decoder_params = vae.decoder.count_params()
                
                st.markdown(f"""
                - **Total Parameters**: {total_params:,}
                - **Encoder Parameters**: {encoder_params:,}
                - **Decoder Parameters**: {decoder_params:,}
                - **Latent Dimension**: {latent_dim}
                - **Beta Parameter**: {beta}
                """)
        
        else:
            st.warning("Please train a model first in the Training tab!")
    
    with tab5:
        st.header("📈 Experiment Tracking")
        
        experiments = mock_db.list_experiments()
        
        if experiments:
            st.subheader("🧪 All Experiments")
            
            # Create experiment comparison table
            exp_data = []
            for exp in experiments:
                full_exp = mock_db.get_experiment(exp['id'])
                if full_exp:
                    exp_data.append({
                        'ID': exp['id'],
                        'Dataset': full_exp['config'].get('dataset', 'N/A'),
                        'Latent Dim': full_exp['config'].get('latent_dim', 'N/A'),
                        'Beta': full_exp['config'].get('beta', 'N/A'),
                        'Final Loss': f"{exp['results'].get('final_loss', 0):.4f}",
                        'Best Val Loss': f"{exp['results'].get('best_val_loss', exp['results'].get('final_loss', 0)):.4f}"
                    })
            
            if exp_data:
                import pandas as pd
                df = pd.DataFrame(exp_data)
                st.dataframe(df, use_container_width=True)
                
                # Best experiment
                best_exp = mock_db.get_best_experiment()
                if best_exp:
                    st.success(f"🏆 Best Experiment: **{best_exp['id']}** with validation loss: **{best_exp['results'].get('best_val_loss', best_exp['results'].get('final_loss', 0)):.4f}**")
            
            # Experiment comparison chart
            if len(experiments) > 1:
                st.subheader("📊 Loss Comparison")
                
                losses = [exp['results'].get('final_loss', 0) for exp in experiments]
                exp_names = [exp['id'].split('_')[-1] for exp in experiments]  # Use timestamp part
                
                fig = px.bar(
                    x=exp_names, y=losses,
                    title="Final Loss Comparison Across Experiments",
                    labels={'x': 'Experiment', 'y': 'Final Loss'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No experiments found. Train some models to see results here!")
        
        # Clear experiments button
        if st.button("🗑️ Clear All Experiments"):
            mock_db.experiments = []
            mock_db.models = {}
            mock_db.metrics = {}
            st.success("All experiments cleared!")
            st.experimental_rerun()

if __name__ == "__main__":
    main()
