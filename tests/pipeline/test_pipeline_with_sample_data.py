"""
Test Pipeline with Sample Data

This script demonstrates how to use the pipeline with a small sample dataset
without needing to connect to BigQuery.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_pipeline')

# Import our components
from ember_ml.models.optimized_rbm import OptimizedRBM
from ember_ml.core.stride_aware_cfc import (
    create_liquid_network_with_motor_neuron,
    create_lstm_gated_liquid_network
)


def generate_sample_data(n_samples=1000, n_features=20, n_classes=2):
    """
    Generate a sample dataset for testing.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    logger.info(f"Generating sample data with {n_samples} samples and {n_features} features")
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=n_features // 10,
        n_classes=n_classes,
        random_state=42
    )
    
    # Split into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42
    )
    
    logger.info(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_rbm(X_train, n_hidden=64, epochs=10, use_gpu=True):
    """
    Train an RBM on the training data.
    
    Args:
        X_train: Training data
        n_hidden: Number of hidden units
        epochs: Number of training epochs
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Trained RBM
    """
    logger.info(f"Training RBM with {n_hidden} hidden units for {epochs} epochs")
    
    # Create RBM
    rbm = OptimizedRBM(
        n_visible=X_train.shape[1],
        n_hidden=n_hidden,
        learning_rate=0.01,
        momentum=0.5,
        weight_decay=0.0001,
        batch_size=100,
        use_binary_states=False,
        use_gpu=use_gpu,
        verbose=True
    )
    
    # Define a generator to yield data in batches
    def data_generator(data, batch_size=100):
        # Shuffle data
        indices = np.random.permutation(len(data))
        data = data[indices]
        
        for i in range(0, len(data), batch_size):
            yield data[i:i+batch_size]
    
    # Train RBM
    training_errors = rbm.train_in_chunks(
        data_generator(X_train, batch_size=100),
        epochs=epochs,
        k=1
    )
    
    logger.info(f"RBM training completed with final error: {training_errors[-1]:.4f}")
    
    return rbm, training_errors


def extract_rbm_features(rbm, data):
    """
    Extract features from trained RBM.
    
    Args:
        rbm: Trained RBM
        data: Input data
        
    Returns:
        RBM features
    """
    logger.info(f"Extracting RBM features from {len(data)} samples")
    
    # Define a generator to yield data in batches
    def data_generator(data, batch_size=1000):
        for i in range(0, len(data), batch_size):
            yield data[i:i+batch_size]
    
    # Extract features
    rbm_features = rbm.transform_in_chunks(
        data_generator(data, batch_size=1000)
    )
    
    logger.info(f"Extracted {rbm_features.shape[1]} RBM features")
    
    return rbm_features


def train_liquid_network(X_train, y_train, X_val, y_val, input_dim, network_type='standard', epochs=50):
    """
    Train a liquid neural network.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        input_dim: Input dimension
        network_type: Type of network ('standard' or 'lstm_gated')
        epochs: Number of training epochs
        
    Returns:
        Trained liquid network
    """
    logger.info(f"Training {network_type} liquid network for {epochs} epochs")
    
    # Reshape features for sequence input
    X_train_seq = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_val_seq = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
    
    # Reshape targets to match output dimension
    y_train = y_train.reshape(-1, 1).astype(np.float32)
    y_val = y_val.reshape(-1, 1).astype(np.float32)
    
    # Create liquid neural network
    if network_type == 'lstm_gated':
        liquid_network = create_lstm_gated_liquid_network(
            input_dim=input_dim,
            units=128,
            lstm_units=32,
            output_dim=1,
            sparsity_level=0.5,
            stride_length=1,
            time_scale_factor=1.0,
            threshold=0.5,
            adaptive_threshold=True
        )
    else:  # standard
        liquid_network = create_liquid_network_with_motor_neuron(
            input_dim=input_dim,
            units=128,
            output_dim=1,
            sparsity_level=0.5,
            stride_length=1,
            time_scale_factor=1.0,
            threshold=0.5,
            adaptive_threshold=True,
            mixed_memory=True
        )
    
    # Set up callbacks
    callbacks = [
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        
        # Learning rate scheduling
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train liquid network
    history = liquid_network.fit(
        X_train_seq,
        y_train,
        validation_data=(X_val_seq, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info(f"Liquid network training completed with final loss: {history.history['loss'][-1]:.4f}")
    
    return liquid_network, history


def process_test_data(liquid_network, X_test, threshold=0.5):
    """
    Process test data through the liquid network.
    
    Args:
        liquid_network: Trained liquid network
        X_test: Test features
        threshold: Threshold for triggering
        
    Returns:
        Motor outputs and trigger signals
    """
    logger.info(f"Processing {len(X_test)} test samples")
    
    # Reshape for sequence input
    X_test_seq = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    # Process through liquid network
    outputs = liquid_network.predict(X_test_seq)
    
    # Extract motor neuron outputs and trigger signals
    if isinstance(outputs, list):
        motor_outputs = outputs[0]
        trigger_signals = outputs[1][0]  # First element is trigger
    else:
        motor_outputs = outputs
        trigger_signals = (motor_outputs > threshold).astype(float)
    
    logger.info(f"Motor neuron output range: {motor_outputs.min():.4f} to {motor_outputs.max():.4f}")
    logger.info(f"Trigger rate: {trigger_signals.mean():.4f}")
    
    return motor_outputs, trigger_signals


def plot_results(training_errors, history, motor_outputs, trigger_signals):
    """
    Plot training and test results.
    
    Args:
        training_errors: RBM training errors
        history: Liquid network training history
        motor_outputs: Motor neuron outputs
        trigger_signals: Trigger signals
    """
    # Create directory for plots
    os.makedirs('./plots', exist_ok=True)
    
    # Plot RBM training errors
    plt.figure(figsize=(10, 6))
    plt.plot(training_errors)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Error')
    plt.title('RBM Training Error')
    plt.savefig('./plots/rbm_training_error.png')
    plt.close()
    
    # Plot liquid network training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train')
    plt.plot(history.history['val_mae'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./plots/liquid_network_training.png')
    plt.close()
    
    # Plot motor neuron outputs and triggers
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(motor_outputs[:100], label='Motor Neuron Output')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Sample')
    plt.ylabel('Output Value')
    plt.title('Motor Neuron Output')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(trigger_signals[:100], 'g', label='Trigger Signal')
    plt.axhline(y=trigger_signals.mean(), color='r', linestyle='--', 
               label=f'Trigger Rate: {trigger_signals.mean():.2f}')
    plt.xlabel('Sample')
    plt.ylabel('Trigger (0/1)')
    plt.title('Exploration Trigger Signals')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./plots/motor_neuron_output.png')
    plt.close()
    
    logger.info("Results plotted and saved to ./plots directory")


def main():
    """Main function for testing the pipeline with sample data."""
    # Generate sample data
    X_train, X_val, X_test, y_train, y_val, y_test = generate_sample_data(
        n_samples=1000,
        n_features=20,
        n_classes=2
    )
    
    # Train RBM
    rbm, training_errors = train_rbm(
        X_train,
        n_hidden=64,
        epochs=10,
        use_gpu=True
    )
    
    # Extract RBM features
    train_rbm_features = extract_rbm_features(rbm, X_train)
    val_rbm_features = extract_rbm_features(rbm, X_val)
    test_rbm_features = extract_rbm_features(rbm, X_test)
    
    # Train liquid network
    liquid_network, history = train_liquid_network(
        train_rbm_features,
        y_train,
        val_rbm_features,
        y_val,
        input_dim=train_rbm_features.shape[1],
        network_type='standard',
        epochs=50
    )
    
    # Process test data
    motor_outputs, trigger_signals = process_test_data(
        liquid_network,
        test_rbm_features
    )
    
    # Plot results
    plot_results(
        training_errors,
        history,
        motor_outputs,
        trigger_signals
    )
    
    # Save models
    os.makedirs('./models', exist_ok=True)
    rbm.save('./models/rbm.npy')
    liquid_network.save('./models/liquid_network')
    logger.info("Models saved to ./models directory")
    
    logger.info("Pipeline test completed successfully")


if __name__ == "__main__":
    main()