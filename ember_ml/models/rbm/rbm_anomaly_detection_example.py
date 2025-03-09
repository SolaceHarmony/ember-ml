"""
RBM-based Anomaly Detection Example

This script demonstrates how to use the RBM-based anomaly detector
with the generic feature extraction library to detect anomalies in data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

# Import our modules from emberharmony
from ember_ml import (
    GenericCSVLoader,
    GenericTypeDetector,
    GenericFeatureEngineer,
    TemporalStrideProcessor,
    RBMBasedAnomalyDetector,
    RBMVisualizer
)


def generate_telemetry_data(n_samples=1000, n_features=10, anomaly_fraction=0.05):
    """
    Generate synthetic telemetry data with anomalies.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        anomaly_fraction: Fraction of anomalous samples
        
    Returns:
        DataFrame with telemetry data and anomaly labels
    """
    # Generate normal data
    normal_data = np.random.normal(0, 1, (n_samples, n_features))
    
    # Add some correlations between features
    for i in range(1, n_features):
        normal_data[:, i] = normal_data[:, i] * 0.5 + normal_data[:, 0] * 0.5
    
    # Add some temporal patterns
    for i in range(n_samples):
        normal_data[i, :] += np.sin(i / 50) * 0.5
    
    # Generate anomalies
    n_anomalies = int(n_samples * anomaly_fraction)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    
    # Create different types of anomalies
    for idx in anomaly_indices:
        anomaly_type = np.random.randint(0, 3)
        
        if anomaly_type == 0:
            # Spike anomaly
            feature_idx = np.random.randint(0, n_features)
            normal_data[idx, feature_idx] += np.random.uniform(3, 5)
        elif anomaly_type == 1:
            # Correlation anomaly
            normal_data[idx, :] = np.random.normal(0, 1, n_features)
        else:
            # Collective anomaly
            normal_data[idx, :] += np.random.uniform(2, 3, n_features)
    
    # Create DataFrame
    columns = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(normal_data, columns=columns)
    
    # Add timestamp column
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='5min')
    df['timestamp'] = timestamps
    
    # Add anomaly label
    df['anomaly'] = 0
    df.loc[anomaly_indices, 'anomaly'] = 1
    
    return df


def save_to_csv(df, filename='telemetry_data.csv'):
    """Save DataFrame to CSV file."""
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    return filename


def main():
    """Main function to demonstrate RBM-based anomaly detection."""
    print("RBM-based Anomaly Detection Example")
    print("===================================")
    
    # Create output directories if they don't exist
    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('outputs/animations', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic telemetry data
    print("\nGenerating synthetic telemetry data...")
    telemetry_df = generate_telemetry_data(n_samples=1000, n_features=10, anomaly_fraction=0.05)
    print(f"Generated {len(telemetry_df)} samples with {telemetry_df['anomaly'].sum()} anomalies")
    
    # Save data to CSV
    csv_file = save_to_csv(telemetry_df)
    
    # Load data using GenericCSVLoader
    print("\nLoading data using GenericCSVLoader...")
    loader = GenericCSVLoader()
    df = loader.load_csv(csv_file)
    
    # Detect column types
    print("\nDetecting column types...")
    detector = GenericTypeDetector()
    column_types = detector.detect_column_types(df)
    
    # Print column types
    for type_name, cols in column_types.items():
        print(f"{type_name.capitalize()} columns: {cols}")
    
    # Engineer features
    print("\nEngineering features...")
    engineer = GenericFeatureEngineer()
    df_engineered = engineer.engineer_features(df, column_types)
    
    # Get numeric features for anomaly detection
    numeric_features = column_types.get('numeric', [])
    numeric_features = [col for col in numeric_features if col != 'anomaly']  # Exclude anomaly label
    
    if not numeric_features:
        print("No numeric features available for anomaly detection")
        return
    
    # Extract features
    features_df = df_engineered[numeric_features]
    
    # Split data into normal and anomalous
    normal_indices = df_engineered['anomaly'] == 0
    anomaly_indices = df_engineered['anomaly'] == 1
    
    normal_features = features_df[normal_indices].values
    anomaly_features = features_df[anomaly_indices].values
    
    # Split normal data into training and validation sets
    n_normal = len(normal_features)
    n_train = int(n_normal * 0.8)
    
    train_features = normal_features[:n_train]
    val_features = normal_features[n_train:]
    
    # Initialize RBM-based anomaly detector
    print("\nInitializing RBM-based anomaly detector...")
    detector = RBMBasedAnomalyDetector(
        n_hidden=5,
        learning_rate=0.01,
        momentum=0.5,
        weight_decay=0.0001,
        batch_size=10,
        anomaly_threshold_percentile=95.0,
        anomaly_score_method='reconstruction',
        track_states=True
    )
    
    # Train anomaly detector
    print("\nTraining anomaly detector...")
    start_time = time.time()
    detector.fit(
        X=train_features,
        validation_data=val_features,
        epochs=30,
        k=1,
        early_stopping_patience=5,
        verbose=True
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Print detector summary
    print("\nAnomaly Detector Summary:")
    print(detector.summary())
    
    # Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"outputs/models/rbm_anomaly_detector_{timestamp}"
    detector.save(model_path)
    
    # Detect anomalies
    print("\nDetecting anomalies...")
    
    # Combine validation and anomaly data for testing
    test_features = np.vstack([val_features, anomaly_features])
    test_labels = np.hstack([
        np.zeros(len(val_features)),
        np.ones(len(anomaly_features))
    ])
    
    # Predict anomalies
    predicted_anomalies = detector.predict(test_features)
    anomaly_scores = detector.anomaly_score(test_features)
    
    # Compute metrics
    true_positives = np.sum((predicted_anomalies == 1) & (test_labels == 1))
    false_positives = np.sum((predicted_anomalies == 1) & (test_labels == 0))
    true_negatives = np.sum((predicted_anomalies == 0) & (test_labels == 0))
    false_negatives = np.sum((predicted_anomalies == 0) & (test_labels == 1))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    
    # Initialize visualizer
    visualizer = RBMVisualizer()
    
    # Plot training curve
    print("\nPlotting training curve...")
    visualizer.plot_training_curve(detector.rbm, show=True)
    
    # Plot weight matrix
    print("\nPlotting weight matrix...")
    visualizer.plot_weight_matrix(detector.rbm, show=True)
    
    # Plot reconstructions
    print("\nPlotting reconstructions...")
    visualizer.plot_reconstructions(detector.rbm, test_features[:5], show=True)
    
    # Plot anomaly scores
    print("\nPlotting anomaly scores...")
    plt.figure(figsize=(10, 6))
    plt.hist(anomaly_scores[test_labels == 0], bins=30, alpha=0.7, label='Normal')
    plt.hist(anomaly_scores[test_labels == 1], bins=30, alpha=0.7, label='Anomaly')
    plt.axvline(detector.anomaly_threshold, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    plt.title('Anomaly Score Distribution')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plot_path = f"outputs/plots/anomaly_scores_{timestamp}.png"
    plt.savefig(plot_path)
    plt.show()
    
    # Animate weight evolution
    print("\nAnimating weight evolution...")
    visualizer.animate_weight_evolution(detector.rbm, show=True)
    
    # Animate dreaming
    print("\nAnimating dreaming process...")
    visualizer.animate_dreaming(detector.rbm, n_steps=50, show=True)
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()