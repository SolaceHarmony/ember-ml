"""
RBM-based Anomaly Detector

This module provides an anomaly detection system based on Restricted Boltzmann Machines.
It integrates with the generic feature extraction library to provide end-to-end
anomaly detection capabilities.
"""

import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

# Import our modules
from ember_ml.models.rbm import RestrictedBoltzmannMachine


class RBMBasedAnomalyDetector:
    """
    Anomaly detection system based on Restricted Boltzmann Machines.
    
    This class uses an RBM to learn the normal patterns in data and
    then detect anomalies as patterns that deviate significantly from
    the learned normal patterns.
    
    The detector can work with both raw data and features extracted
    by the generic feature extraction library.
    """
    
    def __init__(
        self,
        n_hidden: int = 10,
        learning_rate: float = 0.01,
        momentum: float = 0.5,
        weight_decay: float = 0.0001,
        batch_size: int = 10,
        anomaly_threshold_percentile: float = 95.0,
        anomaly_score_method: str = 'reconstruction',
        track_states: bool = True
    ):
        """
        Initialize the RBM-based anomaly detector.
        
        Args:
            n_hidden: Number of hidden units in the RBM
            learning_rate: Learning rate for RBM training
            momentum: Momentum for RBM training
            weight_decay: Weight decay for RBM training
            batch_size: Batch size for RBM training
            anomaly_threshold_percentile: Percentile for anomaly threshold
            anomaly_score_method: Method for computing anomaly scores
                ('reconstruction' or 'free_energy')
            track_states: Whether to track RBM states for visualization
        """
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.anomaly_threshold_percentile = anomaly_threshold_percentile
        self.anomaly_score_method = anomaly_score_method
        self.track_states = track_states
        
        # RBM model (initialized during fit)
        self.rbm = None
        
        # Preprocessing parameters
        self.feature_means = None
        self.feature_stds = None
        self.feature_mins = None
        self.feature_maxs = None
        self.scaling_method = 'standard'  # 'standard' or 'minmax'
        
        # Anomaly detection parameters
        self.anomaly_threshold = None
        self.anomaly_scores_mean = None
        self.anomaly_scores_std = None
        
        # Training metadata
        self.n_features = None
        self.training_time = 0
        self.is_fitted = False
    
    def preprocess(
        self,
        X: np.ndarray,
        fit: bool = False,
        scaling_method: str = 'standard'
    ) -> np.ndarray:
        """
        Preprocess data for RBM training or anomaly detection.
        
        Args:
            X: Input data [n_samples, n_features]
            fit: Whether to fit preprocessing parameters
            scaling_method: Scaling method ('standard' or 'minmax')
            
        Returns:
            Preprocessed data
        """
        if fit:
            self.scaling_method = scaling_method
            self.n_features = X.shape[1]
            
            if scaling_method == 'standard':
                # Compute mean and std for standardization
                self.feature_means = np.mean(X, axis=0)
                self.feature_stds = np.std(X, axis=0)
                self.feature_stds[self.feature_stds == 0] = 1.0  # Avoid division by zero
            else:
                # Compute min and max for min-max scaling
                self.feature_mins = np.min(X, axis=0)
                self.feature_maxs = np.max(X, axis=0)
                # Avoid division by zero
                self.feature_maxs[self.feature_maxs == self.feature_mins] += 1e-8
        
        # Apply scaling
        if self.scaling_method == 'standard':
            X_scaled = (X - self.feature_means) / self.feature_stds
        else:
            X_scaled = (X - self.feature_mins) / (self.feature_maxs - self.feature_mins)
        
        return X_scaled
    
    def fit(
        self,
        X: np.ndarray,
        validation_data: Optional[np.ndarray] = None,
        epochs: int = 50,
        k: int = 1,
        early_stopping_patience: int = 5,
        scaling_method: str = 'standard',
        verbose: bool = True
    ) -> 'RBMBasedAnomalyDetector':
        """
        Fit the anomaly detector to normal data.
        
        Args:
            X: Normal data [n_samples, n_features]
            validation_data: Optional validation data
            epochs: Number of training epochs
            k: Number of Gibbs sampling steps
            early_stopping_patience: Patience for early stopping
            scaling_method: Scaling method ('standard' or 'minmax')
            verbose: Whether to print progress
            
        Returns:
            Self
        """
        start_time = time.time()
        
        # Preprocess data
        X_scaled = self.preprocess(X, fit=True, scaling_method=scaling_method)
        
        # Preprocess validation data if provided
        if validation_data is not None:
            validation_data_scaled = self.preprocess(validation_data, fit=False)
        else:
            validation_data_scaled = None
        
        # Initialize RBM
        self.rbm = RestrictedBoltzmannMachine(
            n_visible=self.n_features,
            n_hidden=self.n_hidden,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            use_binary_states=False,
            track_states=self.track_states
        )
        
        # Train RBM
        self.rbm.train(
            data=X_scaled,
            epochs=epochs,
            k=k,
            validation_data=validation_data_scaled,
            early_stopping_patience=early_stopping_patience,
            verbose=verbose
        )
        
        # Compute anomaly scores on training data
        anomaly_scores = self.rbm.anomaly_score(X_scaled, method=self.anomaly_score_method)
        
        # Compute anomaly threshold
        self.anomaly_threshold = np.percentile(
            anomaly_scores,
            self.anomaly_threshold_percentile
        )
        
        # Compute statistics of anomaly scores
        self.anomaly_scores_mean = np.mean(anomaly_scores)
        self.anomaly_scores_std = np.std(anomaly_scores)
        
        self.training_time = time.time() - start_time
        self.is_fitted = True
        
        if verbose:
            print(f"Anomaly detector trained in {self.training_time:.2f} seconds")
            print(f"Anomaly threshold: {self.anomaly_threshold:.4f}")
            print(f"Anomaly scores mean: {self.anomaly_scores_mean:.4f}, std: {self.anomaly_scores_std:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict whether samples are anomalies.
        
        Args:
            X: Input data [n_samples, n_features]
            
        Returns:
            Boolean array indicating anomalies [n_samples]
        """
        if not self.is_fitted:
            raise ValueError("Anomaly detector is not fitted yet. Call fit() first.")
        
        # Preprocess data
        X_scaled = self.preprocess(X, fit=False)
        
        # Compute anomaly scores
        anomaly_scores = self.rbm.anomaly_score(X_scaled, method=self.anomaly_score_method)
        
        # Determine anomalies
        anomalies = anomaly_scores > self.anomaly_threshold
        
        return anomalies
    
    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for input data.
        
        Args:
            X: Input data [n_samples, n_features]
            
        Returns:
            Anomaly scores [n_samples]
        """
        if not self.is_fitted:
            raise ValueError("Anomaly detector is not fitted yet. Call fit() first.")
        
        # Preprocess data
        X_scaled = self.preprocess(X, fit=False)
        
        # Compute anomaly scores
        return self.rbm.anomaly_score(X_scaled, method=self.anomaly_score_method)
    
    def anomaly_probability(self, X: np.ndarray) -> np.ndarray:
        """
        Compute probability of being an anomaly.
        
        This uses a sigmoid function to map anomaly scores to [0, 1].
        
        Args:
            X: Input data [n_samples, n_features]
            
        Returns:
            Anomaly probabilities [n_samples]
        """
        if not self.is_fitted:
            raise ValueError("Anomaly detector is not fitted yet. Call fit() first.")
        
        # Compute anomaly scores
        scores = self.anomaly_score(X)
        
        # Normalize scores
        normalized_scores = (scores - self.anomaly_scores_mean) / self.anomaly_scores_std
        
        # Map to [0, 1] using sigmoid
        return 1.0 / (1.0 + np.exp(-normalized_scores))
    
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruct input data using the RBM.
        
        Args:
            X: Input data [n_samples, n_features]
            
        Returns:
            Reconstructed data [n_samples, n_features]
        """
        if not self.is_fitted:
            raise ValueError("Anomaly detector is not fitted yet. Call fit() first.")
        
        # Preprocess data
        X_scaled = self.preprocess(X, fit=False)
        
        # Reconstruct data
        X_reconstructed = self.rbm.reconstruct(X_scaled)
        
        # Inverse scaling
        if self.scaling_method == 'standard':
            X_reconstructed = X_reconstructed * self.feature_stds + self.feature_means
        else:
            X_reconstructed = X_reconstructed * (self.feature_maxs - self.feature_mins) + self.feature_mins
        
        return X_reconstructed
    
    def save(self, filepath: str) -> None:
        """
        Save the anomaly detector to a file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Anomaly detector is not fitted yet. Call fit() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare model data
        model_data = {
            'n_hidden': self.n_hidden,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'anomaly_threshold_percentile': self.anomaly_threshold_percentile,
            'anomaly_score_method': self.anomaly_score_method,
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds,
            'feature_mins': self.feature_mins,
            'feature_maxs': self.feature_maxs,
            'scaling_method': self.scaling_method,
            'anomaly_threshold': self.anomaly_threshold,
            'anomaly_scores_mean': self.anomaly_scores_mean,
            'anomaly_scores_std': self.anomaly_scores_std,
            'n_features': self.n_features,
            'training_time': self.training_time,
            'is_fitted': self.is_fitted,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save RBM separately
        rbm_filepath = filepath + '.rbm'
        self.rbm.save(rbm_filepath)
        
        # Save model data
        np.save(filepath, model_data, allow_pickle=True)
        print(f"Anomaly detector saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'RBMBasedAnomalyDetector':
        """
        Load an anomaly detector from a file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded anomaly detector
        """
        # Load model data
        model_data = np.load(filepath, allow_pickle=True).item()
        
        # Create detector
        detector = cls(
            n_hidden=model_data['n_hidden'],
            learning_rate=model_data['learning_rate'],
            momentum=model_data['momentum'],
            weight_decay=model_data['weight_decay'],
            batch_size=model_data['batch_size'],
            anomaly_threshold_percentile=model_data['anomaly_threshold_percentile'],
            anomaly_score_method=model_data['anomaly_score_method']
        )
        
        # Set model parameters
        detector.feature_means = model_data['feature_means']
        detector.feature_stds = model_data['feature_stds']
        detector.feature_mins = model_data['feature_mins']
        detector.feature_maxs = model_data['feature_maxs']
        detector.scaling_method = model_data['scaling_method']
        detector.anomaly_threshold = model_data['anomaly_threshold']
        detector.anomaly_scores_mean = model_data['anomaly_scores_mean']
        detector.anomaly_scores_std = model_data['anomaly_scores_std']
        detector.n_features = model_data['n_features']
        detector.training_time = model_data['training_time']
        detector.is_fitted = model_data['is_fitted']
        
        # Load RBM
        rbm_filepath = filepath + '.rbm'
        detector.rbm = RestrictedBoltzmannMachine.load(rbm_filepath)
        
        return detector
    
    def summary(self) -> str:
        """
        Get a summary of the anomaly detector.
        
        Returns:
            Summary string
        """
        if not self.is_fitted:
            return "RBM-based Anomaly Detector (not fitted)"
        
        summary = [
            "RBM-based Anomaly Detector Summary",
            "==================================",
            f"Features: {self.n_features}",
            f"Hidden units: {self.n_hidden}",
            f"Scaling method: {self.scaling_method}",
            f"Anomaly score method: {self.anomaly_score_method}",
            f"Anomaly threshold: {self.anomaly_threshold:.4f} ({self.anomaly_threshold_percentile}th percentile)",
            f"Anomaly scores mean: {self.anomaly_scores_mean:.4f}",
            f"Anomaly scores std: {self.anomaly_scores_std:.4f}",
            f"Training time: {self.training_time:.2f} seconds",
            "",
            "RBM Summary:",
            "-----------"
        ]
        
        # Add RBM summary
        rbm_summary = self.rbm.summary().split('\n')
        summary.extend(rbm_summary[1:])  # Skip the first line (title)
        
        return "\n".join(summary)


# Integration with generic feature extraction
def detect_anomalies_from_features(
    features_df: pd.DataFrame,
    n_hidden: int = 10,
    anomaly_threshold_percentile: float = 95.0,
    training_fraction: float = 0.8,
    epochs: int = 50,
    verbose: bool = True
) -> Tuple[RBMBasedAnomalyDetector, np.ndarray, np.ndarray]:
    """
    Detect anomalies from features extracted by the generic feature extraction library.
    
    Args:
        features_df: DataFrame with extracted features
        n_hidden: Number of hidden units in the RBM
        anomaly_threshold_percentile: Percentile for anomaly threshold
        training_fraction: Fraction of data to use for training
        epochs: Number of training epochs
        verbose: Whether to print progress
        
    Returns:
        Tuple of (anomaly detector, anomaly flags, anomaly scores)
    """
    # Convert features to numpy array
    features = features_df.values
    
    # Split into training and testing sets
    n_samples = len(features)
    n_train = int(n_samples * training_fraction)
    
    # Shuffle data
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_features = features[train_indices]
    test_features = features[test_indices]
    
    # Create and fit anomaly detector
    detector = RBMBasedAnomalyDetector(
        n_hidden=n_hidden,
        anomaly_threshold_percentile=anomaly_threshold_percentile
    )
    
    detector.fit(
        X=train_features,
        epochs=epochs,
        verbose=verbose
    )
    
    # Detect anomalies
    anomaly_flags = detector.predict(test_features)
    anomaly_scores = detector.anomaly_score(test_features)
    
    if verbose:
        n_anomalies = np.sum(anomaly_flags)
        print(f"Detected {n_anomalies} anomalies out of {len(test_features)} samples "
              f"({n_anomalies/len(test_features)*100:.2f}%)")
    
    return detector, anomaly_flags, anomaly_scores