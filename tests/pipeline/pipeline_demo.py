"""
Integrated Pipeline Demo

This script demonstrates the complete data processing pipeline:
1. Feature extraction from BigQuery using terabyte-scale feature extractor
2. Feature learning with Restricted Boltzmann Machines
3. Processing through CfC-based liquid neural network with LSTM gating
4. Motor neuron output for triggering deeper exploration
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import time
import argparse
from typing import Dict, List, Optional, Tuple, Union, Any, Generator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('pipeline_demo')

# Import our components (using the purified backend-agnostic implementation)
from ember_ml.features.terabyte_feature_extractor import TerabyteFeatureExtractor, TerabyteTemporalStrideProcessor
from ember_ml.models.optimized_rbm import OptimizedRBM
from ember_ml.core.stride_aware_cfc import (
    create_liquid_network_with_motor_neuron,
    create_lstm_gated_liquid_network,
    create_multi_stride_liquid_network
)

# Check if ncps is available
try:
    from ncps import wirings
    NCPS_AVAILABLE = True
except ImportError:
    NCPS_AVAILABLE = False
    logger.warning("ncps package not available. AutoNCP wiring will not be available.")


class IntegratedPipeline:
    """
    Integrated pipeline for processing terabyte-scale data through
    feature extraction, RBM, and liquid neural network components.
    """
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "US",
        chunk_size: int = 100000,
        max_memory_gb: float = 16.0,
        rbm_hidden_units: int = 64,
        cfc_units: int = 128,
        lstm_units: int = 32,
        stride_perspectives: List[int] = [1, 3, 5],
        sparsity_level: float = 0.5,
        threshold: float = 0.5,
        use_gpu: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the integrated pipeline.
        
        Args:
            project_id: GCP project ID (optional if using in BigQuery Studio)
            location: BigQuery location (default: "US")
            chunk_size: Number of rows to process per chunk
            max_memory_gb: Maximum memory usage in GB
            rbm_hidden_units: Number of hidden units in RBM
            cfc_units: Number of units in CfC circuit
            lstm_units: Number of units in LSTM gating
            stride_perspectives: List of stride lengths to use
            sparsity_level: Sparsity level for the connections
            threshold: Initial threshold for triggering exploration
            use_gpu: Whether to use GPU acceleration if available
            verbose: Whether to print progress information
        """
        self.project_id = project_id
        self.location = location
        self.chunk_size = chunk_size
        self.max_memory_gb = max_memory_gb
        self.rbm_hidden_units = rbm_hidden_units
        self.cfc_units = cfc_units
        self.lstm_units = lstm_units
        self.stride_perspectives = stride_perspectives
        self.sparsity_level = sparsity_level
        self.threshold = threshold
        self.use_gpu = use_gpu
        self.verbose = verbose
        
        # Initialize components
        self.feature_extractor = None
        self.temporal_processor = None
        self.rbm = None
        self.liquid_network = None
        
        # For tracking processing
        self.feature_dim = None
        self.rbm_feature_dim = None
        self.processing_time = {}
        
        logger.info(f"Initialized IntegratedPipeline with rbm_hidden_units={rbm_hidden_units}, "
                   f"cfc_units={cfc_units}, lstm_units={lstm_units}")
    
    def initialize_feature_extractor(self, credentials_path: Optional[str] = None):
        """
        Initialize the feature extractor component.
        
        Args:
            credentials_path: Optional path to service account credentials
        """
        start_time = time.time()
        
        # Create feature extractor
        self.feature_extractor = TerabyteFeatureExtractor(
            project_id=self.project_id,
            location=self.location,
            chunk_size=self.chunk_size,
            max_memory_gb=self.max_memory_gb,
            verbose=self.verbose
        )
        
        # Set up BigQuery connection
        self.feature_extractor.setup_bigquery_connection(credentials_path)
        
        # Create temporal processor
        self.temporal_processor = TerabyteTemporalStrideProcessor(
            window_size=10,
            stride_perspectives=self.stride_perspectives,
            pca_components=32,
            batch_size=10000,
            use_incremental_pca=True,
            verbose=self.verbose
        )
        
        self.processing_time['feature_extractor_init'] = time.time() - start_time
        logger.info(f"Feature extractor initialized in {self.processing_time['feature_extractor_init']:.2f}s")
    
    def initialize_rbm(self, input_dim: int):
        """
        Initialize the RBM component.
        
        Args:
            input_dim: Dimension of input features
        """
        start_time = time.time()
        
        # Create RBM
        self.rbm = OptimizedRBM(
            n_visible=input_dim,
            n_hidden=self.rbm_hidden_units,
            learning_rate=0.01,
            momentum=0.5,
            weight_decay=0.0001,
            batch_size=100,
            use_binary_states=False,
            use_gpu=self.use_gpu,
            verbose=self.verbose
        )
        
        self.feature_dim = input_dim
        self.rbm_feature_dim = self.rbm_hidden_units
        
        self.processing_time['rbm_init'] = time.time() - start_time
        logger.info(f"RBM initialized in {self.processing_time['rbm_init']:.2f}s")
    
    def initialize_liquid_network(self, input_dim: int, network_type: str = 'standard'):
        """
        Initialize the liquid neural network component.
        
        Args:
            input_dim: Dimension of input features
            network_type: Type of network ('standard', 'lstm_gated', or 'multi_stride')
        """
        start_time = time.time()
        
        # Check if ncps is available
        if not NCPS_AVAILABLE:
            raise ImportError("ncps package is required for liquid neural network")
        
        # Create liquid neural network based on type
        if network_type == 'lstm_gated':
            self.liquid_network = create_lstm_gated_liquid_network(
                input_dim=input_dim,
                units=self.cfc_units,
                lstm_units=self.lstm_units,
                output_dim=1,
                sparsity_level=self.sparsity_level,
                stride_length=self.stride_perspectives[0],
                time_scale_factor=1.0,
                threshold=self.threshold,
                adaptive_threshold=True
            )
        elif network_type == 'multi_stride':
            self.liquid_network = create_multi_stride_liquid_network(
                input_dim=input_dim,
                stride_perspectives=self.stride_perspectives,
                units_per_stride=self.cfc_units // len(self.stride_perspectives),
                output_dim=1,
                sparsity_level=self.sparsity_level,
                time_scale_factor=1.0,
                threshold=self.threshold,
                adaptive_threshold=True
            )
        else:  # standard
            self.liquid_network = create_liquid_network_with_motor_neuron(
                input_dim=input_dim,
                units=self.cfc_units,
                output_dim=1,
                sparsity_level=self.sparsity_level,
                stride_length=self.stride_perspectives[0],
                time_scale_factor=1.0,
                threshold=self.threshold,
                adaptive_threshold=True,
                mixed_memory=True
            )
        
        self.processing_time['liquid_network_init'] = time.time() - start_time
        logger.info(f"Liquid network ({network_type}) initialized in {self.processing_time['liquid_network_init']:.2f}s")
    
    def extract_features(
        self,
        table_id: str,
        target_column: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Tuple:
        """
        Extract features from BigQuery table.
        
        Args:
            table_id: BigQuery table ID (dataset.table)
            target_column: Target variable name
            limit: Optional row limit for testing
            
        Returns:
            Tuple of (train_features, val_features, test_features)
        """
        start_time = time.time()
        
        # Check if feature extractor is initialized
        if self.feature_extractor is None:
            self.initialize_feature_extractor()
        
        # Prepare data
        logger.info(f"Extracting features from {table_id}")
        result = self.feature_extractor.prepare_data(
            table_id=table_id,
            target_column=target_column,
            limit=limit
        )
        
        if result is None:
            raise ValueError("Feature extraction failed")
        
        # Unpack results
        train_df, val_df, test_df, train_features, val_features, test_features, scaler, imputer = result
        
        # Update feature dimension
        self.feature_dim = len(train_features)
        
        # Initialize RBM if not already initialized
        if self.rbm is None:
            self.initialize_rbm(self.feature_dim)
        
        self.processing_time['feature_extraction'] = time.time() - start_time
        logger.info(f"Feature extraction completed in {self.processing_time['feature_extraction']:.2f}s")
        logger.info(f"Extracted {self.feature_dim} features")
        
        return train_df[train_features], val_df[val_features], test_df[test_features]
    
    def apply_temporal_processing(self, features_df: pd.DataFrame) -> Dict[int, np.ndarray]:
        """
        Apply temporal processing to features.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Dictionary of stride perspectives with processed data
        """
        start_time = time.time()
        
        # Check if temporal processor is initialized
        if self.temporal_processor is None:
            self.temporal_processor = TerabyteTemporalStrideProcessor(
                window_size=10,
                stride_perspectives=self.stride_perspectives,
                pca_components=32,
                batch_size=10000,
                use_incremental_pca=True,
                verbose=self.verbose
            )
        
        # Define a generator to yield data in batches
        def data_generator(df, batch_size=10000):
            for i in range(0, len(df), batch_size):
                yield df.iloc[i:i+batch_size].values
        
        # Process data
        logger.info(f"Applying temporal processing with strides {self.stride_perspectives}")
        stride_perspectives = self.temporal_processor.process_large_dataset(
            data_generator(features_df, batch_size=10000)
        )
        
        self.processing_time['temporal_processing'] = time.time() - start_time
        logger.info(f"Temporal processing completed in {self.processing_time['temporal_processing']:.2f}s")
        
        # Log stride perspective shapes
        for stride, data in stride_perspectives.items():
            logger.info(f"Stride {stride}: shape {data.shape}")
        
        return stride_perspectives
    
    def train_rbm(self, features: Union[np.ndarray, pd.DataFrame], epochs: int = 10) -> OptimizedRBM:
        """
        Train RBM on features.
        
        Args:
            features: Feature array or DataFrame
            epochs: Number of training epochs
            
        Returns:
            Trained RBM
        """
        start_time = time.time()
        
        # Convert to numpy array if DataFrame
        if isinstance(features, pd.DataFrame):
            features = features.values
        
        # Check if RBM is initialized
        if self.rbm is None:
            self.initialize_rbm(features.shape[1])
        
        # Define a generator to yield data in batches
        def data_generator(data, batch_size=100):
            # Shuffle data
            indices = np.random.permutation(len(data))
            data = data[indices]
            
            for i in range(0, len(data), batch_size):
                yield data[i:i+batch_size]
        
        # Train RBM
        logger.info(f"Training RBM with {self.rbm_hidden_units} hidden units for {epochs} epochs")
        training_errors = self.rbm.train_in_chunks(
            data_generator(features, batch_size=100),
            epochs=epochs,
            k=1
        )
        
        self.processing_time['rbm_training'] = time.time() - start_time
        logger.info(f"RBM training completed in {self.processing_time['rbm_training']:.2f}s")
        logger.info(f"Final reconstruction error: {training_errors[-1]:.4f}")
        
        return self.rbm
    
    def extract_rbm_features(self, features: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Extract features from trained RBM.
        
        Args:
            features: Feature array or DataFrame
            
        Returns:
            RBM features
        """
        start_time = time.time()
        
        # Convert to numpy array if DataFrame
        if isinstance(features, pd.DataFrame):
            features = features.values
        
        # Check if RBM is trained
        if self.rbm is None:
            raise ValueError("RBM must be trained before extracting features")
        
        # Define a generator to yield data in batches
        def data_generator(data, batch_size=1000):
            for i in range(0, len(data), batch_size):
                yield data[i:i+batch_size]
        
        # Extract features
        logger.info(f"Extracting RBM features from {len(features)} samples")
        rbm_features = self.rbm.transform_in_chunks(
            data_generator(features, batch_size=1000)
        )
        
        self.processing_time['rbm_feature_extraction'] = time.time() - start_time
        logger.info(f"RBM feature extraction completed in {self.processing_time['rbm_feature_extraction']:.2f}s")
        logger.info(f"Extracted {rbm_features.shape[1]} RBM features")
        
        return rbm_features
    
    def train_liquid_network(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        epochs: int = 100,
        batch_size: int = 32,
        network_type: str = 'standard'
    ) -> tf.keras.Model:
        """
        Train liquid neural network on RBM features.
        
        Args:
            features: Feature array
            targets: Target array
            validation_data: Optional validation data
            epochs: Number of training epochs
            batch_size: Batch size for training
            network_type: Type of network ('standard', 'lstm_gated', or 'multi_stride')
            
        Returns:
            Trained liquid neural network
        """
        start_time = time.time()
        
        # Check if liquid network is initialized
        if self.liquid_network is None:
            self.initialize_liquid_network(features.shape[1], network_type)
        
        # Reshape features for sequence input if needed
        if len(features.shape) == 2:
            # Add sequence dimension
            features = features.reshape(features.shape[0], 1, features.shape[1])
        
        # Reshape validation data if provided
        if validation_data is not None:
            val_features, val_targets = validation_data
            if len(val_features.shape) == 2:
                val_features = val_features.reshape(val_features.shape[0], 1, val_features.shape[1])
            validation_data = (val_features, val_targets)
        
        # Set up callbacks
        callbacks = [
            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            
            # Learning rate scheduling
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            
            # TensorBoard logging
            tf.keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1
            )
        ]
        
        # Train liquid network
        logger.info(f"Training {network_type} liquid network for {epochs} epochs")
        history = self.liquid_network.fit(
            features,
            targets,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1 if self.verbose else 0
        )
        
        self.processing_time['liquid_network_training'] = time.time() - start_time
        logger.info(f"Liquid network training completed in {self.processing_time['liquid_network_training']:.2f}s")
        
        # Log training results
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1] if validation_data is not None else None
        
        logger.info(f"Final training loss: {final_loss:.4f}")
        if final_val_loss is not None:
            logger.info(f"Final validation loss: {final_val_loss:.4f}")
        
        return self.liquid_network
    
    def process_data(
        self,
        features: np.ndarray,
        return_triggers: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Process data through the complete pipeline.
        
        Args:
            features: Feature array
            return_triggers: Whether to return trigger signals
            
        Returns:
            Motor neuron outputs and trigger signals
        """
        start_time = time.time()
        
        # Check if all components are initialized
        if self.rbm is None:
            raise ValueError("RBM must be trained before processing data")
        if self.liquid_network is None:
            raise ValueError("Liquid network must be trained before processing data")
        
        # Extract RBM features
        rbm_features = self.rbm.transform(features)
        
        # Reshape for sequence input if needed
        if len(rbm_features.shape) == 2:
            rbm_features = rbm_features.reshape(rbm_features.shape[0], 1, rbm_features.shape[1])
        
        # Process through liquid network
        outputs = self.liquid_network.predict(rbm_features)
        
        self.processing_time['data_processing'] = time.time() - start_time
        logger.info(f"Data processing completed in {self.processing_time['data_processing']:.2f}s")
        
        # Return outputs based on return_triggers
        if return_triggers:
            if isinstance(outputs, list):
                motor_outputs = outputs[0]
                trigger_signals = outputs[1][0]  # First element is trigger
                return motor_outputs, trigger_signals
            else:
                # If model doesn't have separate trigger output
                motor_outputs = outputs
                trigger_signals = (motor_outputs > self.threshold).astype(float)
                return motor_outputs, trigger_signals
        else:
            if isinstance(outputs, list):
                return outputs[0]  # Just return motor outputs
            else:
                return outputs
    
    def save_model(self, directory: str):
        """
        Save all model components.
        
        Args:
            directory: Directory to save models
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save RBM
        if self.rbm is not None:
            rbm_path = os.path.join(directory, "rbm.npy")
            self.rbm.save(rbm_path)
            logger.info(f"RBM saved to {rbm_path}")
        
        # Save liquid network
        if self.liquid_network is not None:
            liquid_network_path = os.path.join(directory, "liquid_network")
            self.liquid_network.save(liquid_network_path)
            logger.info(f"Liquid network saved to {liquid_network_path}")
        
        # Save processing times
        processing_times_path = os.path.join(directory, "processing_times.csv")
        pd.DataFrame([self.processing_time]).to_csv(processing_times_path, index=False)
        logger.info(f"Processing times saved to {processing_times_path}")
    
    def load_model(self, directory: str, network_type: str = 'standard'):
        """
        Load all model components.
        
        Args:
            directory: Directory to load models from
            network_type: Type of liquid network
        """
        # Load RBM
        rbm_path = os.path.join(directory, "rbm.npy")
        if os.path.exists(rbm_path):
            self.rbm = OptimizedRBM.load(rbm_path, use_gpu=self.use_gpu)
            self.rbm_feature_dim = self.rbm.n_hidden
            logger.info(f"RBM loaded from {rbm_path}")
        
        # Load liquid network
        liquid_network_path = os.path.join(directory, "liquid_network")
        if os.path.exists(liquid_network_path):
            self.liquid_network = tf.keras.models.load_model(
                liquid_network_path,
                custom_objects={
                    'MotorNeuron': MotorNeuron,
                    'AdaptiveExplorationTrigger': AdaptiveExplorationTrigger
                }
            )
            logger.info(f"Liquid network loaded from {liquid_network_path}")
        
        # Load processing times
        processing_times_path = os.path.join(directory, "processing_times.csv")
        if os.path.exists(processing_times_path):
            self.processing_time = pd.read_csv(processing_times_path).iloc[0].to_dict()
            logger.info(f"Processing times loaded from {processing_times_path}")
    
    def summary(self) -> str:
        """
        Get a summary of the pipeline.
        
        Returns:
            Summary string
        """
        summary = [
            "Integrated Pipeline Summary",
            "==========================",
            f"Feature dimension: {self.feature_dim}",
            f"RBM hidden units: {self.rbm_hidden_units}",
            f"CfC units: {self.cfc_units}",
            f"LSTM units: {self.lstm_units}",
            f"Stride perspectives: {self.stride_perspectives}",
            f"Sparsity level: {self.sparsity_level}",
            f"Threshold: {self.threshold}",
            f"GPU acceleration: {self.use_gpu}",
            "",
            "Processing Times:",
        ]
        
        for key, value in self.processing_time.items():
            summary.append(f"  {key}: {value:.2f}s")
        
        if self.rbm is not None:
            summary.append("")
            summary.append("RBM Summary:")
            summary.append(self.rbm.summary())
        
        if self.liquid_network is not None:
            summary.append("")
            summary.append("Liquid Network Summary:")
            summary.append(str(self.liquid_network.summary()))
        
        return "\n".join(summary)


def main():
    """Main function for the pipeline demo."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Integrated Pipeline Demo")
    parser.add_argument("--project-id", type=str, help="GCP project ID")
    parser.add_argument("--table-id", type=str, help="BigQuery table ID (dataset.table)")
    parser.add_argument("--target-column", type=str, help="Target column name")
    parser.add_argument("--limit", type=int, default=10000, help="Row limit for testing")
    parser.add_argument("--rbm-hidden-units", type=int, default=64, help="Number of hidden units in RBM")
    parser.add_argument("--cfc-units", type=int, default=128, help="Number of units in CfC circuit")
    parser.add_argument("--lstm-units", type=int, default=32, help="Number of units in LSTM gating")
    parser.add_argument("--network-type", type=str, default="standard", 
                        choices=["standard", "lstm_gated", "multi_stride"],
                        help="Type of liquid network")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--save-dir", type=str, default="./models", help="Directory to save models")
    parser.add_argument("--load-models", action="store_true", help="Load models from save-dir")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = IntegratedPipeline(
        project_id=args.project_id,
        rbm_hidden_units=args.rbm_hidden_units,
        cfc_units=args.cfc_units,
        lstm_units=args.lstm_units,
        use_gpu=not args.no_gpu,
        verbose=args.verbose
    )
    
    # Load models if requested
    if args.load_models:
        pipeline.load_model(args.save_dir, args.network_type)
    
    # Extract features if table_id is provided
    if args.table_id:
        train_features, val_features, test_features = pipeline.extract_features(
            table_id=args.table_id,
            target_column=args.target_column,
            limit=args.limit
        )
        
        # Apply temporal processing
        train_temporal = pipeline.apply_temporal_processing(train_features)
        val_temporal = pipeline.apply_temporal_processing(val_features)
        
        # Train RBM
        pipeline.train_rbm(train_features, epochs=args.epochs)
        
        # Extract RBM features
        train_rbm_features = pipeline.extract_rbm_features(train_features)
        val_rbm_features = pipeline.extract_rbm_features(val_features)
        
        # Create dummy targets for demonstration
        # In a real application, you would use actual targets
        train_targets = np.random.rand(len(train_rbm_features), 1)
        val_targets = np.random.rand(len(val_rbm_features), 1)
        
        # Train liquid network
        pipeline.train_liquid_network(
            features=train_rbm_features,
            targets=train_targets,
            validation_data=(val_rbm_features, val_targets),
            epochs=args.epochs,
            batch_size=args.batch_size,
            network_type=args.network_type
        )
        
        # Process test data
        test_rbm_features = pipeline.extract_rbm_features(test_features)
        motor_outputs, trigger_signals = pipeline.process_data(test_rbm_features)
        
        # Print results
        logger.info(f"Processed {len(test_rbm_features)} test samples")
        logger.info(f"Motor neuron output range: {motor_outputs.min():.4f} to {motor_outputs.max():.4f}")
        logger.info(f"Trigger rate: {trigger_signals.mean():.4f}")
        
        # Save models
        pipeline.save_model(args.save_dir)
    
    # Print pipeline summary
    print(pipeline.summary())


if __name__ == "__main__":
    main()