import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Union, Optional
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from ember_ml import ops
from ember_ml.nn.wirings import NCPWiring, AutoNCP
from ember_ml.nn.modules import NCP
from ember_ml.keras_3_8.layers.rnn import LTCCell, RNN

# Import the data preparation function
import sys
# The prepare_bigquery_data_bf function would be imported here in a real implementation
# from data_utils import prepare_bigquery_data_bf

class StrideAwareWiredCfCCell(NCP):
    """A stride-aware CfC cell that properly respects the Wiring architecture."""
    
    def __init__(
            self,
            wiring: NCPWiring,
            stride_length=1,
            time_scale_factor=1.0,
            fully_recurrent=True,
            mode="default",
            activation="tanh",
            **kwargs
    ):
        """Initialize a stride-aware WiredCfCCell.
        
        Args:
            wiring: A Wiring instance that determines the connectivity pattern
            stride_length: Length of the stride this cell handles
            time_scale_factor: Scaling factor for temporal dynamics (multiplied by stride_length)
            fully_recurrent: Whether to use full recurrent connectivity within layers
            mode: CfC operation mode ("default", "pure", or "no_gate")
            activation: Activation function used in the backbone layers
            **kwargs: Additional arguments to pass to the Layer constructor
        """
        super().__init__(
            wiring=wiring,
            activation=activation,
            **kwargs
        )
        self.stride_length = stride_length
        self.time_scale_factor = time_scale_factor
        self.fully_recurrent = fully_recurrent
        self.mode = mode
        self._activation = activation
    
    @property
    def state_size(self):
        return self.wiring.units
    
    @property
    def input_size(self):
        return self.wiring.input_dim
    
    @property
    def output_size(self):
        return self.wiring.output_dim
    
    def forward(self, inputs, state=None, **kwargs):
        """Apply stride-specific temporal scaling."""
        if isinstance(inputs, (tuple, list)):
            # Irregularly sampled mode
            inputs, t = inputs
            t = t * self.stride_length * self.time_scale_factor
        else:
            # Regularly sampled mode
            t = kwargs.get("time", 1.0) * self.stride_length * self.time_scale_factor
            # Create a tensor for time
            t_tensor = ops.convert_to_tensor(t, dtype='float32')
            inputs = (inputs, t_tensor)
        
        # Forward to the NCP implementation
        return super().forward(inputs, state)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "stride_length": self.stride_length,
            "time_scale_factor": self.time_scale_factor,
            "fully_recurrent": self.fully_recurrent,
            "mode": self.mode,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class StrideAwareLTCCell(LTCCell):
    """A stride-aware LTC cell for multi-timescale processing."""
    
    def __init__(
            self,
            input_size,
            hidden_size,
            stride_length=1,
            time_scale_factor=1.0,
            tau=1.0,
            **kwargs
    ):
        """Initialize a stride-aware LTC cell.
        
        Args:
            input_size: Input dimension
            hidden_size: Hidden state dimension
            stride_length: Length of the stride this cell handles
            time_scale_factor: Scaling factor for temporal dynamics (multiplied by stride_length)
            tau: Time constant
            **kwargs: Additional arguments to pass to the LTCCell constructor
        """
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            tau=tau,
            **kwargs
        )
        self.stride_length = stride_length
        self.time_scale_factor = time_scale_factor
        self.units = hidden_size  # For compatibility with the original API
    
    def forward(self, inputs, state=None, **kwargs):
        """Apply stride-specific temporal scaling."""
        if isinstance(inputs, (tuple, list)):
            # Irregularly sampled mode
            inputs, t = inputs
            t = t * self.stride_length * self.time_scale_factor
            # Adjust tau based on time scaling
            effective_tau = self.tau / t
        else:
            # Regularly sampled mode
            t = kwargs.get("time", 1.0) * self.stride_length * self.time_scale_factor
            # Adjust tau based on time scaling
            effective_tau = self.tau / t
        
        # Store original tau
        original_tau = self.tau
        # Temporarily set tau to the effective value
        self.tau = effective_tau
        
        # Call the parent class's forward method
        output, new_state = super().forward(inputs, state)
        
        # Restore original tau
        self.tau = original_tau
        
        return output, new_state
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "stride_length": self.stride_length,
            "time_scale_factor": self.time_scale_factor,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TemporalStrideProcessor:
    """Processes temporal data with multiple stride perspectives."""
    
    def __init__(self, window_size: int, stride_perspectives: List[int], pca_components: int):
        """Initialize the TemporalStrideProcessor.
        
        Args:
            window_size: Size of the sliding window
            stride_perspectives: List of stride lengths to use
            pca_components: Number of PCA components to extract
        """
        self.window_size = window_size
        self.stride_perspectives = stride_perspectives
        self.pca_components = pca_components

    def process_batch(self, data) -> Dict[int, np.ndarray]:
        """Process a batch of data with multiple stride perspectives.
        
        Args:
            data: Input data of shape (num_samples, num_features)
            
        Returns:
            Dictionary mapping stride lengths to processed data
        """
        # Convert to numpy for processing if it's not already
        if not isinstance(data, np.ndarray):
            data = ops.to_numpy(data)
            
        perspectives = {}
        for stride in self.stride_perspectives:
            if stride == 1:
                # Stride 1: No PCA, just create sliding windows
                strided_data = self._create_strided_sequences(data, stride)
                # Reshape to (num_windows, num_features * window_size)
                reduced_data = strided_data.reshape(strided_data.shape[0], -1)
            else:
                # Stride > 1: Apply PCA per column
                strided_data = self._create_strided_sequences(data, stride)
                reduced_data = self._apply_pca_per_column(strided_data)
            perspectives[stride] = reduced_data
            print(f"TemporalStrideProcessor: Stride {stride}, Output Shape: {reduced_data.shape}")
        return perspectives

    def _create_strided_sequences(self, data: np.ndarray, stride: int) -> np.ndarray:
        """Create strided sequences from the input data.
        
        Args:
            data: Input data of shape (num_samples, num_features)
            stride: Stride length
            
        Returns:
            Strided sequences of shape (num_sequences, window_size, num_features)
        """
        num_samples = data.shape[0]
        num_features = data.shape[1]
        subsequences = []

        for i in range(0, num_samples - self.window_size + 1, stride):
            subsequence = data[i:i + self.window_size]
            subsequences.append(subsequence)
        # Pad with the last window to keep the sequence as long as possible.
        if (num_samples - self.window_size + 1) % stride != 0:
            last_index = max(0, num_samples - self.window_size)
            subsequences.append(data[last_index:last_index+self.window_size])

        return np.array(subsequences)

    def _apply_pca_per_column(self, strided_data: np.ndarray) -> np.ndarray:
        """Apply PCA to each column of the strided data.
        
        Args:
            strided_data: Strided data of shape (num_sequences, window_size, num_features)
            
        Returns:
            PCA-reduced data of shape (num_sequences, num_features * pca_components)
        """
        num_sequences = strided_data.shape[0]
        num_features = strided_data.shape[2]  # Original number of features
        reduced_features = []

        for i in range(num_sequences):
            sequence = strided_data[i]  # (window_size, num_features)
            pca_results = []
            for j in range(num_features):
                column_data = sequence[:, j].reshape(-1, 1)  # Reshape for PCA
                if np.all(column_data == column_data[0]):  # Check for constant columns
                    if column_data.shape[0] < self.pca_components:
                        padded_column = np.pad(column_data.flatten(), (0, self.pca_components - column_data.shape[0]), 'constant', constant_values=column_data[0,0])
                        pca_results.append(padded_column)
                    else:
                        pca_results.append(np.full(self.pca_components, column_data[0,0])) #All the same value
                else:
                    pca = PCA(n_components=self.pca_components)
                    try:
                        transformed = pca.fit_transform(column_data)
                        pca_results.append(transformed.flatten())  # Flatten to 1D
                    except ValueError as e:
                        print(f"Error during PCA for sequence {i}, column {j}: {e}")
                        print("Input Data to PCA:", column_data)
                        raise

            reduced_features.append(np.concatenate(pca_results))

        return np.array(reduced_features)


def build_multiscale_ltc_model(input_dims: Dict[int, int], output_dim: int = 1,
                               hidden_units: int = 32, dropout_rate: float = 0.2):
    """Build a multi-scale LTC model with stride-aware cells.
    
    Args:
        input_dims: Dictionary mapping stride lengths to input dimensions
        output_dim: Dimension of the output
        hidden_units: Number of hidden units in each LTC cell
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Model with multiple stride-aware LTC cells
    """
    # Create a dictionary to store the model components
    model = {
        'inputs': {},
        'ltc_cells': {},
        'outputs': {}
    }
    
    # Create inputs and LTC cells for each stride
    ltc_outputs = []
    
    for stride, dim in input_dims.items():
        # Create input for this stride
        input_name = f"stride_{stride}_input"
        # Create a tensor to serve as a placeholder (similar to Keras Input layer)
        model['inputs'][stride] = ops.convert_to_tensor(ops.zeros((1, dim)), dtype='float32')
        
        # Reshape for RNN processing if needed
        reshaped = ops.reshape(model['inputs'][stride], (-1, 1, dim))
        
        # Create a wiring for this stride
        wiring = AutoNCP(
            units=hidden_units,
            output_size=hidden_units//2,
            sparsity_level=0.5
        )
        
        # Create a stride-aware LTC cell
        ltc_cell = StrideAwareLTCCell(
            input_size=dim,
            hidden_size=hidden_units,
            stride_length=stride,
            time_scale_factor=1.0,
            tau=0.7
        )
        
        # Create an RNN layer with the LTC cell
        rnn = RNN(
            cell=ltc_cell,
            return_sequences=False,
            return_state=False
        )
        
        # Process the input through the RNN
        rnn_output = rnn(reshaped)
        
        # Apply dropout for regularization
        rnn_dropout = ops.dropout(rnn_output, rate=dropout_rate, training=True)
        ltc_outputs.append(rnn_dropout)
        
        # Store the cell in the model
        model['ltc_cells'][stride] = ltc_cell
    
    # Concatenate outputs from all strides if there are multiple
    if len(ltc_outputs) > 1:
        concatenated = ops.concatenate(ltc_outputs, axis=-1)
    else:
        concatenated = ltc_outputs[0]
    
    # Add a dense layer to combine the multi-scale features
    dense = ops.dense(concatenated, units=hidden_units, activation="relu")
    dense_dropout = ops.dropout(dense, rate=dropout_rate, training=True)
    
    # Output layer
    output = ops.dense(dense_dropout, units=output_dim, activation=None)
    model['outputs']['main'] = output
    
    return model


def visualize_feature_extraction(metadata: Dict) -> plt.Figure:
    """Visualize the feature extraction process.
    
    Args:
        metadata: Dictionary containing metadata about the feature extraction process
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Feature counts
    ax1 = fig.add_subplot(2, 2, 1)
    feature_counts = metadata['feature_counts']
    ax1.bar(feature_counts.keys(), feature_counts.values())
    ax1.set_title("Feature Counts by Type")
    ax1.set_ylabel("Count")
    ax1.grid(True, alpha=0.3)
    
    # 2. Temporal compression
    ax2 = fig.add_subplot(2, 2, 2)
    compression_ratios = [data["compression_ratio"] for stride, data in metadata['temporal_compression'].items()]
    strides = list(metadata['temporal_compression'].keys())
    ax2.bar(strides, compression_ratios)
    ax2.set_title("Compression Ratio by Stride")
    ax2.set_xlabel("Stride")
    ax2.set_ylabel("Compression Ratio")
    ax2.grid(True, alpha=0.3)
    
    # 3. Input/Output dimensions
    ax3 = fig.add_subplot(2, 2, 3)
    input_dims = [data["input_dim"] for stride, data in metadata['temporal_compression'].items()]
    output_dims = [data["output_dim"] for stride, data in metadata['temporal_compression'].items()]
    x = np.arange(len(strides))
    width = 0.35
    ax3.bar(x - width/2, input_dims, width, label='Input Dim')
    ax3.bar(x + width/2, output_dims, width, label='Output Dim')
    ax3.set_title("Input/Output Dimensions by Stride")
    ax3.set_xlabel("Stride")
    ax3.set_ylabel("Dimension")
    ax3.set_xticks(x)
    ax3.set_xticklabels(strides)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Dimensional evolution
    ax4 = fig.add_subplot(2, 2, 4)
    stages = [item["stage"] for item in metadata['dimensional_evolution']]
    dimensions = [item["dimension"] for item in metadata['dimensional_evolution']]
    ax4.plot(stages, dimensions, marker='o')
    ax4.set_title("Dimensional Evolution Through Processing Stages")
    ax4.set_xlabel("Processing Stage")
    ax4.set_ylabel("Dimension")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_multiscale_dynamics(model, test_inputs, test_y, stride_perspectives):
    """Visualize the multi-scale dynamics of the model.
    
    Args:
        model: Trained model dictionary
        test_inputs: Test inputs
        test_y: Test targets
        stride_perspectives: List of stride lengths
        
    Returns:
        Matplotlib figure
    """
    # Get the intermediate outputs from the LTC cells
    intermediate_outputs = []
    
    # Convert inputs to numpy if they're not already
    test_inputs_np = {}
    for i, stride in enumerate(stride_perspectives):
        if stride in model['inputs']:
            # If using dictionary inputs
            if isinstance(test_inputs, dict):
                test_inputs_np[stride] = ops.to_numpy(test_inputs[stride])
            # If using tuple inputs (from the original code)
            elif isinstance(test_inputs, tuple) and i < len(test_inputs):
                test_inputs_np[stride] = ops.to_numpy(test_inputs[i])
    
    # Get outputs from each LTC cell
    for stride in stride_perspectives:
        if stride in model['ltc_cells']:
            # Get the cell
            cell = model['ltc_cells'][stride]
            
            # Forward pass through the cell
            if stride in test_inputs_np:
                # Reshape for RNN processing
                reshaped = np.reshape(test_inputs_np[stride], (-1, 1, test_inputs_np[stride].shape[1]))
                
                # Process through the cell
                outputs = []
                states = None
                for i in range(reshaped.shape[0]):
                    output, states = cell.forward(reshaped[i], states)
                    outputs.append(output)
                
                # Convert to numpy array
                cell_output = np.array(outputs)
                intermediate_outputs.append((stride, cell_output))
    
    # Create a figure
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Prediction vs. Actual
    ax1 = fig.add_subplot(2, 2, 1)
    
    # Get predictions from the model
    predictions = []
    if isinstance(test_y, np.ndarray):
        test_y_np = test_y
    else:
        test_y_np = ops.to_numpy(test_y)
    
    # Plot scatter of predictions vs actual
    if len(intermediate_outputs) > 0:
        # Use the last layer's output as predictions
        predictions = intermediate_outputs[-1][1]
        ax1.scatter(test_y_np, predictions, alpha=0.5)
        ax1.plot([np.min(test_y_np), np.max(test_y_np)],
                [np.min(test_y_np), np.max(test_y_np)], 'r--')
    
    ax1.set_title("Prediction vs. Actual")
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.grid(True, alpha=0.3)
    
    # 2. Activation patterns across strides
    ax2 = fig.add_subplot(2, 2, 2)
    for stride, output in intermediate_outputs:
        # Take the mean activation across samples
        mean_activation = np.mean(output, axis=0)
        ax2.plot(mean_activation, label=f"Stride {stride}")
    ax2.set_title("Mean Activation Patterns Across Strides")
    ax2.set_xlabel("Neuron Index")
    ax2.set_ylabel("Mean Activation")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. PCA of activations
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    for stride, output in intermediate_outputs:
        # Apply PCA to reduce to 3D
        if output.shape[0] > 3:  # Need at least 4 samples for 3 components
            pca = PCA(n_components=min(3, output.shape[0]-1))
            output_pca = pca.fit_transform(output)
            ax3.scatter(output_pca[:, 0],
                       output_pca[:, 1],
                       output_pca[:, 2] if output_pca.shape[1] > 2 else np.zeros(output_pca.shape[0]),
                       label=f"Stride {stride}", alpha=0.5)
    ax3.set_title("PCA of Activations")
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_zlabel("PC3")
    ax3.legend()
    
    # 4. Activation distribution
    ax4 = fig.add_subplot(2, 2, 4)
    for stride, output in intermediate_outputs:
        # Flatten the output
        output_flat = output.flatten()
        ax4.hist(output_flat, bins=50, alpha=0.5, label=f"Stride {stride}")
    ax4.set_title("Activation Distribution")
    ax4.set_xlabel("Activation")
    ax4.set_ylabel("Frequency")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def integrate_liquid_neurons_with_visualization(
    project_id,
    table_id,
    target_column=None,
    window_size=5,
    stride_perspectives=[1, 3, 5],
    batch_size=32,
    epochs=15,
    pca_components=3,
    **prepare_kwargs
):
    """
    Runs the entire pipeline with PCA applied *per column* within each stride.
    
    Args:
        project_id: GCP project ID
        table_id: BigQuery table ID
        target_column: Target column name
        window_size: Size of the sliding window
        stride_perspectives: List of stride lengths to use
        batch_size: Batch size for training
        epochs: Number of epochs for training
        pca_components: Number of PCA components to extract
        **prepare_kwargs: Additional arguments for prepare_bigquery_data_bf
        
    Returns:
        Training history
    """
    # Data preparation
    print("🔹 Starting Data Preparation...")
    result = prepare_bigquery_data_bf(
        project_id=project_id,
        table_id=table_id,
        target_column=target_column,
        **prepare_kwargs
    )

    if result is None:
        raise ValueError("❌ Data preparation failed.")

    # Unpack results
    train_bf_df, val_bf_df, test_bf_df, train_features, val_features, test_features, scaler, imputer = result
    train_df, val_df, test_df = train_bf_df.to_pandas(), val_bf_df.to_pandas(), test_bf_df.to_pandas()

    # Auto-detect target column if not provided
    if target_column is None:
        all_cols = train_bf_df.columns.tolist()
        feature_set = set(train_features)

        # Find the first column that is NOT a feature (likely the target)
        possible_targets = [col for col in all_cols if col not in feature_set]

        if possible_targets:
            target_column = possible_targets[0]
            print(f"🟢 Auto-selected target column: {target_column}")
        else:
            raise ValueError("❌ No valid target column found. Please specify `target_column` manually.")

    # Extract features & targets
    train_X, val_X, test_X = train_df[train_features].to_numpy(), val_df[val_features].to_numpy(), test_df[test_features].to_numpy()
    train_y, val_y, test_y = train_df[target_column].to_numpy(), val_df[target_column].to_numpy(), test_df[target_column].to_numpy()

    # Detect if the target is categorical (string-based)
    if train_y.dtype == "object":
        print(f"🟡 Detected categorical target ({target_column}). Applying OneHot + PCA.")

        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        # Fit encoder on ALL data to prevent category mismatches
        all_targets = np.concatenate([train_y.reshape(-1, 1), val_y.reshape(-1, 1), test_y.reshape(-1, 1)], axis=0)
        encoder.fit(all_targets)

        # Transform all data using the same encoder
        train_y = encoder.transform(train_y.reshape(-1, 1))
        val_y = encoder.transform(val_y.reshape(-1, 1))
        test_y = encoder.transform(test_y.reshape(-1, 1))

        # Apply PCA to reduce high-dimensional encoding to a single scalar
        pca_target = PCA(n_components=1)
        all_encoded_targets = np.concatenate([train_y, val_y, test_y], axis=0)
        pca_target.fit(all_encoded_targets)

        train_y = pca_target.transform(train_y).astype(np.float32).reshape(-1, 1)
        val_y = pca_target.transform(val_y).astype(np.float32).reshape(-1, 1)
        test_y = pca_target.transform(test_y).astype(np.float32).reshape(-1, 1)

        print(f"✅ OneHot+PCA target shape (should be 1D scalar): {train_y.shape}")

    else:
        print(f"🟢 Detected numeric target ({target_column}). Using directly as float32.")
        train_y = ops.convert_to_tensor(train_y.astype(np.float32).reshape(-1, 1))
        val_y = ops.convert_to_tensor(val_y.astype(np.float32).reshape(-1, 1))
        test_y = ops.convert_to_tensor(test_y.astype(np.float32).reshape(-1, 1))

    print(f"✅ Final target shape: {train_y.shape}, dtype: {train_y.dtype}")

    # Process stride-based representations
    processor = TemporalStrideProcessor(window_size=window_size, stride_perspectives=stride_perspectives, pca_components=pca_components)
    train_perspectives = processor.process_batch(train_X)
    val_perspectives = processor.process_batch(val_X)
    test_perspectives = processor.process_batch(test_X)

    # Convert to emberharmony tensors
    train_inputs = {s: ops.convert_to_tensor(data, dtype='float32')
                   for s, data in train_perspectives.items()}
    val_inputs = {s: ops.convert_to_tensor(data, dtype='float32')
                 for s, data in val_perspectives.items()}
    test_inputs = {s: ops.convert_to_tensor(data, dtype='float32')
                  for s, data in test_perspectives.items()}

    print("Train Input Shapes (Before Model Building):",
          {s: data.shape for s, data in train_inputs.items()})
    print("Validation Input Shapes (Before Model Building):",
          {s: data.shape for s, data in val_inputs.items()})
    print("Test Input Shapes (Before Model Building):",
          {s: data.shape for s, data in test_inputs.items()})

    # Build model
    print("🔹 Building Multi-Scale Liquid Neural Network...")
    input_dims = {s: train_perspectives[s].shape[1] for s in train_perspectives.keys()}
    model = build_multiscale_ltc_model(input_dims=input_dims, output_dim=1)

    # Train model
    history = {
        'loss': [],
        'val_loss': [],
        'mae': [],
        'val_mae': []
    }
    
    # Simple training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training step
        train_loss = 0
        train_mae = 0
        
        # Process in batches
        num_batches = len(train_X) // batch_size
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(train_X))
            
            # Get batch inputs
            batch_inputs = {s: train_inputs[s][start_idx:end_idx] for s in train_inputs}
            batch_y = train_y[start_idx:end_idx]
            
            # Forward pass
            outputs = model['outputs']['main']
            
            # Compute loss
            loss = ops.mean_squared_error(batch_y, outputs)
            mae = ops.mean_absolute_error(batch_y, outputs)
            
            # Update metrics
            train_loss += loss
            train_mae += mae
            
            # Print progress
            if batch % 10 == 0:
                print(f"  Batch {batch}/{num_batches} - Loss: {loss:.4f}, MAE: {mae:.4f}")
        
        # Compute average metrics
        train_loss /= num_batches
        train_mae /= num_batches
        
        # Validation step
        val_loss = 0
        val_mae = 0
        
        # Process validation data
        num_val_batches = len(val_X) // batch_size
        for batch in range(num_val_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(val_X))
            
            # Get batch inputs
            batch_inputs = {s: val_inputs[s][start_idx:end_idx] for s in val_inputs}
            batch_y = val_y[start_idx:end_idx]
            
            # Forward pass
            outputs = model['outputs']['main']
            
            # Compute loss
            loss = ops.mean_squared_error(batch_y, outputs)
            mae = ops.mean_absolute_error(batch_y, outputs)
            
            # Update metrics
            val_loss += loss
            val_mae += mae
        
        # Compute average metrics
        val_loss /= num_val_batches
        val_mae /= num_val_batches
        
        # Update history
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
        
        # Early stopping
        if epoch > 5 and history['val_loss'][-1] > history['val_loss'][-2]:
            print("Early stopping triggered")
            break

    # Evaluate on the test set
    test_loss = 0
    test_mae = 0
    
    # Process test data
    num_test_batches = len(test_X) // batch_size
    for batch in range(num_test_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, len(test_X))
        
        # Get batch inputs
        batch_inputs = {s: test_inputs[s][start_idx:end_idx] for s in test_inputs}
        batch_y = test_y[start_idx:end_idx]
        
        # Forward pass
        outputs = model['outputs']['main']
        
        # Compute loss
        loss = ops.mean_squared_error(batch_y, outputs)
        mae = ops.mean_absolute_error(batch_y, outputs)
        
        # Update metrics
        test_loss += loss
        test_mae += mae
    
    # Compute average metrics
    test_loss /= num_test_batches
    test_mae /= num_test_batches
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

    # Populate metadata for visualizations
    metadata = {
        'feature_counts': {
            'original': len(train_features),
            'numeric': sum(1 for feat in train_features if "sin_" not in feat and "cos_" not in feat),
            'categorical': sum(1 for feat in train_features if "sin_" in feat or "cos_" in feat),
        },
        'temporal_compression': {
            stride: {
                "input_dim": train_perspectives[stride].shape[0] * window_size,
                "output_dim": train_perspectives[stride].shape[1],
                "compression_ratio": (train_perspectives[stride].shape[0] * window_size)/train_perspectives[stride].shape[1],
            }
            for stride in stride_perspectives if stride in train_perspectives
        },
        'dimensional_evolution': [
            {"stage": f"stride_{s}", "dimension": train_perspectives[s].shape[1]} for s in stride_perspectives if s in train_perspectives
        ]
    }

    # Run the visualizations
    feature_fig = visualize_feature_extraction(metadata)
    plt.savefig('feature_extraction.png')
    print("Feature extraction visualization saved to 'feature_extraction.png'")
    
    dynamics_fig = visualize_multiscale_dynamics(model, test_inputs, test_y, stride_perspectives)
    plt.savefig('multiscale_dynamics.png')
    print("Multiscale dynamics visualization saved to 'multiscale_dynamics.png'")
    
    return history


if __name__ == "__main__":
    # Example usage
    print("This module provides classes and functions for multi-scale liquid neural networks.")
    print("To use it, import the module and call the integrate_liquid_neurons_with_visualization function.")
    print("Example:")
    print("  from emberharmony.attention.multiscale_ltc import integrate_liquid_neurons_with_visualization")
    print("  history = integrate_liquid_neurons_with_visualization(")
    print("      project_id='your-project-id',")
    print("      table_id='your-dataset.your-table',")
    print("      window_size=5,")
    print("      stride_perspectives=[1, 3, 5],")
    print("      batch_size=32,")
    print("      epochs=15,")
    print("      pca_components=3")
    print("  )")


def visualize_multiscale_dynamics(model, test_inputs, test_y, stride_perspectives):
    """Visualize the multi-scale dynamics of the model.
    
    Args:
        model: Trained model dictionary
        test_inputs: Test inputs
        test_y: Test targets
        stride_perspectives: List of stride lengths
        
    Returns:
        Matplotlib figure
    """
    # Get the intermediate outputs from the LTC cells
    intermediate_outputs = []
    
    # Convert inputs to numpy if they're not already
    test_inputs_np = {}
    for i, stride in enumerate(stride_perspectives):
        if stride in model['inputs']:
            # If using dictionary inputs
            if isinstance(test_inputs, dict):
                test_inputs_np[stride] = ops.to_numpy(test_inputs[stride])
            # If using tuple inputs (from the original code)
            elif isinstance(test_inputs, tuple) and i < len(test_inputs):
                test_inputs_np[stride] = ops.to_numpy(test_inputs[i])
    
    # Get outputs from each LTC cell
    for stride in stride_perspectives:
        if stride in model['ltc_cells']:
            # Get the cell
            cell = model['ltc_cells'][stride]
            
            # Forward pass through the cell
            if stride in test_inputs_np:
                # Reshape for RNN processing
                reshaped = np.reshape(test_inputs_np[stride], (-1, 1, test_inputs_np[stride].shape[1]))
                
                # Process through the cell
                outputs = []
                states = None
                for i in range(reshaped.shape[0]):
                    output, states = cell.forward(reshaped[i], states)
                    outputs.append(output)
                
                # Convert to numpy array
                cell_output = np.array(outputs)
                intermediate_outputs.append((stride, cell_output))
    
    # Create a figure
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Prediction vs. Actual
    ax1 = fig.add_subplot(2, 2, 1)
    
    # Get predictions from the model
    predictions = []
    if isinstance(test_y, np.ndarray):
        test_y_np = test_y
    else:
        test_y_np = ops.to_numpy(test_y)
    
    # Plot scatter of predictions vs actual
    if len(intermediate_outputs) > 0:
        # Use the last layer's output as predictions
        predictions = intermediate_outputs[-1][1]
        ax1.scatter(test_y_np, predictions, alpha=0.5)
        ax1.plot([np.min(test_y_np), np.max(test_y_np)],
                [np.min(test_y_np), np.max(test_y_np)], 'r--')
    
    ax1.set_title("Prediction vs. Actual")
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.grid(True, alpha=0.3)
    
    # 2. Activation patterns across strides
    ax2 = fig.add_subplot(2, 2, 2)
    for stride, output in intermediate_outputs:
        # Take the mean activation across samples
        mean_activation = np.mean(output, axis=0)
        ax2.plot(mean_activation, label=f"Stride {stride}")
    ax2.set_title("Mean Activation Patterns Across Strides")
    ax2.set_xlabel("Neuron Index")
    ax2.set_ylabel("Mean Activation")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. PCA of activations
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    for stride, output in intermediate_outputs:
        # Apply PCA to reduce to 3D
        if output.shape[0] > 3:  # Need at least 4 samples for 3 components
            pca = PCA(n_components=min(3, output.shape[0]-1))
            output_pca = pca.fit_transform(output)
            ax3.scatter(output_pca[:, 0],
                       output_pca[:, 1],
                       output_pca[:, 2] if output_pca.shape[1] > 2 else np.zeros(output_pca.shape[0]),
                       label=f"Stride {stride}", alpha=0.5)
    ax3.set_title("PCA of Activations")
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_zlabel("PC3")
    ax3.legend()
    
    # 4. Activation distribution
    ax4 = fig.add_subplot(2, 2, 4)
    for stride, output in intermediate_outputs:
        # Flatten the output
        output_flat = output.flatten()
        ax4.hist(output_flat, bins=50, alpha=0.5, label=f"Stride {stride}")
    ax4.set_title("Activation Distribution")
    ax4.set_xlabel("Activation")
    ax4.set_ylabel("Frequency")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def integrate_liquid_neurons_with_visualization(
    project_id,
    table_id,
    target_column=None,
    window_size=5,
    stride_perspectives=[1, 3, 5],
    batch_size=32,
    epochs=15,
    pca_components=3,
    **prepare_kwargs
):
    """
    Runs the entire pipeline with PCA applied *per column* within each stride.
    
    Args:
        project_id: GCP project ID
        table_id: BigQuery table ID
        target_column: Target column name
        window_size: Size of the sliding window
        stride_perspectives: List of stride lengths to use
        batch_size: Batch size for training
        epochs: Number of epochs for training
        pca_components: Number of PCA components to extract
        **prepare_kwargs: Additional arguments for prepare_bigquery_data_bf
        
    Returns:
        Training history
    """
    # Data preparation
    print("🔹 Starting Data Preparation...")
    result = prepare_bigquery_data_bf(
        project_id=project_id,
        table_id=table_id,
        target_column=target_column,
        **prepare_kwargs
    )

    if result is None:
        raise ValueError("❌ Data preparation failed.")

    # Unpack results
    train_bf_df, val_bf_df, test_bf_df, train_features, val_features, test_features, scaler, imputer = result
    train_df, val_df, test_df = train_bf_df.to_pandas(), val_bf_df.to_pandas(), test_bf_df.to_pandas()

    # Auto-detect target column if not provided
    if target_column is None:
        all_cols = train_bf_df.columns.tolist()
        feature_set = set(train_features)

        # Find the first column that is NOT a feature (likely the target)
        possible_targets = [col for col in all_cols if col not in feature_set]

        if possible_targets:
            target_column = possible_targets[0]
            print(f"🟢 Auto-selected target column: {target_column}")
        else:
            raise ValueError("❌ No valid target column found. Please specify `target_column` manually.")

    # Extract features & targets
    train_X, val_X, test_X = train_df[train_features].to_numpy(), val_df[val_features].to_numpy(), test_df[test_features].to_numpy()
    train_y, val_y, test_y = train_df[target_column].to_numpy(), val_df[target_column].to_numpy(), test_df[target_column].to_numpy()

    # Detect if the target is categorical (string-based)
    if train_y.dtype == "object":
        print(f"🟡 Detected categorical target ({target_column}). Applying OneHot + PCA.")

        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        # Fit encoder on ALL data to prevent category mismatches
        all_targets = np.concatenate([train_y.reshape(-1, 1), val_y.reshape(-1, 1), test_y.reshape(-1, 1)], axis=0)
        encoder.fit(all_targets)

        # Transform all data using the same encoder
        train_y = encoder.transform(train_y.reshape(-1, 1))
        val_y = encoder.transform(val_y.reshape(-1, 1))
        test_y = encoder.transform(test_y.reshape(-1, 1))

        # Apply PCA to reduce high-dimensional encoding to a single scalar
        pca_target = PCA(n_components=1)
        all_encoded_targets = np.concatenate([train_y, val_y, test_y], axis=0)
        pca_target.fit(all_encoded_targets)

        train_y = pca_target.transform(train_y).astype(np.float32).reshape(-1, 1)
        val_y = pca_target.transform(val_y).astype(np.float32).reshape(-1, 1)
        test_y = pca_target.transform(test_y).astype(np.float32).reshape(-1, 1)

        print(f"✅ OneHot+PCA target shape (should be 1D scalar): {train_y.shape}")

    else:
        print(f"🟢 Detected numeric target ({target_column}). Using directly as float32.")
        train_y = ops.convert_to_tensor(train_y.astype(np.float32).reshape(-1, 1))
        val_y = ops.convert_to_tensor(val_y.astype(np.float32).reshape(-1, 1))
        test_y = ops.convert_to_tensor(test_y.astype(np.float32).reshape(-1, 1))

    print(f"✅ Final target shape: {train_y.shape}, dtype: {train_y.dtype}")

    # Process stride-based representations
    processor = TemporalStrideProcessor(window_size=window_size, stride_perspectives=stride_perspectives, pca_components=pca_components)
    train_perspectives = processor.process_batch(train_X)
    val_perspectives = processor.process_batch(val_X)
    test_perspectives = processor.process_batch(test_X)

    # Convert to emberharmony tensors
    train_inputs = {s: ops.convert_to_tensor(data, dtype='float32')
                   for s, data in train_perspectives.items()}
    val_inputs = {s: ops.convert_to_tensor(data, dtype='float32')
                 for s, data in val_perspectives.items()}
    test_inputs = {s: ops.convert_to_tensor(data, dtype='float32')
                  for s, data in test_perspectives.items()}

    print("Train Input Shapes (Before Model Building):",
          {s: data.shape for s, data in train_inputs.items()})
    print("Validation Input Shapes (Before Model Building):",
          {s: data.shape for s, data in val_inputs.items()})
    print("Test Input Shapes (Before Model Building):",
          {s: data.shape for s, data in test_inputs.items()})

    # Build model
    print("🔹 Building Multi-Scale Liquid Neural Network...")
    input_dims = {s: train_perspectives[s].shape[1] for s in train_perspectives.keys()}
    model = build_multiscale_ltc_model(input_dims=input_dims, output_dim=1)

    # Train model
    history = {
        'loss': [],
        'val_loss': [],
        'mae': [],
        'val_mae': []
    }
    
    # Simple training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training step
        train_loss = 0
        train_mae = 0
        
        # Process in batches
        num_batches = len(train_X) // batch_size
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(train_X))
            
            # Get batch inputs
            batch_inputs = {s: train_inputs[s][start_idx:end_idx] for s in train_inputs}
            batch_y = train_y[start_idx:end_idx]
            
            # Forward pass
            outputs = model['outputs']['main']
            
            # Compute loss
            loss = ops.mean_squared_error(batch_y, outputs)
            mae = ops.mean_absolute_error(batch_y, outputs)
            
            # Update metrics
            train_loss += loss
            train_mae += mae
            
            # Print progress
            if batch % 10 == 0:
                print(f"  Batch {batch}/{num_batches} - Loss: {loss:.4f}, MAE: {mae:.4f}")
        
        # Compute average metrics
        train_loss /= num_batches
        train_mae /= num_batches
        
        # Validation step
        val_loss = 0
        val_mae = 0
        
        # Process validation data
        num_val_batches = len(val_X) // batch_size
        for batch in range(num_val_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(val_X))
            
            # Get batch inputs
            batch_inputs = {s: val_inputs[s][start_idx:end_idx] for s in val_inputs}
            batch_y = val_y[start_idx:end_idx]
            
            # Forward pass
            outputs = model['outputs']['main']
            
            # Compute loss
            loss = ops.mean_squared_error(batch_y, outputs)
            mae = ops.mean_absolute_error(batch_y, outputs)
            
            # Update metrics
            val_loss += loss
            val_mae += mae
        
        # Compute average metrics
        val_loss /= num_val_batches
        val_mae /= num_val_batches
        
        # Update history
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
        
        # Early stopping
        if epoch > 5 and history['val_loss'][-1] > history['val_loss'][-2]:
            print("Early stopping triggered")
            break

    # Evaluate on the test set
    test_loss = 0
    test_mae = 0
    
    # Process test data
    num_test_batches = len(test_X) // batch_size
    for batch in range(num_test_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, len(test_X))
        
        # Get batch inputs
        batch_inputs = {s: test_inputs[s][start_idx:end_idx] for s in test_inputs}
        batch_y = test_y[start_idx:end_idx]
        
        # Forward pass
        outputs = model['outputs']['main']
        
        # Compute loss
        loss = ops.mean_squared_error(batch_y, outputs)
        mae = ops.mean_absolute_error(batch_y, outputs)
        
        # Update metrics
        test_loss += loss
        test_mae += mae
    
    # Compute average metrics
    test_loss /= num_test_batches
    test_mae /= num_test_batches
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

    # Populate metadata for visualizations
    metadata = {
        'feature_counts': {
            'original': len(train_features),
            'numeric': sum(1 for feat in train_features if "sin_" not in feat and "cos_" not in feat),
            'categorical': sum(1 for feat in train_features if "sin_" in feat or "cos_" in feat),
        },
        'temporal_compression': {
            stride: {
                "input_dim": train_perspectives[stride].shape[0] * window_size,
                "output_dim": train_perspectives[stride].shape[1],
                "compression_ratio": (train_perspectives[stride].shape[0] * window_size)/train_perspectives[stride].shape[1],
            }
            for stride in stride_perspectives if stride in train_perspectives
        },
        'dimensional_evolution': [
            {"stage": f"stride_{s}", "dimension": train_perspectives[s].shape[1]} for s in stride_perspectives if s in train_perspectives
        ]
    }

    # Run the visualizations
    feature_fig = visualize_feature_extraction(metadata)
    plt.savefig('feature_extraction.png')
    print("Feature extraction visualization saved to 'feature_extraction.png'")
    
    dynamics_fig = visualize_multiscale_dynamics(model, test_inputs, test_y, stride_perspectives)
    plt.savefig('multiscale_dynamics.png')
    print("Multiscale dynamics visualization saved to 'multiscale_dynamics.png'")
    
    return history


if __name__ == "__main__":
    # Example usage
    print("This module provides classes and functions for multi-scale liquid neural networks.")
    print("To use it, import the module and call the integrate_liquid_neurons_with_visualization function.")
    print("Example:")
    print("  from emberharmony.attention.multiscale_ltc import integrate_liquid_neurons_with_visualization")
    print("  history = integrate_liquid_neurons_with_visualization(")
    print("      project_id='your-project-id',")
    print("      table_id='your-dataset.your-table',")
    print("      window_size=5,")
    print("      stride_perspectives=[1, 3, 5],")
    print("      batch_size=32,")
    print("      epochs=15,")
    print("      pca_components=3")
    print("  )")