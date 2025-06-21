import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler # Keep StandardScaler import as per user feedback (defer sklearn issues)

from ember_ml import ops
from ember_ml.nn.modules import AutoNCP, Dense, Dropout # Keep AutoNCP import
from ember_ml.nn.modules.rnn import CfC # Keep CfC import
# from ember_ml.nn import Module # Module is not directly used here, Sequential is
from ember_ml.nn import tensor # For tensor.EmberTensor, tensor.stack, etc.
# from ember_ml.nn.tensor import EmberTensor # Direct import for type hinting if preferred - EmberTensor class itself is not used here
from ember_ml.nn.tensor.types import TensorLike # For type hinting backend tensors
from ember_ml.nn.container import Sequential, BatchNormalization
from ember_ml.training import optimizer as ember_optimizer # Added
from ember_ml.training import loss as ember_loss # Added
from ember_ml.training.optimizer import GradientTape # Added

# --------------------------
# 📊 Telemetry Data Pipeline
# --------------------------
class TelemetryProcessor:
    def __init__(self, seq_len=64, stride=16, test_size=0.2):
        self.seq_len = seq_len
        self.stride = stride
        self.test_size = test_size
        # TODO: Ideally, replace sklearn.preprocessing.StandardScaler with an ember_ml equivalent.
        self.scaler = StandardScaler()
        
    def load_data(self, csv_path):
        """Load and preprocess telemetry data."""
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)
        
        numeric_cols = df.select_dtypes(include=np.number).columns.drop('timestamp', errors='ignore')
        data = df[numeric_cols].values # Convert to numpy array for scaler
        
        # tensor.item is used as ops.multiply/subtract might return scalar tensor if inputs are scalars
        split_idx = int(tensor.item(ops.multiply(float(len(data)), ops.subtract(1.0, self.test_size))))
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        train_norm = self.scaler.fit_transform(train_data)
        test_norm = self.scaler.transform(test_data)
        
        return train_norm, test_norm
        
    def create_sequences(self, data) -> TensorLike: # data is numpy array here, returns backend tensor
        """Convert raw data into overlapping sequences."""
        sequences = []
        # Ensure len(data) and self.seq_len are Python ints for range
        # standard ops are fine for Python numbers
        for i in range(0, (len(data) - self.seq_len) + 1, self.stride):
            sequences.append(data[i : i + self.seq_len])
        # tensor.convert_to_tensor now returns a backend tensor
        return tensor.convert_to_tensor(np.array(sequences), dtype=tensor.EmberDType.float32)


# --------------------------
# 🔥 Multi-Scale Liquid Neural Network
# --------------------------
class LiquidNeuralNetwork: # This class itself is not an ember_ml.nn.Module, it just builds one
    def __init__(self, input_shape, model_size=128):
        self.input_shape = input_shape # Should be tuple e.g. (seq_len, num_features)
        self.model_size = model_size
        self.model: Sequential = self._build_model() # Type hint for self.model
        
    def _build_model(self) -> Sequential: # Return type is Sequential
        # Fast timescale layer for immediate feature detection
        wiring_fast = AutoNCP(
            units=self.model_size,
            output_size=self.model_size // 4, # Standard integer division is fine for layer params
            sparsity_level=0.5
        )
        ltc_fast = CfC(wiring_fast, return_sequences=True, mixed_memory=True)
        
        # Medium timescale layer for pattern recognition
        wiring_med = AutoNCP(
            units=self.model_size // 2,
            output_size=self.model_size // 8,
            sparsity_level=0.4
        )
        ltc_med = CfC(wiring_med, return_sequences=True, mixed_memory=True)
        
        # Slow timescale layer for trend analysis
        wiring_slow = AutoNCP(
            units=self.model_size // 4,
            output_size=self.model_size // 16,
            sparsity_level=0.3
        )
        ltc_slow = CfC(wiring_slow, return_sequences=False, mixed_memory=True)
        
        model = Sequential([
            # Input shape is implicitly handled by the first layer that receives data.
            # No explicit Input layer like Keras needed if Sequential handles it.
            # Or, the first layer (ltc_fast) needs to be able to receive input_shape.
            # For CfC, input_shape might be passed to its build method or __init__.
            # This part depends on ember_ml.nn.Sequential and layer implementation details.
            # We assume CfC can take input_shape or it's inferred.
            ltc_fast,
            BatchNormalization(),
            ltc_med,
            BatchNormalization(),
            ltc_slow,
            BatchNormalization(),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid') # Sigmoid output for binary classification
        ])
        
        # Optimizer and model.compile are removed. Handled in training loop.
        return model

# --------------------------
# 📈 Training Callbacks (Removed)
# --------------------------
# class MultiScaleMonitor(keras.callbacks.Callback): ...
# MultiScaleMonitor Keras callback has been removed.
# Custom logging for layer outputs would need to be integrated into the manual training loop if desired.
# Other Keras callbacks (EarlyStopping, ReduceLROnPlateau, TensorBoard)
# also need to be replaced with equivalent logic within the custom training loop or via ember_ml native callbacks if available.

# --------------------------
# 🚀 Training & Deployment
# --------------------------

# Python generator for data batching
def data_generator(features: TensorLike, labels: TensorLike, batch_size: int, shuffle: bool = True):
    # features and labels are expected to be backend tensors
    dataset_size = tensor.shape(features)[0] # tensor.shape works on backend tensors

    # Determine device from features tensor for creating new tensors
    # This assumes features is not None and has a device attribute or ops.get_device_of_tensor works
    # If features can be from any backend, device handling needs to be robust.
    # For simplicity, assume features is an EmberTensor or backend tensor from which device can be inferred.
    device_for_indices = ops.get_device_of_tensor(features) if hasattr(features, 'device') else ops.get_device()


    indices_tensor = tensor.arange(dataset_size, device=device_for_indices) # Returns backend tensor
    if shuffle:
        shuffled_indices_tensor = tensor.random_permutation(indices_tensor) # Returns backend tensor
    else:
        shuffled_indices_tensor = indices_tensor

    for start_idx in range(0, dataset_size, batch_size):
        end_idx = min(start_idx + batch_size, dataset_size)

        num_elements_in_batch = end_idx - start_idx
        if num_elements_in_batch == 0:
            continue

        # batch_indices_tensor is a backend tensor
        batch_indices_tensor = tensor.slice_tensor(shuffled_indices_tensor,
                                                   starts=[start_idx],
                                                   sizes=[num_elements_in_batch])

        # tensor.gather expects backend tensors for data and indices
        yield tensor.gather(features, batch_indices_tensor, axis=0), \
              tensor.gather(labels, batch_indices_tensor, axis=0)

def main():
    epochs = 50 # Define epochs
    batch_size = 64 # Define batch_size

    # Initialize data processor
    processor = TelemetryProcessor(seq_len=64, stride=16)
    
    # Load and process data
    print("Loading telemetry data...")
    # Returns numpy arrays
    train_data_np, test_data_np = processor.load_data("network_telemetry.csv")
    
    # Create sequences (returns EmberTensors)
    print("Creating sequences...")
    train_seq_features = processor.create_sequences(train_data_np)
    test_seq_features = processor.create_sequences(test_data_np)
    
    # Create synthetic anomaly labels for demonstration
    print("Generating synthetic labels...")
    # Using tensor.random_bernoulli which outputs 0s and 1s.
    # Ensure shape matches (num_samples, 1) if loss function expects it.
    # For binary cross-entropy, labels are often (batch_size,) or (batch_size, 1)
    # Let's assume (batch_size,) is fine, or (batch_size, 1) and adjust if needed.
    # tensor.random_bernoulli needs a probability tensor as input, or a shape and a scalar p.
    # For now, using tensor.cast(tensor.random_uniform(...)) as a placeholder for bernoulli if direct equiv is tricky.
    # This creates 0s or 1s.
    # train_labels_np = np.random.binomial(1, 0.1, size=(tensor.shape(train_seq_features)[0],))
    # test_labels_np = np.random.binomial(1, 0.1, size=(tensor.shape(test_seq_features)[0],))
    # train_labels = tensor.convert_to_tensor(train_labels_np, dtype=tensor.EmberDType.float32)
    # test_labels = tensor.convert_to_tensor(test_labels_np, dtype=tensor.EmberDType.float32)
    
    # Alternative label generation using ember_ml.tensor:
    # Create a probability tensor for bernoulli (0.1 probability of being 1)
    train_p_tensor = tensor.full((tensor.shape(train_seq_features)[0],), 0.1, dtype=tensor.EmberDType.float32)
    test_p_tensor = tensor.full((tensor.shape(test_seq_features)[0],), 0.1, dtype=tensor.EmberDType.float32)
    train_labels = tensor.random_bernoulli(train_p_tensor, p=0.1) # Assuming p overrides tensor values if tensor is just shape holder
    test_labels = tensor.random_bernoulli(test_p_tensor, p=0.1)
    # Ensure labels are float32 for loss calculation if needed
    train_labels = tensor.cast(train_labels, dtype=tensor.EmberDType.float32)
    test_labels = tensor.cast(test_labels, dtype=tensor.EmberDType.float32)
    # Reshape labels to (batch_size, 1) if loss function expects 2D labels
    train_labels = tensor.reshape(train_labels, (tensor.shape(train_labels)[0], 1))
    test_labels = tensor.reshape(test_labels, (tensor.shape(test_labels)[0], 1))

    # Initialize model
    print("Building liquid neural network...")
    # input_shape for LiquidNeuralNetwork is (seq_len, num_features)
    input_shape_tuple = (processor.seq_len, tensor.shape(train_seq_features)[-1])
    lnn = LiquidNeuralNetwork(input_shape=input_shape_tuple, model_size=128)
    model = lnn.model # Get the Sequential model
    
    # Optimizer & Loss Instantiation
    # Note: Keras Adam's clipnorm/clipvalue are specific.
    # ember_optimizer.Adam might need manual gradient clipping if those exact features are required.
    optimizer = ember_optimizer.Adam(learning_rate=0.001)
    loss_fn = ember_loss.BinaryCrossEntropyLoss() # Assuming sigmoid output means binary classification

    # Training Loop
    print("\nTraining model...")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0.0
        num_batches = 0

        # Training phase
        # TODO: Implement proper shuffling for each epoch if data_generator doesn't do it internally for all data
        for batch_features, batch_labels in data_generator(train_seq_features, train_labels, batch_size, shuffle=True):
            with GradientTape() as tape:
                # Assume model's parameters are tracked by the model instance itself
                # Tape needs to watch model parameters.
                # This might require model.trainable_parameters() or similar.
                # For now, assume tape can get them from model instance passed to gradient.
                tape.watch(model.parameters()) # Explicitly watch model parameters

                predictions = model(batch_features) # Forward pass
                loss = loss_fn(batch_labels, predictions) # Note: Keras loss order is (y_true, y_pred)
                                                       # Adjust if ember_loss is (y_pred, y_true)

            gradients = tape.gradient(loss, model.parameters())
            optimizer.apply_gradients(zip(gradients, model.parameters())) # Keras-like way
            # Or if optimizer.step(gradients, model.parameters()) is the API

            # loss is a scalar backend tensor. Accumulate its Python value.
            epoch_loss += tensor.item(loss)
            num_batches += 1 # num_batches is Python int

        avg_epoch_loss = epoch_loss / float(num_batches) if num_batches > 0 else 0.0
        print(f"Training Loss: {avg_epoch_loss}") # avg_epoch_loss is Python float

        # Validation phase
        val_loss_sum = 0.0 # Accumulate as Python float
        num_val_batches = 0
        # TODO: Add metrics calculation (accuracy, AUC)
        for batch_features_val, batch_labels_val in data_generator(test_seq_features, test_labels, batch_size, shuffle=False):
            val_predictions = model(batch_features_val) # backend tensor
            loss = loss_fn(batch_labels_val, val_predictions) # scalar backend tensor
            val_loss_sum += tensor.item(loss)
            num_val_batches += 1

        avg_val_loss = val_loss_sum / float(num_val_batches) if num_val_batches > 0 else 0.0
        print(f"Validation Loss: {avg_val_loss}")

        # TODO: Implement EarlyStopping, ReduceLROnPlateau logic here if needed
        # TODO: Implement TensorBoard equivalent logging here if needed

    # Evaluate model (final evaluation on test set)
    print("\nEvaluating model...")
    final_test_loss_sum = 0.0 # Accumulate as Python float
    num_test_batches = 0
    # TODO: Add final metrics calculation (accuracy, AUC)
    for batch_features_test, batch_labels_test in data_generator(test_seq_features, test_labels, batch_size, shuffle=False):
        test_predictions = model(batch_features_test) # backend tensor
        loss = loss_fn(batch_labels_test, test_predictions) # scalar backend tensor
        final_test_loss_sum += tensor.item(loss)
        num_test_batches += 1

    avg_final_test_loss = final_test_loss_sum / float(num_test_batches) if num_test_batches > 0 else 0.0
    print(f"\nTest Results:")
    print(f"Loss: {avg_final_test_loss:.4f}")
    # Print other metrics here

    # Save model
    print("\nSaving model...")
    # model.save("liquid_anomaly_detector.h5") # Keras saving
    # Placeholder for ember_ml model saving.
    # This needs a defined way to save/load ember_ml.nn.Module states.
    try:
        # Example: model.save_weights('liquid_anomaly_detector_ember.ckpt')
        # Or: tensor.save(model.state_dict(), 'liquid_anomaly_detector_ember.pth')
        print("Model saving placeholder: `model.save_weights('path/to/weights.ckpt')` or similar needed.")
    except Exception as e:
        print(f"Model saving not yet implemented in ember_ml or failed: {e}")

    # CoreML Conversion (Commented out)
    # print("\nConverting to CoreML format...")
    # CoreML conversion would require a new path from an ember_ml saved model format.
    # try:
    #     import coremltools as ct
    #     # This part needs to be adapted based on how ember_ml models are saved and loaded.
    #     # coreml_model = ct.convert(
    #     #     "ember_ml_model_path_or_object", # Replace with actual model path/object
    #     #     inputs=[ct.TensorType(name="input", shape=(1, *input_shape_tuple))],
    #     #     compute_precision=ct.precision.FLOAT32
    #     # )
    #     # coreml_model.save("LiquidAnomalyDetector.mlpackage")
    #     # print("CoreML model saved successfully!")
    #     print("CoreML conversion skipped. Needs adaptation for ember_ml models.")
    # except ImportError:
    #     print("CoreML tools not available. Skipping CoreML conversion.")
    # except Exception as e:
    #     print(f"CoreML conversion failed: {e}")


if __name__ == "__main__":
    main()