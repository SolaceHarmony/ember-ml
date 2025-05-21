import pandas as pd

from ember_ml import ops
from ember_ml.nn import tensor # Added import
from ember_ml.nn import wirings # Added import
# from ember_ml.nn.wirings import AutoNCP # AutoNCP is used via wirings.AutoNCP
from ember_ml.nn.modules.rnn import CfC
# from ember_ml.nn import Module, Sequential # Module is not directly used, Sequential is
from ember_ml.nn import Sequential
from ember_ml.nn.container.batch_normalization import BatchNormalization
from ember_ml.nn.container.dense import Dense
from ember_ml.nn.container.common.dropout import Dropout
from ember_ml.training import Adam # Added Adam optimizer
from sklearn.preprocessing import StandardScaler

def generate_log_data(num_logs=1000):
    """Generate synthetic Splunk-like log data with more pronounced anomaly patterns"""
    log_entries = {
        "timestamp": pd.date_range(start="2025-02-11", periods=num_logs, freq="1min"),
        "location": (
            lambda choices, n: tensor.gather(
                tensor.convert_to_tensor(choices),
                tensor.random_categorical(tensor.zeros(len(choices)), n)
            ).to_numpy()
        )(["Switch_A", "Switch_B", "Switch_C", "Switch_D", "Switch_E"], num_logs),
        "message": (
            lambda choices, probs, n: tensor.gather(
                tensor.convert_to_tensor(choices),
                tensor.random_categorical(ops.log(tensor.convert_to_tensor(probs)), n)
            ).to_numpy()
        )(
            ["Link down", "Link up", "High latency", "Packet loss", "Auth failure", "Config mismatch", "Power issue"],
            [0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1],
            num_logs
        ),  # Adjusted probabilities
        "severity": (
            lambda choices, probs, n: tensor.gather(
                tensor.convert_to_tensor(choices),
                tensor.random_categorical(ops.log(tensor.convert_to_tensor(probs)), n)
            ).to_numpy()
        )(
            ["Low", "Medium", "High", "Critical"],
            [0.4, 0.3, 0.2, 0.1],
            num_logs
        )  # More realistic severity distribution
    }
    
    df = pd.DataFrame(log_entries)
    
    # Insert more distinct anomalous patterns
    num_anomalies = num_logs // 20  # Increased frequency
    # anomaly_indices = np.random.choice(num_logs-10, num_anomalies, replace=False)
    # Replace np.random.choice with tensor operations
    population_size = num_logs - 10
    if population_size >= num_anomalies and num_anomalies > 0 : # Ensure population is large enough and num_anomalies is positive
        population = tensor.arange(population_size)
        shuffled_population = tensor.random_permutation(population)
        anomaly_indices_tensor = shuffled_population[:num_anomalies]
        # Convert to numpy array for df.loc indexing if necessary, or use directly if supported
        # For now, assuming direct iteration or conversion later if needed by df.loc
        # This part might need adjustment based on how df.loc handles tensor indices
        # or if anomaly_indices is used in a context requiring a list/numpy array.
        # For simplicity in this diff, we'll assume it can be iterated or converted.
        # If direct iteration over a tensor is problematic for the loop below,
        # we might need anomaly_indices_tensor.to_numpy().tolist()
        # However, to avoid re-introducing numpy, let's assume tensor iteration works or can be adapted.
        # For the loop `for idx in anomaly_indices:`, idx needs to be an integer.
        # So, conversion to a list of Python ints is likely necessary.
        anomaly_indices = anomaly_indices_tensor.to_numpy().tolist() # Eagerly convert for the loop
    else:
        anomaly_indices = []


    for idx in anomaly_indices:
        # Create stronger correlated events
        df.loc[idx:idx+4, "severity"] = ["Critical", "Critical", "Critical", "High", "High"]
        df.loc[idx:idx+4, "location"] = df.loc[idx, "location"]
        df.loc[idx:idx+2, "message"] = "Link down"  # Three consecutive link downs
        df.loc[idx+3:idx+4, "message"] = "Packet loss"  # Followed by packet loss
        
        # Add cascade effect
        if idx + 8 < num_logs:
            df.loc[idx+5:idx+7, "severity"] = "High"
            df.loc[idx+5:idx+7, "location"] = df.loc[idx, "location"]
            df.loc[idx+5:idx+7, "message"] = ["High latency", "High latency", "Auth failure"]
    
    return df

class LiquidAnomalyDetector:
    """Anomaly detector using CfC (Closed-form Continuous-time) neural networks"""
    
    def __init__(self, total_neurons=100, motor_neurons=5, sequence_length=10):
        self.sequence_length = sequence_length
        self.location_map = {}
        self.message_map = {}
        self.severity_map = {"Low": 0.25, "Medium": 0.5, "High": 0.75, "Critical": 1.0}
        
        # Build multi-scale liquid neural network
        self.model = self._build_model(total_neurons, motor_neurons)
        self.scaler = StandardScaler()
        
    def _build_model(self, total_neurons, motor_neurons):
        # First layer - fast timescale for immediate anomaly detection
        wiring_fast = wirings.AutoNCP(
            units=total_neurons,
            output_size=motor_neurons,
            sparsity_level=0.5
        )

        ltc_fast = CfC(
            wiring_fast,
            return_sequences=True
        )

        # Second layer - medium timescale for pattern recognition
        wiring_med = wirings.AutoNCP(
            units=total_neurons // 2,
            output_size=motor_neurons // 2,
            sparsity_level=0.4
        )

        ltc_med = CfC(
            wiring_med,
            return_sequences=True
        )

        # Third layer - slow timescale for long-term dependencies
        wiring_slow = wirings.AutoNCP(
            units=total_neurons // 4,
            output_size=motor_neurons // 4,
            sparsity_level=0.3
        )

        ltc_slow = CfC(
            wiring_slow,
            return_sequences=False
        )

        model = Sequential([
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
            Dense(1, activation='sigmoid')  # Anomaly probability
        ])

        optimizer = Adam( # Replaced keras.optimizers.Adam
            lr=0.001, # Renamed learning_rate to lr
            # clipvalue=0.5, # clipvalue is not a direct parameter, may need to be handled by gradient clipping if necessary
            weight_decay=0.01 # Added weight_decay for L2 regularization
        )

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']  # Added AUC metric
        )
        
        return model
        
    def _encode_features(self, df):
        """Encode log features with temporal correlation"""
        # Update maps with new values
        for loc in df["location"].unique():
            if loc not in self.location_map:
                self.location_map[loc] = len(self.location_map)
        for msg in df["message"].unique():
            if msg not in self.message_map:
                self.message_map[msg] = len(self.message_map)
        
        # Create feature matrix
        num_features = len(self.location_map) + len(self.message_map) + 1  # +1 for severity
        features = tensor.zeros((len(df), num_features))
        
        for i, row in df.iterrows():
            # One-hot encode location
            features[i, self.location_map[row["location"]]] = 1
            
            # One-hot encode message
            msg_offset = len(self.location_map)
            features[i, msg_offset + self.message_map[row["message"]]] = 1
            
            # Add severity
            features[i, -1] = self.severity_map[row["severity"]]
        
        # Scale features
        features = self.scaler.fit_transform(features)
        
        # Create sequences
        sequences = []
        for i in range(len(features) - self.sequence_length):
            seq = features[i:i + self.sequence_length]
            # Pad or truncate features to match total_neurons
            # Assuming self.model.input_shape[-1] gives an integer for dimensions
            padded_seq = tensor.zeros((self.sequence_length, self.model.input_shape[-1]))
            # Tensor assignment will depend on EmberTensor's capabilities.
            # This might need ops.assign or direct slicing if EmberTensor supports it.
            # For now, assuming direct slice assignment works or can be adapted.
            # seq is likely a numpy array from self.scaler.fit_transform(features)
            # so it should be converted to tensor before assignment or padding.
            seq_tensor = tensor.convert_to_tensor(seq)
            padded_seq[:, :seq_tensor.shape[1]] = seq_tensor
            sequences.append(padded_seq)
            
        return tensor.convert_to_tensor(sequences) # Convert list of tensors to a single tensor
    
    def _detect_anomalies(self, sequences, threshold=0.8):
        """Detect anomalies using the liquid neural network"""
        predictions = self.model.predict(sequences)
        # Convert predictions to boolean array and flatten
        return (predictions.squeeze() > threshold).astype(bool)
    
    def _generate_labels(self, df):
        """Generate labels for training data based on known anomaly patterns"""
        labels = tensor.zeros(len(df) - self.sequence_length)
        
        # Label sequences as anomalies based on multiple criteria
        for i in range(len(df) - self.sequence_length):
            sequence = df.iloc[i:i + self.sequence_length]
            
            # Criteria 1: Multiple critical events
            critical_events = (sequence['severity'] == 'Critical').sum()
            
            # Criteria 2: Repeated severe issues
            repeated_issues = (
                ((sequence['message'] == 'Link down').sum() >= 2) or
                ((sequence['message'] == 'Auth failure').sum() >= 2) or
                ((sequence['message'] == 'Power issue').sum() >= 2)
            )
            
            # Criteria 3: Cascading failures (same location with increasing severity)
            same_location = sequence['location'].nunique() == 1
            high_severity = (sequence['severity'].isin(['High', 'Critical'])).sum() >= 3
            
            # Mark as anomaly if meets any criteria combination
            if (critical_events >= 2) or \
               (critical_events >= 1 and repeated_issues) or \
               (same_location and high_severity and repeated_issues):
                labels[i] = 1
                
        return labels

    def train(self, df, epochs=10, validation_split=0.2):
        """Train the model on log data"""
        print("\nTraining model...")
        sequences = self._encode_features(df)
        labels = self._generate_labels(df)
        
        if sequences.shape[0] == 0:
            print("Not enough data for training")
            return
        
        history = self.model.fit(
            sequences, labels,
            epochs=epochs,
            validation_split=validation_split,
            batch_size=32,
            verbose=1
        )
        return history

    def process_logs(self, df, threshold=0.9):
        """Process logs and detect anomalies with confidence scores"""
        # Encode features into sequences
        sequences = self._encode_features(df)
        
        if sequences.shape[0] == 0:
            print("Not enough data for sequence analysis")
            return tensor.convert_to_tensor([]) # Return empty EmberTensor
        
        # Train the model if it hasn't been trained
        if not hasattr(self, '_trained'):
            history = self.train(df)
            self._trained = True
            
            # Print training summary
            print("\nTraining Summary:")
            if history and hasattr(history, 'history'): # Check if history is not None and has history attribute
                final_epoch = history.history
                print(f"Final Training Accuracy: {final_epoch['accuracy'][-1]:.2%}")
                print(f"Final Validation Accuracy: {final_epoch['val_accuracy'][-1]:.2%}")
                print(f"Final AUC Score: {final_epoch['AUC'][-1]:.3f}")
                print("-" * 50)
            else:
                print("Training did not produce a history object (e.g., not enough data).")
        
        # Get raw predictions
        raw_predictions = self.model.predict(sequences)
        # Detect anomalies using threshold
        # Assuming raw_predictions.squeeze() returns a tensor-like object
        squeezed_predictions = raw_predictions.squeeze() # If squeeze is not a method, use tensor.squeeze(raw_predictions)
        anomalies = ops.greater(squeezed_predictions, threshold)
        
        # Print anomaly information
        print("\nDetected Anomalies:")
        anomaly_count = 0
        for i, (is_anomaly, confidence) in enumerate(zip(anomalies, raw_predictions)):
            if is_anomaly:
                anomaly_count += 1
                idx = i + self.sequence_length
                print(f"\nANOMALY #{anomaly_count} at {df['timestamp'].iloc[idx]}")
                print(f"Confidence: {confidence[0]:.2%}")
                print(f"Location: {df['location'].iloc[idx]}")
                print(f"Message: {df['message'].iloc[idx]}")
                print(f"Severity: {df['severity'].iloc[idx]}")
                
                # Show context (previous events at same location)
                context = df.iloc[idx-self.sequence_length:idx]
                same_location = context[context['location'] == df['location'].iloc[idx]]
                if not same_location.empty:
                    print("\nPreceding events at same location:")
                    for _, event in same_location.iterrows():
                        print(f"  {event['timestamp']}: {event['message']} ({event['severity']})")
                print("-" * 50)
        
        return anomalies

def main():
    # Generate synthetic log data
    print("Generating synthetic log data...")
    logs = generate_log_data(num_logs=1000)
    
    # Initialize detector
    print("\nInitializing liquid neural network anomaly detector...")
    detector = LiquidAnomalyDetector(total_neurons=64, motor_neurons=16, sequence_length=10)
    
    print("\nProcessing logs...")
    anomalies = detector.process_logs(logs)
    
    # Print summary
    print("\nDetection Summary:")
    # Assuming 'anomalies' is a 1D tensor of booleans or 0s and 1s
    # For sum(anomalies), if it's a boolean tensor, it might need casting or a specific sum op.
    # ops.sum should work. For len(anomalies), use .shape[0].
    num_anomalies_detected = ops.sum(tensor.cast(anomalies, tensor.int32)) # Cast to int for sum
    total_processed = anomalies.shape[0]
    print(f"Total sequences processed: {total_processed}")
    print(f"Anomalies detected: {num_anomalies_detected}")
    if total_processed > 0:
        anomaly_rate = (num_anomalies_detected / total_processed) * 100
        print(f"Anomaly rate: {anomaly_rate:.2f}%")
    else:
        print("Anomaly rate: N/A (no sequences processed)")

if __name__ == "__main__":
    main()
