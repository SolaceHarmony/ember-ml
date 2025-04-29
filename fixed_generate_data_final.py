def generate_data(n_samples=1000, n_features=10, anomaly_fraction=0.05):
    """Generate synthetic data with anomalies."""
    # Import necessary modules
    from ember_ml.nn import tensor
    from ember_ml import ops
    import numpy as np
    import pandas as pd
    
    # Generate normal data
    normal_data = tensor.random_normal((n_samples, n_features), mean=0.0, stddev=1.0)
    normal_data = tensor.to_numpy(normal_data)
    
    # Add correlations between features
    normal_tensor = tensor.convert_to_tensor(normal_data)
    for i in range(1, n_features):
        feature_i = tensor.slice_tensor(normal_tensor, [0, i], [-1, 1])
        feature_0 = tensor.slice_tensor(normal_tensor, [0, 0], [-1, 1])
        weighted_i = ops.multiply(feature_i, 0.5)
        weighted_0 = ops.multiply(feature_0, 0.5)
        combined = ops.add(weighted_i, weighted_0)
        # Update the tensor using bracket assignment
        normal_tensor[:, i] = tensor.squeeze(combined, axis=1)
    
    # Add temporal patterns
    for i in range(n_samples):
        time_value = ops.divide(tensor.convert_to_tensor(i, dtype=tensor.float32), 50.0)
        sin_value = ops.multiply(ops.sin(time_value), 0.5)
        row = tensor.slice_tensor(normal_tensor, [i, 0], [1, -1])
        updated_row = ops.add(row, sin_value)
        # Update the tensor using bracket assignment
        normal_tensor[i, :] = tensor.squeeze(updated_row, axis=0)
    
    # Convert back to numpy
    normal_data = tensor.to_numpy(normal_tensor)
    
    # Generate anomalies
    n_anomalies = int(n_samples * anomaly_fraction)
    anomaly_indices = ops.random_choice(n_samples, n_anomalies, replace=False)
    
    # Create different types of anomalies
    normal_tensor = tensor.convert_to_tensor(normal_data)
    for idx in anomaly_indices:
        anomaly_type = np.random.randint(0, 3)
        
        if anomaly_type == 0:  # Spike anomaly
            feature_idx = np.random.randint(0, n_features)
            spike_value = tensor.random_uniform(3.0, 5.0)
            current_value = normal_tensor[idx, feature_idx]
            updated_value = ops.add(current_value, tensor.convert_to_tensor(spike_value))
            # Update the tensor using bracket assignment
            normal_tensor[idx, feature_idx] = updated_value
            
        elif anomaly_type == 1:  # Correlation anomaly
            random_values = tensor.random_normal((n_features,), mean=0.0, stddev=1.0)
            # Update the tensor using bracket assignment
            normal_tensor[idx, :] = random_values
            
        else:  # Collective anomaly
            random_values = tensor.random_uniform((n_features,), minval=2.0, maxval=3.0)
            current_row = normal_tensor[idx, :]
            updated_row = ops.add(current_row, random_values)
            # Update the tensor using bracket assignment
            normal_tensor[idx, :] = updated_row
    
    # Convert back to numpy
    data = tensor.to_numpy(normal_tensor)
    
    # Create DataFrame
    columns = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=columns)
    
    # Add anomaly label
    df['anomaly'] = 0
    df.loc[anomaly_indices, 'anomaly'] = 1
    
    return df