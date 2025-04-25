"""
Test script for the fixed generate_data function.
"""

from ember_ml.ops import set_backend
from fixed_generate_data import generate_data

# Set backend to numpy
set_backend('numpy')

# Generate data with the fixed function
print("Generating synthetic data with anomalies...")
df = generate_data(n_samples=100, n_features=5, anomaly_fraction=0.05)
print(f"Generated {len(df)} samples with {df['anomaly'].sum()} anomalies")

# Display the first few rows
print("\nFirst few rows of the generated data:")
print(df.head())

print("\nTest completed successfully!")