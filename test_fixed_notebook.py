"""
Test script for the fixed notebook code.
"""

# Import the fixed notebook code
from fixed_notebook import generate_data

# Generate data with the fixed function
print("Testing fixed notebook code...")
print("Generating synthetic data with anomalies...")
df = generate_data(n_samples=100, n_features=5, anomaly_fraction=0.05)
print(f"Generated {len(df)} samples with {df['anomaly'].sum()} anomalies")

# Display the first few rows
print("\nFirst few rows of the generated data:")
print(df.head())

print("\nTest completed successfully!")