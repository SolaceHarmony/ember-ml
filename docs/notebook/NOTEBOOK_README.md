# BigQuery Data Preparation and Feature Extraction for Liquid Neural Networks

This notebook demonstrates how to extract and process data from BigQuery tables, train Restricted Boltzmann Machines (RBMs), and feed the output into liquid neural networks with motor neurons for triggering deeper exploration.

## Prerequisites

Before running this notebook, you need to install the required packages:

```bash
pip install google-cloud-bigquery bigframes matplotlib emberharmony
```

You also need to set up your Google Cloud credentials:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/credentials.json"
```

## Important: Jupyter Kernel Configuration

The notebook requires `bigframes` and `emberharmony` to be available in the Python environment used by Jupyter. Our testing shows that Jupyter might be using a different Python environment than the one where these packages are installed.

### Checking Your Jupyter Kernel

Run the following code in a notebook cell to check which Python environment Jupyter is using:

```python
import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    import bigframes
    print(f"bigframes version: {bigframes.__version__}")
except ImportError as e:
    print(f"Error importing bigframes: {e}")

try:
    import emberharmony
    print(f"emberharmony imported successfully")
except ImportError as e:
    print(f"Error importing emberharmony: {e}")
```

### Setting Up a Conda Environment for Jupyter

If you're using Conda, you can create a new environment with all the required packages and register it as a Jupyter kernel:

```bash
# Create a new conda environment
conda create -n bigframes_env python=3.12

# Activate the environment
conda activate bigframes_env

# Install the required packages
pip install google-cloud-bigquery bigframes matplotlib emberharmony

# Install ipykernel to make the environment available to Jupyter
pip install ipykernel

# Register the environment as a Jupyter kernel
python -m ipykernel install --user --name bigframes_env --display-name "Python (bigframes_env)"
```

Then, when you open the notebook, select the "Python (bigframes_env)" kernel from the kernel menu.

### Using the Existing Conda Environment

If you already have a Conda environment with the required packages installed, you can register it as a Jupyter kernel:

```bash
# Activate the environment
conda activate your_env_name

# Install ipykernel if not already installed
pip install ipykernel

# Register the environment as a Jupyter kernel
python -m ipykernel install --user --name your_env_name --display-name "Python (your_env_name)"
```

Then, when you open the notebook, select the "Python (your_env_name)" kernel from the kernel menu.

## Running the Notebook

1. Open the notebook in Jupyter or JupyterLab:

```bash
jupyter notebook bigquery_feature_extraction_liquid_nn.ipynb
```

2. Select the kernel that has `bigframes` and `emberharmony` installed.

3. Run the cells in order, starting with the setup and imports cell.

4. Make sure to update the following variables in the notebook:
   - `PROJECT_ID`: Your Google Cloud project ID
   - `CREDENTIALS_PATH`: Path to your service account credentials
   - `TABLE_ID`: The BigQuery table ID to use
   - `TARGET_COLUMN`: The target column for prediction (optional)
   - `LIMIT`: The number of rows to limit the query to (for testing)

## Notebook Structure

The notebook is organized into the following sections:

1. **Setup and Imports**: Sets up the environment and imports required libraries
2. **BigQuery Connection Setup**: Configures the connection to BigQuery
3. **Explore Available Tables**: Explores available tables in the BigQuery project
4. **Extract Features from BigQuery**: Extracts features from a BigQuery table using BigFrames
5. **Apply Temporal Stride Processing**: Applies temporal stride processing to the extracted features
6. **Train RBM and Liquid Neural Network**: Trains an RBM, extracts features, trains a liquid network, and analyzes the results
7. **Save Models**: Saves the trained models for future use
8. **Using the Integrated Pipeline**: Demonstrates how to use the integrated pipeline for a more streamlined workflow
9. **Conclusion**: Summarizes the pipeline and provides next steps

## Troubleshooting

If you encounter any errors related to missing modules, make sure you have selected the correct kernel that has all the required packages installed.

If you encounter errors related to the `get_backend()` function, make sure you're importing it from the correct module:

```python
from emberharmony.backend import get_backend
```

## Notes

- The notebook is designed to work with terabyte-sized tables efficiently through chunked processing.
- It uses BigFrames for BigQuery integration and emberharmony for GPU-accelerated tensor operations.
- The backend is automatically detected and used based on what's available on your system.