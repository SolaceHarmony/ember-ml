# Notebook Simulation

This directory contains a simulation of a Jupyter notebook using Python scripts. Each script represents a cell in the notebook, and they can be run in sequence to simulate the notebook execution.

## How it Works

1. Each script (`notebook_cell_X_*.py`) represents a cell in the notebook.
2. Each script pickles its output data to a file (`notebook_cell_X_output.pkl`).
3. The next script loads the pickled data from the previous script to continue the workflow.
4. The `run_notebook_simulation.py` script runs all the cells in sequence.

## Files

- `notebook_cell_1_setup.py`: Set up imports and initialize variables
- `notebook_cell_2_connection.py`: Set up BigQuery connection
- `notebook_cell_3_explore.py`: Explore available tables
- `notebook_cell_4_extract.py`: Extract features from BigQuery (original with issue)
- `notebook_cell_4_extract_fixed.py`: Extract features from BigQuery (fixed)

## Running the Simulation

To run the entire notebook simulation:

```bash
python run_notebook_simulation.py
```

To run individual cells:

```bash
python notebook_cell_1_setup.py
python notebook_cell_2_connection.py
python notebook_cell_3_explore.py
python notebook_cell_4_extract_fixed.py  # Use the fixed version
```

## Debugging

If you encounter any issues, you can:

1. Check the output files (`notebook_cell_X_output.pkl`) to see what data is being passed between cells.
2. Modify the scripts to add more logging or debugging information.
3. Run individual cells to isolate the issue.

## Key Fixes

The main issue in the original notebook was that we were trying to pass a `process_chunk` function to the `prepare_data` method of the `TerabyteFeatureExtractor` class, but this method doesn't accept a `processing_fn` parameter directly. This resulted in the error:

```
TypeError: TerabyteFeatureExtractor.prepare_data() got an unexpected keyword argument 'processing_fn'
```

The fix is simple: remove the `processing_fn` parameter from the call to `prepare_data`. Here's the corrected code:

```python
# Extract features - FIXED: removed processing_fn parameter
result = feature_extractor.prepare_data(
    table_id=TABLE_ID,
    target_column=TARGET_COLUMN,
    limit=LIMIT
    # REMOVED: processing_fn=process_chunk
)
```

This ensures that the `prepare_data` method is called with the correct parameters.