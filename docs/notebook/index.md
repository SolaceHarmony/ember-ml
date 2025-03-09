# Notebook Documentation

This directory contains documentation related to Jupyter notebooks, notebook issues, and their fixes.

## Available Documentation

- [**Notebook README**](NOTEBOOK_README.md): Overview of the BigQuery Data Preparation and Feature Extraction notebook
- [**Notebook Issue Summary**](NOTEBOOK_ISSUE_SUMMARY.md): Summary of the issue with the `TerabyteFeatureExtractor.prepare_data()` method
- [**Notebook Simulation README**](NOTEBOOK_SIMULATION_README.md): Instructions for running the notebook simulation
- [**Notebook Issue Resolution**](notebook_issue_resolution.md): Complete documentation of the issue, investigation, fix, and lessons learned

## Key Issues and Fixes

The main issue documented here is related to the `TerabyteFeatureExtractor.prepare_data()` method in a Jupyter notebook. The issue was that we were trying to pass a `process_chunk` function to the `prepare_data` method, but this method doesn't accept a `processing_fn` parameter directly.

The fix was to remove the `processing_fn` parameter from the call to `prepare_data`.

## Notebook Simulation

To understand how the notebook works and how the fix was implemented, you can run the notebook simulation:

```bash
python run_notebook_simulation.py
```

This will run all the cells in sequence and show the output of each cell.

## Best Practices

Based on the lessons learned from fixing the notebook issue, here are some best practices for working with Jupyter notebooks:

1. **Understand the API**: Make sure you understand the API of the functions you're using, especially when working with complex data processing pipelines.

2. **Read the Source Code**: When encountering issues, reading the source code can provide valuable insights into how the function is supposed to be used.

3. **Create Simulations**: Creating simple simulations can be a powerful way to debug issues in a controlled environment, especially when working with complex systems like Jupyter notebooks.

4. **Use Pickle for State Management**: Using pickle to save and load state between scripts is a simple way to simulate how Jupyter maintains state between cells.