#!/bin/bash

# This script cleans up temporary files created during debugging

# Remove all temporary notebook files
rm -f bigquery_feature_extraction_*_executed.ipynb
rm -f bigquery_feature_extraction_complete.ipynb
rm -f bigquery_feature_extraction_fixed.ipynb
rm -f bigquery_feature_extraction_raw.ipynb
rm -f bigquery_feature_extraction_liquid_nn_fixed.ipynb
rm -f bigquery_feature_extraction_liquid_nn_fixed_executed.ipynb

# Remove all temporary Python files
rm -f create_*.py
rm -f fixed_notebook_cell_bigframes.py
rm -f test_raw.ipynb
rm -f test_notebook.ipynb
rm -f test_notebook_executed.ipynb
rm -f test_imports.ipynb
rm -f test_imports.nbconvert.ipynb
rm -f test_imports.py

echo "Cleanup complete!"