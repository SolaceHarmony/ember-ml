# Notebook Tools Organization Plan

## Overview

This document outlines a plan for organizing the notebook simulation tools that were created to debug and test the emberharmony codebase. Currently, these tools are scattered in the project root directory, which clutters the project and makes them difficult to maintain.

## Current State

The notebook simulation tools currently exist as separate Python scripts in the project root:

```
/
├── notebook_cell_1_setup.py
├── notebook_cell_2_connection.py
├── notebook_cell_3_explore.py
├── notebook_cell_4_extract.py
├── notebook_cell_4_extract_fixed.py
├── run_notebook_simulation.py
├── NOTEBOOK_SIMULATION_README.md
├── NOTEBOOK_ISSUE_SUMMARY.md
├── NOTEBOOK_ISSUE_SUMMARY_UPDATED.md
├── FIXED_NOTEBOOK_CELL.md
├── FINAL_SUMMARY.md
└── ...
```

This approach has several drawbacks:

1. **Project Clutter**: The tools add noise to the project root directory
2. **Poor Discoverability**: It's difficult to find and understand the tools
3. **Limited Reusability**: The tools aren't structured for reuse in other contexts
4. **Maintenance Challenges**: Updates require modifying multiple files in different locations

## Proposed Structure

We propose organizing the notebook tools into a dedicated directory structure:

```
emberharmony/
├── tools/
│   └── notebook_simulation/
│       ├── README.md
│       ├── run_simulation.py
│       ├── update_notebook.py
│       ├── fix_notebook.sh
│       ├── cells/
│       │   ├── __init__.py
│       │   ├── cell_1_setup.py
│       │   ├── cell_2_connection.py
│       │   ├── cell_3_explore.py
│       │   ├── cell_4_extract.py
│       │   └── cell_4_extract_fixed.py
│       └── docs/
│           ├── notebook_issue_summary.md
│           ├── fixed_notebook_cell.md
│           └── final_summary.md
```

This structure provides several benefits:

1. **Clean Project Root**: Removes clutter from the project root
2. **Improved Discoverability**: Tools are organized in a logical structure
3. **Better Reusability**: Tools can be imported and reused in other contexts
4. **Easier Maintenance**: Related files are grouped together

## Implementation Plan

### Phase 1: Create Directory Structure

1. Create the necessary directories:
   ```bash
   mkdir -p emberharmony/tools/notebook_simulation/cells
   mkdir -p emberharmony/tools/notebook_simulation/docs
   ```

2. Create package initialization files:
   ```bash
   touch emberharmony/tools/__init__.py
   touch emberharmony/tools/notebook_simulation/__init__.py
   touch emberharmony/tools/notebook_simulation/cells/__init__.py
   ```

### Phase 2: Migrate Files

1. Move notebook cell scripts to the cells directory:
   ```bash
   mv notebook_cell_*.py emberharmony/tools/notebook_simulation/cells/
   ```

2. Rename cell scripts for consistency:
   ```bash
   cd emberharmony/tools/notebook_simulation/cells/
   for file in notebook_cell_*.py; do
     new_name=$(echo $file | sed 's/notebook_cell_/cell_/')
     mv $file $new_name
   done
   ```

3. Move runner scripts to the notebook_simulation directory:
   ```bash
   mv run_notebook_simulation.py emberharmony/tools/notebook_simulation/run_simulation.py
   mv update_notebook.py emberharmony/tools/notebook_simulation/
   mv fix_notebook.sh emberharmony/tools/notebook_simulation/
   ```

4. Move documentation files to the docs directory:
   ```bash
   mv NOTEBOOK_SIMULATION_README.md emberharmony/tools/notebook_simulation/README.md
   mv NOTEBOOK_ISSUE_SUMMARY*.md emberharmony/tools/notebook_simulation/docs/
   mv FIXED_NOTEBOOK_CELL.md emberharmony/tools/notebook_simulation/docs/fixed_notebook_cell.md
   mv FINAL_SUMMARY.md emberharmony/tools/notebook_simulation/docs/final_summary.md
   ```

### Phase 3: Update Imports and References

1. Update imports in cell scripts:
   ```python
   # Before
   import pickle
   
   # After
   import pickle
   import os
   import sys
   
   # Add parent directory to path if running as script
   if __name__ == "__main__":
       sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
   ```

2. Update run_simulation.py to use the new structure:
   ```python
   # Before
   cell_scripts = [
       'notebook_cell_1_setup.py',
       'notebook_cell_2_connection.py',
       'notebook_cell_3_explore.py',
       'notebook_cell_4_extract.py'
   ]
   
   # After
   import os
   
   CELLS_DIR = os.path.join(os.path.dirname(__file__), "cells")
   
   cell_scripts = [
       os.path.join(CELLS_DIR, 'cell_1_setup.py'),
       os.path.join(CELLS_DIR, 'cell_2_connection.py'),
       os.path.join(CELLS_DIR, 'cell_3_explore.py'),
       os.path.join(CELLS_DIR, 'cell_4_extract.py')
   ]
   ```

3. Update pickle file paths in cell scripts:
   ```python
   # Before
   with open('notebook_cell_1_output.pkl', 'wb') as f:
       pickle.dump(output_data, f)
   
   # After
   output_dir = os.path.dirname(os.path.abspath(__file__))
   output_path = os.path.join(output_dir, 'cell_1_output.pkl')
   
   with open(output_path, 'wb') as f:
       pickle.dump(output_data, f)
   ```

### Phase 4: Create Package Structure

1. Update `__init__.py` files to expose key functionality:

   **emberharmony/tools/notebook_simulation/\_\_init\_\_.py**:
   ```python
   """
   Notebook simulation tools for emberharmony.
   
   This package provides tools for simulating Jupyter notebooks
   to debug and test emberharmony functionality.
   """
   
   from .run_simulation import run_cell, main as run_simulation
   
   __all__ = ['run_cell', 'run_simulation']
   ```

   **emberharmony/tools/notebook_simulation/cells/\_\_init\_\_.py**:
   ```python
   """
   Notebook cell implementations for simulation.
   
   This module contains implementations of individual notebook cells
   that can be run independently or as part of a simulation.
   """
   
   import os
   
   # Define cell script paths
   CELL_1_SETUP = os.path.join(os.path.dirname(__file__), 'cell_1_setup.py')
   CELL_2_CONNECTION = os.path.join(os.path.dirname(__file__), 'cell_2_connection.py')
   CELL_3_EXPLORE = os.path.join(os.path.dirname(__file__), 'cell_3_explore.py')
   CELL_4_EXTRACT = os.path.join(os.path.dirname(__file__), 'cell_4_extract.py')
   CELL_4_EXTRACT_FIXED = os.path.join(os.path.dirname(__file__), 'cell_4_extract_fixed.py')
   
   # Define cell sequence
   DEFAULT_CELL_SEQUENCE = [
       CELL_1_SETUP,
       CELL_2_CONNECTION,
       CELL_3_EXPLORE,
       CELL_4_EXTRACT
   ]
   
   FIXED_CELL_SEQUENCE = [
       CELL_1_SETUP,
       CELL_2_CONNECTION,
       CELL_3_EXPLORE,
       CELL_4_EXTRACT_FIXED
   ]
   
   __all__ = [
       'CELL_1_SETUP', 'CELL_2_CONNECTION', 'CELL_3_EXPLORE', 
       'CELL_4_EXTRACT', 'CELL_4_EXTRACT_FIXED',
       'DEFAULT_CELL_SEQUENCE', 'FIXED_CELL_SEQUENCE'
   ]
   ```

2. Create a setup.py for the tools package:
   ```python
   from setuptools import setup, find_packages

   setup(
       name="emberharmony-tools",
       version="0.1.0",
       packages=find_packages(),
       description="Tools for emberharmony development and testing",
       author="emberharmony Team",
       author_email="team@emberharmony.ai",
       install_requires=[
           "pandas",
           "numpy",
           "matplotlib",
           "google-cloud-bigquery",
           "bigframes"
       ],
       entry_points={
           'console_scripts': [
               'run-notebook-simulation=emberharmony.tools.notebook_simulation.run_simulation:main',
               'fix-notebook=emberharmony.tools.notebook_simulation.update_notebook:main',
           ],
       },
   )
   ```

### Phase 5: Update Documentation

1. Update README.md with new usage instructions:
   ```markdown
   # Notebook Simulation Tools

   This package provides tools for simulating Jupyter notebooks to debug and test emberharmony functionality.

   ## Installation

   ```bash
   # Install the package in development mode
   pip install -e .
   ```

   ## Usage

   ### Running the Simulation

   ```bash
   # Run the entire simulation
   run-notebook-simulation

   # Run a specific cell
   python -m emberharmony.tools.notebook_simulation.cells.cell_1_setup
   ```

   ### Fixing Notebooks

   ```bash
   # Fix a notebook
   fix-notebook path/to/notebook.ipynb
   ```

   ## Directory Structure

   - `cells/`: Individual notebook cell implementations
   - `docs/`: Documentation about notebook issues and fixes
   - `run_simulation.py`: Script to run the entire simulation
   - `update_notebook.py`: Script to fix notebooks with issues
   ```

## Testing Plan

1. **Functionality Testing**:
   - Run the simulation with the new structure
   - Verify that all cells execute correctly
   - Test the notebook fixing functionality

2. **Import Testing**:
   - Test importing the tools as a package
   - Verify that the entry points work correctly

3. **Documentation Testing**:
   - Verify that all documentation links work
   - Ensure that the README provides clear instructions

## Benefits

Organizing the notebook tools in this way provides several benefits:

1. **Cleaner Project Structure**: Removes clutter from the project root
2. **Improved Discoverability**: Makes it easier to find and understand the tools
3. **Better Reusability**: Allows the tools to be imported and reused in other contexts
4. **Easier Maintenance**: Groups related files together for easier updates
5. **Package Management**: Enables installation and distribution as a package

## Conclusion

This plan provides a comprehensive approach to organizing the notebook simulation tools in a dedicated directory structure. By implementing this plan, we will improve the maintainability and usability of these tools, making it easier to debug and test emberharmony functionality in the future.