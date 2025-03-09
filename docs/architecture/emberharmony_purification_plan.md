# emberharmony Purification Plan

## Overview

This document outlines a comprehensive plan to purify the emberharmony codebase by:

1. Replacing direct NumPy usage with emberharmony's backend abstraction system
2. Improving BigQuery data handling for terabyte-scale datasets
3. Organizing notebook simulation tools in a dedicated directory
4. Centralizing documentation in the docs folder

## 1. Backend Abstraction Implementation

### Current Issues

The `terabyte_feature_extractor.py` file directly uses NumPy throughout the codebase:

```python
# Direct NumPy usage examples
import numpy as np
np.random.seed(42)
df['__split_rand'] = np.random.rand(len(df))
df[f'{col}_sin_hour'] = np.sin(2 * np.pi * df[col].dt.hour / 23.0)
batch_data = np.vstack([self.state_buffer, batch_data])
windows_array = np.array(windows)
```

This prevents the code from leveraging emberharmony's backend system that can automatically choose between MLX, PyTorch, or NumPy based on availability, resulting in missed opportunities for GPU acceleration.

### Proposed Changes

Replace direct NumPy operations with emberharmony's ops module:

```python
# Before
import numpy as np
array_data = np.array(some_data)
mean_value = np.mean(array_data)
sin_value = np.sin(2 * np.pi * value / 23.0)

# After
from emberharmony import ops
array_data = ops.convert_to_tensor(some_data)
mean_value = ops.mean(array_data)
sin_value = ops.sin(2 * ops.pi * value / 23.0)
```

### Implementation Steps

1. **Audit NumPy Usage**:
   - Identify all direct NumPy imports and function calls
   - Map NumPy functions to their emberharmony.ops equivalents

2. **Create Backend Utilities**:
   - Develop utility functions for common operations
   - Ensure proper tensor conversion between frameworks

3. **Refactor Code**:
   - Replace NumPy imports with emberharmony.ops
   - Convert array operations to use the backend system
   - Update random number generation to use ops.random

4. **Testing**:
   - Create unit tests to verify backend-agnostic behavior
   - Benchmark performance across different backends

5. **Documentation**:
   - Update API documentation to reflect backend-agnostic approach
   - Create developer guidelines for future code contributions

## 2. BigQuery Data Handling Improvements

### Current Issues

The current implementation uses pandas DataFrames for processing BigQuery data:

```python
# Current approach
df = self.process_bigquery_in_chunks(table_id=table_id, processing_fn=process_chunk)
# Later combines results using pandas
return pd.concat(results, ignore_index=True)
```

This approach can fail with terabyte-scale datasets due to memory constraints, as pandas loads entire DataFrames into memory.

### Proposed Changes

Implement true streaming processing for BigQuery data:

```python
# Streaming approach
def process_bigquery_streaming(self, table_id, processing_fn, **kwargs):
    """Process BigQuery data in a streaming fashion without loading entire dataset."""
    query = self.optimize_bigquery_query(table_id, **kwargs)
    
    # Use BigQuery Storage API for streaming
    from google.cloud.bigquery_storage import BigQueryReadClient
    from google.cloud.bigquery_storage import types
    
    client = BigQueryReadClient()
    session = client.create_read_session(
        parent=f"projects/{self.project_id}",
        read_session=types.ReadSession(
            table=f"projects/{self.project_id}/datasets/{dataset_id}/tables/{table_name}",
            data_format=types.DataFormat.ARROW,
        ),
    )
    
    # Process data in streams
    for stream in client.read_rows(session.streams[0].name):
        # Convert Arrow RecordBatch to a format suitable for processing
        batch_data = convert_arrow_to_processable(stream.arrow_record_batch)
        # Process this batch
        processing_fn(batch_data)
```

### Implementation Steps

1. **Research BigQuery Best Practices**:
   - Study BigQuery Storage API for streaming access
   - Investigate Arrow-based processing for memory efficiency

2. **Develop Streaming Components**:
   - Create streaming iterator classes
   - Implement Arrow-based data conversion utilities

3. **Refactor Data Processing**:
   - Update process_bigquery_in_chunks to use streaming
   - Modify data combination logic to avoid full materialization

4. **Memory Management**:
   - Implement adaptive chunk sizing based on memory usage
   - Add memory monitoring and circuit breakers

5. **Error Handling**:
   - Develop robust recovery mechanisms for streaming failures
   - Implement checkpointing for long-running operations

## 3. Notebook Tools Organization

### Current Issues

The notebook simulation tools are scattered in the project root directory:

```
/
├── notebook_cell_1_setup.py
├── notebook_cell_2_connection.py
├── notebook_cell_3_explore.py
├── notebook_cell_4_extract.py
├── notebook_cell_4_extract_fixed.py
├── run_notebook_simulation.py
├── NOTEBOOK_SIMULATION_README.md
└── ...
```

This clutters the project root and makes it difficult to maintain these tools.

### Proposed Structure

```
emberharmony/
├── tools/
│   └── notebook_simulation/
│       ├── README.md
│       ├── run_simulation.py
│       └── cells/
│           ├── cell_1_setup.py
│           ├── cell_2_connection.py
│           ├── cell_3_explore.py
│           └── cell_4_extract.py
```

### Implementation Steps

1. **Create Directory Structure**:
   - Create the notebook_simulation directory and subdirectories
   - Set up proper Python package structure with __init__.py files

2. **Migrate Files**:
   - Move notebook cell scripts to the cells directory
   - Update imports and relative paths

3. **Refactor Runner Script**:
   - Update run_notebook_simulation.py to work with the new structure
   - Rename to run_simulation.py for clarity

4. **Documentation**:
   - Update README with instructions for the new structure
   - Add examples of how to use the simulation tools

## 4. Documentation Reorganization

### Current Issues

Documentation is scattered throughout the project:

```
/
├── NOTEBOOK_ISSUE_SUMMARY.md
├── NOTEBOOK_ISSUE_SUMMARY_UPDATED.md
├── FIXED_NOTEBOOK_CELL.md
├── FINAL_SUMMARY.md
└── ...
```

### Proposed Structure

```
emberharmony/
├── docs/
│   ├── architecture/
│   │   ├── backend_system.md
│   │   ├── data_processing.md
│   │   └── emberharmony_purification_plan.md
│   ├── tutorials/
│   │   ├── bigquery_integration.md
│   │   └── feature_extraction.md
│   ├── troubleshooting/
│   │   └── notebook_issues.md
│   └── examples/
│       └── terabyte_processing.md
```

### Implementation Steps

1. **Create Documentation Structure**:
   - Set up the docs directory with appropriate subdirectories
   - Create index files for navigation

2. **Migrate Content**:
   - Move existing documentation to appropriate locations
   - Update cross-references between documents

3. **Standardize Format**:
   - Establish consistent Markdown formatting
   - Create templates for different document types

4. **Generate API Documentation**:
   - Set up automatic API documentation generation
   - Ensure docstrings follow a consistent format

## Implementation Timeline

### Phase 1: Planning and Setup (1-2 weeks)
- Complete detailed audit of NumPy usage
- Research BigQuery best practices
- Create directory structures for tools and docs
- Develop testing strategy

### Phase 2: Backend Abstraction (2-3 weeks)
- Implement emberharmony.ops replacements
- Create utility functions
- Update random number generation
- Test and benchmark

### Phase 3: BigQuery Streaming (3-4 weeks)
- Implement streaming components
- Refactor data processing logic
- Add memory management
- Test with large datasets

### Phase 4: Organization and Documentation (1-2 weeks)
- Migrate notebook tools
- Reorganize documentation
- Update cross-references
- Create index and navigation

## Conclusion

This purification plan addresses the core architectural issues in the emberharmony codebase. By implementing these changes, we will:

1. Enable automatic backend selection for optimal performance
2. Handle terabyte-scale data efficiently
3. Improve code organization and maintainability
4. Provide comprehensive, well-organized documentation

The result will be a more robust, efficient, and maintainable codebase that better leverages GPU resources and follows best practices for large-scale data processing.