# BigQuery Streaming Implementation Guide

## Overview

This document provides a detailed implementation guide for improving BigQuery data handling in the `terabyte_feature_extractor.py` file to efficiently process terabyte-scale datasets. This is part of the larger emberharmony purification plan.

## Background

The current implementation uses pandas DataFrames for processing BigQuery data, which can cause memory issues with terabyte-scale datasets. Google Cloud's BigQuery supports datasets so large that it's impossible to hold them in memory, requiring a streaming approach to process data efficiently.

## Current State Analysis

The current implementation has several limitations:

1. **Memory Constraints**: Loading entire chunks into pandas DataFrames can exhaust memory
2. **Inefficient Chunking**: The current chunking mechanism uses LIMIT/OFFSET, which is inefficient for large datasets
3. **Pandas Dependency**: Heavy reliance on pandas for data processing limits scalability
4. **Concatenation Bottleneck**: Combining results with `pd.concat()` requires all data to fit in memory

Key problematic code patterns:

```python
# Loading entire chunks into memory
chunk_df = bf.read_gbq(chunk_query)

# Combining results in memory
return pd.concat(results, ignore_index=True)
```

## Implementation Strategy

### 1. BigQuery Storage API Integration

Replace standard BigQuery API calls with the BigQuery Storage API, which provides streaming access to data:

```python
from google.cloud.bigquery_storage import BigQueryReadClient
from google.cloud.bigquery_storage import types as bq_storage_types

def create_read_session(self, table_id, selected_fields=None):
    """Create a BigQuery Storage API read session."""
    client = BigQueryReadClient()
    project_id, dataset_id, table_id = self._parse_table_id(table_id)
    table_path = f"projects/{project_id}/datasets/{dataset_id}/tables/{table_id}"
    
    # Create read session
    read_options = None
    if selected_fields:
        read_options = bq_storage_types.ReadSession.TableReadOptions(
            selected_fields=selected_fields
        )
    
    session = client.create_read_session(
        parent=f"projects/{project_id}",
        read_session=bq_storage_types.ReadSession(
            table=table_path,
            data_format=bq_storage_types.DataFormat.ARROW,
            read_options=read_options
        ),
        max_stream_count=self.max_streams
    )
    
    return client, session
```

### 2. Arrow-Based Processing

Use Apache Arrow for memory-efficient data processing:

```python
import pyarrow as pa

def process_arrow_batch(self, arrow_batch, processing_fn=None):
    """Process an Arrow RecordBatch efficiently."""
    # Convert to pandas only if needed by processing_fn
    if processing_fn and hasattr(processing_fn, 'requires_pandas') and processing_fn.requires_pandas:
        # Convert to pandas with zero-copy when possible
        pandas_batch = arrow_batch.to_pandas()
        return processing_fn(pandas_batch)
    elif processing_fn:
        # Process Arrow batch directly
        return processing_fn(arrow_batch)
    else:
        return arrow_batch
```

### 3. Streaming Iterator Implementation

Create a streaming iterator to process data without loading everything into memory:

```python
class BigQueryStreamingIterator:
    """Iterator for streaming BigQuery data using Storage API."""
    
    def __init__(self, client, session, processing_fn=None, max_rows_per_batch=None):
        self.client = client
        self.session = session
        self.processing_fn = processing_fn
        self.max_rows_per_batch = max_rows_per_batch
        self.streams = session.streams
        self.current_stream_index = 0
        self.current_stream_iterator = None
        
    def __iter__(self):
        return self
        
    def __next__(self):
        # If we don't have a stream iterator or it's exhausted, get the next stream
        if self.current_stream_iterator is None or not self._has_next():
            if self.current_stream_index >= len(self.streams):
                raise StopIteration
                
            # Get the next stream
            stream = self.streams[self.current_stream_index]
            self.current_stream_iterator = self.client.read_rows(stream.name)
            self.current_stream_index += 1
            
        # Get the next batch from the current stream
        arrow_batch = next(self.current_stream_iterator).arrow_record_batch
        
        # Process the batch if needed
        if self.processing_fn:
            return self.processing_fn(arrow_batch)
        else:
            return arrow_batch
            
    def _has_next(self):
        """Check if the current stream has more data."""
        try:
            # Peek at the next item
            next(self.current_stream_iterator)
            # Reset the iterator (this is a bit of a hack, but necessary)
            self.current_stream_iterator = self.client.read_rows(
                self.streams[self.current_stream_index - 1].name
            )
            return True
        except StopIteration:
            return False
```

### 4. Memory-Efficient Processing

Implement memory-efficient processing functions:

```python
def process_bigquery_streaming(
    self,
    table_id: str,
    processing_fn: Optional[callable] = None,
    selected_fields: Optional[List[str]] = None,
    where_clause: Optional[str] = None,
    max_streams: int = 4
) -> Generator:
    """
    Process a BigQuery table in a streaming fashion to handle terabyte-scale data.
    
    Args:
        table_id: BigQuery table ID (dataset.table)
        processing_fn: Function to apply to each batch
        selected_fields: List of fields to select (None for all)
        where_clause: Optional WHERE clause for filtering
        max_streams: Maximum number of parallel streams
        
    Returns:
        Generator yielding processed batches
    """
    # Apply filtering if needed
    if where_clause:
        # Create a temporary view with the filter applied
        temp_view_id = f"temp_view_{int(time.time())}"
        query = f"CREATE OR REPLACE VIEW `{temp_view_id}` AS SELECT * FROM `{table_id}` WHERE {where_clause}"
        self._execute_query(query)
        table_id = temp_view_id
    
    # Create read session
    client, session = self.create_read_session(table_id, selected_fields)
    
    # Create streaming iterator
    stream_iterator = BigQueryStreamingIterator(
        client, 
        session, 
        processing_fn=lambda batch: self.process_arrow_batch(batch, processing_fn),
        max_rows_per_batch=self.chunk_size
    )
    
    # Process data in a streaming fashion
    for batch_idx, batch in enumerate(stream_iterator):
        logger.info(f"Processing batch {batch_idx+1}")
        
        # Monitor memory usage
        self._monitor_memory()
        
        # Yield the processed batch
        yield batch
        
        # Force garbage collection
        gc.collect()
    
    # Clean up temporary view if created
    if where_clause:
        self._execute_query(f"DROP VIEW IF EXISTS `{temp_view_id}`")
```

### 5. Aggregation Strategy

Implement memory-efficient aggregation for results:

```python
def aggregate_streaming_results(
    self,
    stream_generator: Generator,
    aggregation_type: str = 'concat',
    max_memory_percentage: float = 0.7
) -> Union[pd.DataFrame, List, Dict]:
    """
    Aggregate streaming results in a memory-efficient way.
    
    Args:
        stream_generator: Generator yielding processed batches
        aggregation_type: Type of aggregation ('concat', 'list', or 'custom')
        max_memory_percentage: Maximum memory percentage to use before flushing
        
    Returns:
        Aggregated results
    """
    if aggregation_type == 'concat':
        # For DataFrame concatenation, use disk-based chunking
        import tempfile
        import os
        
        # Create temporary directory for chunks
        with tempfile.TemporaryDirectory() as temp_dir:
            chunk_files = []
            current_chunk = []
            current_memory = 0
            max_memory = self.max_memory_gb * 1024 * 1024 * 1024 * max_memory_percentage
            
            # Process batches
            for batch_idx, batch in enumerate(stream_generator):
                current_chunk.append(batch)
                
                # Check memory usage
                current_memory = self._get_current_memory_usage()
                
                # If memory usage exceeds threshold, flush to disk
                if current_memory > max_memory:
                    # Combine current chunk
                    if isinstance(current_chunk[0], pd.DataFrame):
                        combined = pd.concat(current_chunk, ignore_index=True)
                    elif isinstance(current_chunk[0], pa.RecordBatch):
                        combined = pa.concat_tables([pa.Table.from_batches([batch]) for batch in current_chunk])
                    
                    # Save to disk
                    chunk_file = os.path.join(temp_dir, f"chunk_{batch_idx}.parquet")
                    if isinstance(combined, pd.DataFrame):
                        combined.to_parquet(chunk_file)
                    else:
                        combined.to_pandas().to_parquet(chunk_file)
                    
                    chunk_files.append(chunk_file)
                    current_chunk = []
                    
                    # Force garbage collection
                    gc.collect()
            
            # Handle any remaining chunks
            if current_chunk:
                if isinstance(current_chunk[0], pd.DataFrame):
                    combined = pd.concat(current_chunk, ignore_index=True)
                elif isinstance(current_chunk[0], pa.RecordBatch):
                    combined = pa.concat_tables([pa.Table.from_batches([batch]) for batch in current_chunk])
                
                chunk_file = os.path.join(temp_dir, f"chunk_final.parquet")
                if isinstance(combined, pd.DataFrame):
                    combined.to_parquet(chunk_file)
                else:
                    combined.to_pandas().to_parquet(chunk_file)
                
                chunk_files.append(chunk_file)
            
            # Combine all chunks
            if chunk_files:
                return pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
            else:
                return pd.DataFrame()
    
    elif aggregation_type == 'list':
        # Simply collect all results in a list
        return list(stream_generator)
    
    else:
        # Custom aggregation (e.g., for statistical aggregates)
        results = {}
        for batch in stream_generator:
            # Update results based on batch
            # This is a placeholder for custom aggregation logic
            pass
        return results
```

## Implementation Plan

### Phase 1: BigQuery Storage API Integration (1 week)

1. Add BigQuery Storage API dependencies:
   - Update requirements.txt with new dependencies
   - Create compatibility layer for different versions

2. Implement core streaming components:
   - Create BigQueryStreamingIterator class
   - Implement Arrow batch processing utilities

3. Update connection handling:
   - Modify setup_bigquery_connection to support Storage API
   - Add configuration options for streaming parameters

### Phase 2: Processing Pipeline Refactoring (2 weeks)

1. Implement streaming query execution:
   - Create process_bigquery_streaming method
   - Add support for filtering and projection

2. Develop memory-efficient aggregation:
   - Implement disk-based chunking for large results
   - Create adaptive memory management

3. Update data preparation methods:
   - Modify prepare_data to use streaming
   - Update _split_data for streaming compatibility

### Phase 3: Integration and Testing (1 week)

1. Implement comprehensive tests:
   - Unit tests for streaming components
   - Integration tests with large datasets
   - Performance benchmarks against current implementation

2. Create documentation:
   - Update API documentation
   - Add examples of streaming usage
   - Create troubleshooting guide

## Code Examples for Key Components

### Updated TerabyteFeatureExtractor Class

```python
class TerabyteFeatureExtractor:
    """
    Feature extractor optimized for terabyte-scale BigQuery tables.
    
    This class handles feature extraction from very large BigQuery tables by:
    - Streaming data using BigQuery Storage API
    - Processing data in memory-efficient batches
    - Optimizing for large-scale operations
    - Providing progress tracking
    """
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "US",
        chunk_size: int = 100000,
        max_memory_gb: float = 16.0,
        max_streams: int = 4,
        verbose: bool = True
    ):
        """Initialize the terabyte-scale feature extractor."""
        # Existing initialization code...
        
        # Add new parameters
        self.max_streams = max_streams
        
        # Initialize BigQuery Storage API client
        try:
            from google.cloud.bigquery_storage import BigQueryReadClient
            self.storage_client_available = True
        except ImportError:
            logger.warning("BigQuery Storage API not available. Falling back to standard API.")
            self.storage_client_available = False
```

### Prepare Data Method

```python
def prepare_data(
    self,
    table_id: str,
    target_column: Optional[str] = None,
    force_categorical_columns: List[str] = None,
    drop_columns: List[str] = None,
    high_null_threshold: float = 0.9,
    limit: Optional[int] = None,
    index_col: Optional[str] = None,
    use_streaming: bool = True
) -> Tuple:
    """
    Prepares data from a BigQuery table with optimizations for terabyte-scale.
    
    Args:
        table_id: BigQuery table ID (dataset.table)
        target_column: Target variable name. Heuristics if None.
        force_categorical_columns: Always treat as categorical
        drop_columns: Columns to drop
        high_null_threshold: Drop columns with > this % nulls (after encoding)
        limit: Optional row limit for testing
        index_col: Optional index column
        use_streaming: Whether to use streaming processing (recommended for large datasets)
        
    Returns:
        Tuple: (train_df, val_df, test_df, train_features, val_features, test_features, scaler, imputer)
    """
    # Clean up session and memory
    gc.collect()
    
    # Initialize parameters
    if force_categorical_columns is None:
        force_categorical_columns = []
    if drop_columns is None:
        drop_columns = []
    
    # Define a function to process each batch
    def process_batch(batch):
        # Convert Arrow batch to pandas if needed
        if not isinstance(batch, pd.DataFrame):
            if hasattr(batch, 'to_pandas'):
                batch = batch.to_pandas()
            else:
                batch = pd.DataFrame(batch)
        return batch
    
    # Process data
    logger.info(f"Starting data preparation for table {table_id}")
    
    # Choose processing method based on data size and availability
    if use_streaming and self.storage_client_available:
        # Use streaming for large datasets
        logger.info("Using BigQuery Storage API for streaming processing")
        
        # Create streaming generator
        stream_generator = self.process_bigquery_streaming(
            table_id=table_id,
            processing_fn=process_batch,
            selected_fields=None,  # All fields
            where_clause=None if not limit else f"RAND() < {limit / self.get_table_row_count(table_id)}"
        )
        
        # Aggregate results with memory-efficient method
        df = self.aggregate_streaming_results(stream_generator, aggregation_type='concat')
    else:
        # Fall back to chunked processing for smaller datasets or when Storage API is not available
        logger.info("Using chunked processing")
        
        # If limit is specified, adjust chunk size
        if limit:
            max_chunks = (limit + self.chunk_size - 1) // self.chunk_size
            logger.info(f"Limiting to {limit} rows ({max_chunks} chunks)")
        else:
            max_chunks = None
        
        # Read data in chunks
        df = self.process_bigquery_in_chunks(
            table_id=table_id,
            processing_fn=process_batch,
            max_chunks=max_chunks
        )
    
    # Continue with existing processing logic...
    # (Column type detection, splitting, feature engineering, etc.)
```

## Conclusion

This implementation guide provides a detailed roadmap for improving BigQuery data handling in the `terabyte_feature_extractor.py` file. By implementing true streaming processing with the BigQuery Storage API and Apache Arrow, we can efficiently handle terabyte-scale datasets without running into memory constraints.

The key benefits of this approach include:

1. **Scalability**: Process datasets of any size without memory limitations
2. **Performance**: Faster data access through optimized API
3. **Efficiency**: Memory-efficient processing with Arrow
4. **Flexibility**: Support for both streaming and chunked processing

This implementation is a critical part of the emberharmony purification plan, enabling the library to handle truly large-scale data processing tasks efficiently.