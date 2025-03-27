"""
Animated feature processor with visualization capabilities for BigQuery data.

This module provides feature processing with animation and sample tables
for different data types. It's designed to work with the EmberML backend
abstraction system for optimal performance across different hardware.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import pandas as pd

from ember_ml.nn import tensor
from ember_ml import ops

# Set up logging
logger = logging.getLogger(__name__)


class AnimatedFeatureProcessor:
    """
    Feature processor with animated visualization and sample data tables.
    
    This class processes different types of features (numeric, categorical,
    datetime, etc.) with support for animations and sample tables. It uses
    EmberML's backend-agnostic operations to ensure compatibility across
    different compute environments.
    """
    
    def __init__(
        self,
        visualization_enabled: bool = True,
        sample_tables_enabled: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize the animated feature processor.
        
        Args:
            visualization_enabled: Whether to generate visualizations
            sample_tables_enabled: Whether to generate sample data tables
            device: Optional device to place tensors on
        """
        self.visualization_enabled = visualization_enabled
        self.sample_tables_enabled = sample_tables_enabled
        self.device = device
        
        # Storage for animation frames
        self.processing_frames = []
        
        # Storage for sample tables
        self.sample_tables = {}
    
    def process_numeric_features(
        self,
        df: Any,
        columns: List[str],
        with_imputation: bool = True,
        with_outlier_handling: bool = True,
        with_normalization: bool = True
    ) -> Any:
        """
        Process numeric features with animation and sample tables.
        
        Args:
            df: Input DataFrame
            columns: Numeric columns to process
            with_imputation: Whether to perform missing value imputation
            with_outlier_handling: Whether to handle outliers
            with_normalization: Whether to normalize features
            
        Returns:
            Processed features tensor
        """
        if not columns:
            logger.warning("No numeric columns to process")
            return tensor.zeros((len(df), 0), device=self.device)
        
        # Initialize processing frames for animation
        self.processing_frames = []
        
        # Create a sample data table for original values
        if self.sample_tables_enabled:
            self._create_sample_table(df, columns, 'original_numeric', 
                                      'Original Numeric Data')
        
        # Convert to tensor for consistent processing
        data = self._dataframe_to_tensor(df[columns])
        
        # Capture initial state for animation
        if self.visualization_enabled:
            self._capture_frame(data, "Initial", "Numeric Features")
        
        # Step 1: Handle missing values
        if with_imputation:
            data_with_imputed_values = self._handle_missing_values(data)
            if self.visualization_enabled:
                self._capture_frame(data_with_imputed_values, 
                                    "Missing Value Imputation", "Numeric Features")
                
            # Create a sample data table for imputed values
            if self.sample_tables_enabled:
                self._create_sample_table_from_tensor(
                    data_with_imputed_values, columns, 'imputed_numeric',
                    'After Missing Value Imputation'
                )
        else:
            data_with_imputed_values = data
        
        # Step 2: Handle outliers using robust methods
        if with_outlier_handling:
            data_without_outliers = self._handle_outliers(data_with_imputed_values)
            if self.visualization_enabled:
                self._capture_frame(data_without_outliers, 
                                    "Outlier Removal", "Numeric Features")
            
            # Create a sample data table for outlier-handled values
            if self.sample_tables_enabled:
                self._create_sample_table_from_tensor(
                    data_without_outliers, columns, 'outlier_handled_numeric',
                    'After Outlier Handling'
                )
        else:
            data_without_outliers = data_with_imputed_values
        
        # Step 3: Normalize features to [0,1] range
        if with_normalization:
            normalized_data = self._normalize_robust(data_without_outliers)
            if self.visualization_enabled:
                self._capture_frame(normalized_data, 
                                    "Robust Normalization", "Numeric Features")
            
            # Create a sample data table for normalized values
            if self.sample_tables_enabled:
                self._create_sample_table_from_tensor(
                    normalized_data, columns, 'normalized_numeric',
                    'After Robust Normalization'
                )
        else:
            normalized_data = data_without_outliers
        
        # Generate animation if enabled
        if self.visualization_enabled:
            self._generate_processing_animation()
        
        return normalized_data
    
    def process_categorical_features(
        self,
        df: Any,
        columns: List[str],
        encoding: str = 'one_hot',
        max_categories_per_column: int = 100
    ) -> Any:
        """
        Process categorical features with animation and sample tables.
        
        Args:
            df: Input DataFrame
            columns: Categorical columns to process
            encoding: Encoding method ('one_hot', 'target', 'hash')
            max_categories_per_column: Maximum categories per column for one-hot
            
        Returns:
            Processed features tensor
        """
        if not columns:
            logger.warning("No categorical columns to process")
            return tensor.zeros((len(df), 0), device=self.device)
        
        # Initialize processing frames for animation
        self.processing_frames = []
        
        # Create a sample data table for original values
        if self.sample_tables_enabled:
            self._create_sample_table(df, columns, 'original_categorical', 
                                      'Original Categorical Data')
        
        # Process each column separately
        encoded_features_list = []
        encoded_feature_names = []
        
        for col in columns:
            # Get unique values and create mapping
            unique_values = df[col].dropna().unique()
            
            # Skip if too many categories for one-hot encoding
            if encoding == 'one_hot' and len(unique_values) > max_categories_per_column:
                logger.warning(f"Column {col} has {len(unique_values)} unique values, "
                               f"which exceeds the maximum of {max_categories_per_column}. "
                               f"Switching to hash encoding.")
                encoding = 'hash'
            
            # Apply encoding
            if encoding == 'one_hot':
                encoded_features, feature_names = self._one_hot_encode(df, col, unique_values)
            elif encoding == 'target':
                # For now, we'll use one-hot as fallback since target encoding
                # requires a target variable
                encoded_features, feature_names = self._one_hot_encode(df, col, unique_values)
            elif encoding == 'hash':
                encoded_features, feature_names = self._hash_encode(df, col, n_components=16)
            else:
                raise ValueError(f"Unsupported encoding method: {encoding}")
            
            encoded_features_list.append(encoded_features)
            encoded_feature_names.extend(feature_names)
        
        # Combine all encoded features
        if encoded_features_list:
            encoded_data = tensor.concatenate(encoded_features_list, axis=1)
            
            # Create a sample data table for encoded values
            if self.sample_tables_enabled:
                self._create_sample_table_from_tensor(
                    encoded_data, encoded_feature_names, 'encoded_categorical',
                    f'After {encoding.capitalize()} Encoding'
                )
                
            # Generate animation if enabled
            if self.visualization_enabled:
                self._generate_processing_animation()
                
            return encoded_data
        else:
            return tensor.zeros((len(df), 0), device=self.device)
    
    def process_datetime_features(
        self,
        df: Any,
        columns: List[str],
        cyclical_encoding: bool = True,
        include_time: bool = True,
        include_date_parts: bool = True
    ) -> Any:
        """
        Process datetime features with animation and sample tables.
        
        This method extracts meaningful features from datetime columns, including:
        - Cyclical encoding of hour, day, month components
        - Numerical representations of date/time parts
        - Derived features like day of week, quarter, etc.
        
        Args:
            df: Input DataFrame
            columns: Datetime columns to process
            cyclical_encoding: Whether to use cyclical encoding for cyclic features
            include_time: Whether to extract time components (hour, minute, second)
            include_date_parts: Whether to extract date parts (year, month, day)
            
        Returns:
            Processed features tensor
        """
        if not columns:
            logger.warning("No datetime columns to process")
            return tensor.zeros((len(df), 0), device=self.device)
        
        # Initialize processing frames for animation
        self.processing_frames = []
        
        # Create a sample data table for original values
        if self.sample_tables_enabled:
            self._create_sample_table(df, columns, 'original_datetime', 
                                      'Original Datetime Data')
        
        # Process each datetime column
        all_features = []
        all_feature_names = []
        
        for col in columns:
            # Extract date/time components
            try:
                # For pandas-like DataFrames
                components = self._extract_datetime_components(
                    df[col], include_time, include_date_parts
                )
                
                # Process each component
                processed_components = []
                component_names = []
                
                for comp_name, comp_data in components.items():
                    # Convert to tensor
                    comp_tensor = tensor.convert_to_tensor(comp_data, device=self.device)
                    
                    # Apply cyclical encoding for cyclic features if requested
                    if cyclical_encoding and comp_name in ['hour', 'day', 'month', 'day_of_week']:
                        # Get the maximum value for this cycle
                        if comp_name == 'hour':
                            max_val = 24
                        elif comp_name == 'day':
                            max_val = 31
                        elif comp_name == 'month':
                            max_val = 12
                        elif comp_name == 'day_of_week':
                            max_val = 7
                        else:
                            max_val = ops.max(comp_tensor).item() + 1
                        
                        # Apply sin/cos encoding
                        sin_comp, cos_comp = self._cyclical_encode(comp_tensor, max_val)
                        
                        # Add to processed components
                        processed_components.append(sin_comp)
                        processed_components.append(cos_comp)
                        component_names.append(f"{col}_{comp_name}_sin")
                        component_names.append(f"{col}_{comp_name}_cos")
                    else:
                        # For non-cyclic features, normalize to [0,1]
                        normalized_comp = self._normalize_component(comp_tensor)
                        processed_components.append(normalized_comp)
                        component_names.append(f"{col}_{comp_name}")
                
                # Combine all components for this column
                if processed_components:
                    # Reshape components to 2D
                    reshaped_components = []
                    for comp in processed_components:
                        reshaped_comp = ops.reshape(comp, (tensor.shape(comp)[0], 1))
                        reshaped_components.append(reshaped_comp)
                    
                    # Concatenate along columns
                    column_features = ops.concatenate(reshaped_components, axis=1)
                    
                    # Add to all features
                    all_features.append(column_features)
                    all_feature_names.extend(component_names)
                    
                    # Create sample data table for this column's components
                    if self.sample_tables_enabled:
                        self._create_sample_table_from_tensor(
                            column_features, component_names, 
                            f'datetime_{col}_components',
                            f'Datetime Components for {col}'
                        )
                    
                    # Capture animation frame
                    if self.visualization_enabled:
                        self._capture_frame(
                            column_features, 
                            f"Datetime Processing for {col}", 
                            "Datetime Features"
                        )
            except Exception as e:
                logger.warning(f"Error processing datetime column {col}: {e}")
                continue
        
        # Combine all datetime features
        if all_features:
            combined_features = ops.concatenate(all_features, axis=1)
            
            # Create a sample data table for all processed datetime features
            if self.sample_tables_enabled:
                self._create_sample_table_from_tensor(
                    combined_features, all_feature_names, 
                    'processed_datetime',
                    'Processed Datetime Features'
                )
            
            # Generate animation if enabled
            if self.visualization_enabled:
                self._generate_processing_animation()
            
            return combined_features
        else:
            logger.warning("Failed to extract any datetime features")
            return tensor.zeros((len(df), 0), device=self.device)
    
    def process_identifier_features(
        self,
        df: Any,
        columns: List[str],
        n_components: int = 16,
        seed: int = 42
    ) -> Any:
        """
        Process identifier features with animation and sample tables.
        
        Args:
            df: Input DataFrame
            columns: Identifier columns to process
            n_components: Number of hash components
            seed: Random seed
            
        Returns:
            Processed features tensor
        """
        if not columns:
            logger.warning("No identifier columns to process")
            return tensor.zeros((len(df), 0), device=self.device)
        
        # Initialize processing frames for animation
        self.processing_frames = []
        
        # Create a sample data table for original values
        if self.sample_tables_enabled:
            self._create_sample_table(df, columns, 'original_identifier', 
                                      'Original Identifier Data')
        
        # Process each column using hash encoding
        encoded_features_list = []
        encoded_feature_names = []
        
        for col in columns:
            # Apply hash encoding
            encoded_features, feature_names = self._hash_encode(
                df, col, n_components=n_components, seed=seed
            )
            
            encoded_features_list.append(encoded_features)
            encoded_feature_names.extend(feature_names)
            
            # Capture animation frame
            if self.visualization_enabled:
                self._capture_frame(
                    encoded_features, 
                    f"Hash Encoding for {col}", 
                    "Identifier Features"
                )
        
        # Combine all encoded features
        if encoded_features_list:
            encoded_data = tensor.concatenate(encoded_features_list, axis=1)
            
            # Create a sample data table for encoded values
            if self.sample_tables_enabled:
                self._create_sample_table_from_tensor(
                    encoded_data, encoded_feature_names, 'encoded_identifier',
                    'Hash-Encoded Identifier Features'
                )
                
            # Generate animation if enabled
            if self.visualization_enabled:
                self._generate_processing_animation()
                
            return encoded_data
        else:
            return tensor.zeros((len(df), 0), device=self.device)
    
    def _dataframe_to_tensor(self, df: Any) -> Any:
        """
        Convert a DataFrame to a tensor.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tensor representation of DataFrame
        """
        try:
            # For pandas-like DataFrames
            data = df.values
        except AttributeError:
            # For other DataFrame-like objects
            data = df.to_numpy()
        
        # Convert to tensor
        return tensor.convert_to_tensor(data, device=self.device)
    
    def _handle_missing_values(self, data: Any) -> Any:
        """
        Handle missing values in numeric data.
        
        Args:
            data: Input tensor
            
        Returns:
            Tensor with missing values handled
        """
        # Check for NaN values
        nan_mask = ops.isnan(data)
        
        # If no NaNs, return the original data
        if not ops.any(nan_mask):
            return data
        
        # Calculate median for each feature
        # First, replace NaNs with zeros for calculation purposes
        data_no_nan = ops.where(nan_mask, ops.zeros_like(data), data)
        
        # Calculate median for each feature (column)
        medians = []
        for i in range(tensor.shape(data)[1]):
            col_data = data_no_nan[:, i]
            # Sort the data
            sorted_data = ops.sort(col_data)
            # Get median
            n = tensor.shape(sorted_data)[0]
            if n % 2 == 0:
                median = ops.divide(
                    ops.add(sorted_data[n // 2 - 1], sorted_data[n // 2]),
                    tensor.convert_to_tensor(2.0)
                )
            else:
                median = sorted_data[n // 2]
            medians.append(median)
        
        # Convert to tensor
        medians_tensor = ops.stack(medians)
        
        # Reshape medians to match data shape for broadcasting
        medians_tensor = ops.reshape(medians_tensor, (1, -1))
        
        # Replace NaNs with medians
        return ops.where(nan_mask, ops.broadcast_to(medians_tensor, tensor.shape(data)), data)
    
    def _handle_outliers(self, data: Any) -> Any:
        """
        Handle outliers in numeric data using IQR method.
        
        Args:
            data: Input tensor
            
        Returns:
            Tensor with outliers handled
        """
        # Calculate quartiles for each feature
        q1_values = []
        q3_values = []
        
        for i in range(tensor.shape(data)[1]):
            col_data = data[:, i]
            # Sort the data
            sorted_data = ops.sort(col_data)
            n = tensor.shape(sorted_data)[0]
            
            # Calculate quartile indices
            q1_idx = n // 4
            q3_idx = (3 * n) // 4
            
            # Get quartile values
            q1 = sorted_data[q1_idx]
            q3 = sorted_data[q3_idx]
            
            q1_values.append(q1)
            q3_values.append(q3)
        
        # Convert to tensors
        q1_tensor = ops.stack(q1_values)
        q3_tensor = ops.stack(q3_values)
        
        # Calculate IQR
        iqr_tensor = ops.subtract(q3_tensor, q1_tensor)
        
        # Calculate bounds
        lower_bound = ops.subtract(q1_tensor, ops.multiply(
            iqr_tensor, tensor.convert_to_tensor(1.5)
        ))
        upper_bound = ops.add(q3_tensor, ops.multiply(
            iqr_tensor, tensor.convert_to_tensor(1.5)
        ))
        
        # Reshape bounds to match data shape for broadcasting
        lower_bound = ops.reshape(lower_bound, (1, -1))
        upper_bound = ops.reshape(upper_bound, (1, -1))
        
        # Clip values to bounds
        return ops.clip(data, lower_bound, upper_bound)
    
    def _normalize_robust(self, data: Any) -> Any:
        """
        Apply robust normalization to numeric data using quantiles.
        
        Args:
            data: Input tensor
            
        Returns:
            Normalized tensor
        """
        # Calculate min and max for each feature using 5th and 95th percentiles
        min_values = []
        max_values = []
        
        for i in range(tensor.shape(data)[1]):
            col_data = data[:, i]
            # Sort the data
            sorted_data = ops.sort(col_data)
            n = tensor.shape(sorted_data)[0]
            
            # Calculate percentile indices
            p05_idx = max(0, int(0.05 * n))
            p95_idx = min(n - 1, int(0.95 * n))
            
            # Get percentile values
            p05 = sorted_data[p05_idx]
            p95 = sorted_data[p95_idx]
            
            min_values.append(p05)
            max_values.append(p95)
        
        # Convert to tensors
        min_tensor = ops.stack(min_values)
        max_tensor = ops.stack(max_values)
        
        # Reshape min and max to match data shape for broadcasting
        min_tensor = ops.reshape(min_tensor, (1, -1))
        max_tensor = ops.reshape(max_tensor, (1, -1))
        
        # Calculate range
        range_tensor = ops.subtract(max_tensor, min_tensor)
        
        # Avoid division by zero
        epsilon = tensor.convert_to_tensor(1e-8, device=self.device)
        range_tensor = ops.maximum(range_tensor, epsilon)
        
        # Normalize data
        normalized_data = ops.divide(
            ops.subtract(data, min_tensor),
            range_tensor
        )
        
        # Clip to ensure values are in [0, 1]
        return ops.clip(normalized_data, 0, 1)
    
    def _one_hot_encode(
        self,
        df: Any,
        column: str,
        unique_values: Any
    ) -> Tuple[Any, List[str]]:
        """
        One-hot encode a categorical column.
        
        Args:
            df: Input DataFrame
            column: Column to encode
            unique_values: Unique values in the column
            
        Returns:
            Tuple of (encoded_tensor, feature_names)
        """
        # Get column data
        col_data = df[column]
        
        # Initialize encoded data
        n_samples = len(df)
        n_categories = len(unique_values)
        encoded_data = tensor.zeros((n_samples, n_categories), device=self.device)
        
        # Generate feature names
        feature_names = []
        value_to_index = {}
        
        for i, value in enumerate(unique_values):
            # Create safe feature name
            if isinstance(value, str):
                safe_value = value.replace(' ', '_').replace('-', '_').replace('/', '_')
            else:
                safe_value = str(value)
            
            feature_name = f"{column}_{safe_value}"
            feature_names.append(feature_name)
            value_to_index[value] = i
        
        # Encode each row
        for i, value in enumerate(col_data):
            if pd.isna(value):
                # Skip NaN values
                continue
                
            if value in value_to_index:
                idx = value_to_index[value]
                # Set the corresponding column to 1
                encoded_data = tensor.tensor_scatter_nd_update(
                    encoded_data,
                    tensor.reshape(tensor.convert_to_tensor([[i, idx]]), (1, 2)),
                    tensor.reshape(tensor.convert_to_tensor([1.0]), (1,))
                )
        
        return encoded_data, feature_names
    
    def _hash_encode(
        self,
        df: Any,
        column: str,
        n_components: int = 16,
        seed: int = 42
    ) -> Tuple[Any, List[str]]:
        """
        Hash encode a high-cardinality categorical or identifier column.
        
        Args:
            df: Input DataFrame
            column: Column to encode
            n_components: Number of hash components
            seed: Random seed
            
        Returns:
            Tuple of (encoded_tensor, feature_names)
        """
        # Get column data
        col_data = df[column]
        
        # Initialize encoded data
        n_samples = len(df)
        encoded_data = tensor.zeros((n_samples, n_components), device=self.device)
        
        # Generate feature names
        feature_names = [f"{column}_hash_{i}" for i in range(n_components)]
        
        # Set random seed
        tensor.set_seed(seed)
        
        # Encode each value
        for i, value in enumerate(col_data):
            if pd.isna(value):
                # Skip NaN values
                continue
                
            # Convert value to string
            str_value = str(value)
            
            # Generate hash value
            hash_val = hash(str_value + str(seed))
            
            # Use hash to generate pseudo-random values for each component
            for j in range(n_components):
                component_hash = hash(str_value + str(seed + j))
                # Scale to [0, 1]
                component_value = (component_hash % 10000) / 10000.0
                
                # Update the tensor at position [i, j]
                encoded_data = tensor.tensor_scatter_nd_update(
                    encoded_data,
                    tensor.reshape(tensor.convert_to_tensor([[i, j]]), (1, 2)),
                    tensor.reshape(tensor.convert_to_tensor([component_value]), (1,))
                )
        
        # Reset seed
        tensor.set_seed(None)
        
        return encoded_data, feature_names
    
    def _extract_datetime_components(
        self,
        datetime_col: Any,
        include_time: bool = True,
        include_date_parts: bool = True
    ) -> Dict[str, List[Any]]:
        """
        Extract components from a datetime column.
        
        Args:
            datetime_col: Datetime column
            include_time: Whether to include time components
            include_date_parts: Whether to include date parts
            
        Returns:
            Dictionary of component name to values
        """
        components = {}
        
        # Extract date parts
        if include_date_parts:
            components['year'] = datetime_col.dt.year.tolist()
            components['month'] = datetime_col.dt.month.tolist()
            components['day'] = datetime_col.dt.day.tolist()
            components['day_of_week'] = datetime_col.dt.dayofweek.tolist()
            components['day_of_year'] = datetime_col.dt.dayofyear.tolist()
            components['quarter'] = datetime_col.dt.quarter.tolist()
            components['is_month_end'] = datetime_col.dt.is_month_end.astype(int).tolist()
            components['is_month_start'] = datetime_col.dt.is_month_start.astype(int).tolist()
            components['is_quarter_end'] = datetime_col.dt.is_quarter_end.astype(int).tolist()
            components['is_quarter_start'] = datetime_col.dt.is_quarter_start.astype(int).tolist()
            components['is_year_end'] = datetime_col.dt.is_year_end.astype(int).tolist()
            components['is_year_start'] = datetime_col.dt.is_year_start.astype(int).tolist()
        
        # Extract time parts
        if include_time:
            components['hour'] = datetime_col.dt.hour.tolist()
            components['minute'] = datetime_col.dt.minute.tolist()
            components['second'] = datetime_col.dt.second.tolist()
            
            # Add derived time features
            hour_of_day = datetime_col.dt.hour
            components['is_morning'] = (hour_of_day >= 6) & (hour_of_day < 12)
            components['is_afternoon'] = (hour_of_day >= 12) & (hour_of_day < 18)
            components['is_evening'] = (hour_of_day >= 18) & (hour_of_day < 22)
            components['is_night'] = (hour_of_day >= 22) | (hour_of_day < 6)
            
            # Convert boolean to int
            for comp in ['is_morning', 'is_afternoon', 'is_evening', 'is_night']:
                components[comp] = components[comp].astype(int).tolist()
        
        return components
    
    def _cyclical_encode(self, data: Any, period: float) -> Tuple[Any, Any]:
        """
        Apply cyclical encoding using sine and cosine transformations.
        
        Args:
            data: Input tensor
            period: Period of the cycle
            
        Returns:
            Tuple of (sin_component, cos_component)
        """
        # Scale to [0, 2π]
        scaled_data = ops.multiply(
            ops.divide(data, tensor.convert_to_tensor(period, device=self.device)),
            tensor.convert_to_tensor(2.0 * 3.14159, device=self.device)
        )
        
        # Apply sin and cos transformations
        sin_component = ops.sin(scaled_data)
        cos_component = ops.cos(scaled_data)
        
        return sin_component, cos_component
    
    def _normalize_component(self, data: Any) -> Any:
        """
        Normalize a datetime component to [0,1] range.
        
        Args:
            data: Input tensor
            
        Returns:
            Normalized tensor
        """
        # Get min and max values
        min_val = ops.min(data)
        max_val = ops.max(data)
        
        # Calculate range
        data_range = ops.subtract(max_val, min_val)
        
        # Avoid division by zero
        epsilon = tensor.convert_to_tensor(1e-8, device=self.device)
        data_range = ops.maximum(data_range, epsilon)
        
        # Normalize
        return ops.divide(
            ops.subtract(data, min_val),
            data_range
        )
    
    def _create_sample_table(
        self, 
        df: Any, 
        columns: List[str], 
        table_id: str, 
        title: str,
        max_rows: int = 5
    ) -> None:
        """
        Create a sample data table from DataFrame columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to include in the table
            table_id: Unique identifier for the table
            title: Title of the table
