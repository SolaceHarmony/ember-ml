"""Simplified BigQuery feature extractor."""

from typing import List, Optional, Tuple, Any

import pandas as pd

from ember_ml import tensor
from ember_ml.features import bigquery
from ember_ml.features.column_feature_extraction import ColumnFeatureExtractor


class BigQueryFeatureExtractor(ColumnFeatureExtractor):
    """Feature extractor for Google BigQuery tables."""

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        table_id: str,
        credentials_path: Optional[str] = None,
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        datetime_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.credentials_path = credentials_path
        self.numeric_columns = numeric_columns or []
        self.categorical_columns = categorical_columns or []
        self.datetime_columns = datetime_columns or []
        self.target_column = target_column
        self.device = device
        self._client = None

    # --- BigQuery helpers ---
    def initialize_client(self) -> Any:
        self._client = bigquery.initialize_client(self.project_id, self.credentials_path)
        return self._client

    def execute_query(self, query: str) -> pd.DataFrame:
        if self._client is None:
            self.initialize_client()
        return bigquery.execute_query(self._client, query)

    def fetch_table_schema(self) -> dict:
        if self._client is None:
            self.initialize_client()
        return bigquery.fetch_table_schema(self._client, self.dataset_id, self.table_id)

    # --- Column management ---
    def auto_detect_column_types(self) -> None:
        schema = self.fetch_table_schema()
        for name, col_type in schema.items():
            if name == self.target_column:
                continue
            if col_type in {"FLOAT", "INTEGER", "NUMERIC", "BIGNUMERIC"}:
                self.numeric_columns.append(name)
            elif col_type in {"DATETIME", "DATE", "TIMESTAMP"}:
                self.datetime_columns.append(name)
            else:
                self.categorical_columns.append(name)
        self.numeric_columns = sorted(set(self.numeric_columns))
        self.categorical_columns = sorted(set(self.categorical_columns))
        self.datetime_columns = sorted(set(self.datetime_columns))

    # --- Data fetching ---
    def fetch_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        query = f"SELECT * FROM `{self.dataset_id}.{self.table_id}`"
        if limit is not None:
            query += f" LIMIT {limit}"
        return self.execute_query(query)

    # --- Feature extraction ---
    def extract_features(
        self,
        data: Optional[pd.DataFrame] = None,
        limit: Optional[int] = None,
        handle_missing: bool = True,
        handle_outliers: bool = True,
        normalize: bool = True,
    ) -> Tuple[tensor.EmberTensor, List[str]]:
        if data is None:
            data = self.fetch_data(limit=limit)
        numeric_tensor, numeric_names = bigquery.process_numeric_features(
            data,
            self.numeric_columns,
            handle_missing=handle_missing,
            handle_outliers=handle_outliers,
            normalize=normalize,
            device=self.device,
        ) if self.numeric_columns else (None, [])
        categorical_tensor, categorical_names = bigquery.process_categorical_features(
            data,
            self.categorical_columns,
            handle_missing=handle_missing,
            device=self.device,
        ) if self.categorical_columns else (None, [])
        datetime_tensor, datetime_names = bigquery.process_datetime_features(
            data,
            self.datetime_columns,
            device=self.device,
        ) if self.datetime_columns else (None, [])

        tensors = [t for t in [numeric_tensor, categorical_tensor, datetime_tensor] if t is not None]
        names = numeric_names + categorical_names + datetime_names
        if not tensors:
            raise ValueError("No features to extract")
        features = tensor.concatenate(tensors, axis=1) if len(tensors) > 1 else tensors[0]
        return features, names
