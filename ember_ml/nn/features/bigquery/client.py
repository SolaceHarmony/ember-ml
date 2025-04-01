"""
BigQuery client utilities for Ember ML.

This module provides functions for connecting to and querying Google BigQuery.
"""

import logging
from typing import Dict, Optional, Any

import pandas as pd

logger = logging.getLogger(__name__)


def initialize_client(
    project_id: str,
    credentials_path: Optional[str] = None
) -> Any:
    """
    Initialize a BigQuery client.
    
    Args:
        project_id: Google Cloud project ID
        credentials_path: Path to Google Cloud credentials file
        
    Returns:
        BigQuery client object
        
    Raises:
        ImportError: If Google Cloud libraries are not installed
        ConnectionError: If connection to BigQuery fails
    """
    try:
        from google.cloud import bigquery
        from google.oauth2 import service_account
        
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            client = bigquery.Client(
                credentials=credentials,
                project=project_id
            )
        else:
            # Use default credentials
            client = bigquery.Client(project=project_id)
            
        logger.info(f"Connected to BigQuery project: {project_id}")
        
        return client
        
    except ImportError:
        raise ImportError(
            "Google Cloud BigQuery libraries not installed. "
            "Please install with: pip install google-cloud-bigquery"
        )
    except Exception as e:
        raise ConnectionError(f"Failed to connect to BigQuery: {str(e)}")


def execute_query(client: Any, query: str) -> pd.DataFrame:
    """
    Execute a SQL query on BigQuery and return results as a DataFrame.
    
    Args:
        client: BigQuery client object
        query: SQL query string to execute
        
    Returns:
        DataFrame containing query results
        
    Raises:
        RuntimeError: If query execution fails
    """
    try:
        # Execute the query
        query_job = client.query(query)
        
        # Wait for query to complete and fetch results
        results = query_job.result()
        
        # Convert to DataFrame
        df = results.to_dataframe()
        
        logger.info(f"Query executed successfully. Rows returned: {len(df)}")
        
        return df
        
    except Exception as e:
        logger.error(f"Query execution failed: {str(e)}")
        raise RuntimeError(f"Query execution failed: {str(e)}")


def fetch_table_schema(
    client: Any,
    dataset_id: str,
    table_id: str
) -> Dict[str, str]:
    """
    Fetch the schema of a BigQuery table.
    
    Args:
        client: BigQuery client object
        dataset_id: BigQuery dataset ID
        table_id: BigQuery table ID
        
    Returns:
        Dictionary mapping column names to their data types
        
    Raises:
        RuntimeError: If schema fetch fails
    """
    try:
        # Get table reference
        table_ref = client.dataset(dataset_id).table(table_id)
        
        # Get table
        table = client.get_table(table_ref)
        
        # Extract schema
        schema = {field.name: field.field_type for field in table.schema}
        
        logger.info(f"Schema fetched successfully for {dataset_id}.{table_id}")
        
        return schema
        
    except Exception as e:
        logger.error(f"Failed to fetch table schema: {str(e)}")
        raise RuntimeError(f"Failed to fetch table schema: {str(e)}")