import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union
import os

logger = logging.getLogger(__name__)

class DataConnector:
    """Base class for all data connectors"""
    def __init__(self):
        pass
        
    def extract_data(self, source: str) -> pd.DataFrame:
        """Extract data from a source"""
        raise NotImplementedError("Subclasses must implement this method")
        
class FileConnector(DataConnector):
    """Base class for file-based connectors"""
    def __init__(self):
        super().__init__()
        
    def extract_data(self, source: str) -> pd.DataFrame:
        """Extract data from a file"""
        if not os.path.exists(source):
            raise FileNotFoundError(f"File {source} not found")
        
        _, file_extension = os.path.splitext(source)
        file_extension = file_extension.lower()[1:]  # Remove leading dot
        
        if file_extension == 'csv':
            return self._extract_csv(source)
        elif file_extension == 'json':
            return self._extract_json(source)
        elif file_extension in ('xls', 'xlsx'):
            return self._extract_excel(source)
        elif file_extension == 'parquet':
            return self._extract_parquet(source)
        elif file_extension == 'txt':
            return self._extract_text(source)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def _extract_csv(self, file_path: str) -> pd.DataFrame:
        """Extract data from a CSV file"""
        logger.info(f"Loading CSV file: {file_path}")
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}", exc_info=True)
            raise
    
    def _extract_json(self, file_path: str) -> pd.DataFrame:
        """Extract data from a JSON file"""
        logger.info(f"Loading JSON file: {file_path}")
        try:
            return pd.read_json(file_path)
        except Exception as e:
            logger.error(f"Error loading JSON file: {str(e)}", exc_info=True)
            raise
    
    def _extract_excel(self, file_path: str) -> pd.DataFrame:
        """Extract data from an Excel file"""
        logger.info(f"Loading Excel file: {file_path}")
        try:
            # This requires the openpyxl package for xlsx files
            return pd.read_excel(file_path)
        except Exception as e:
            logger.error(f"Error loading Excel file: {str(e)}", exc_info=True)
            raise
    
    def _extract_parquet(self, file_path: str) -> pd.DataFrame:
        """Extract data from a Parquet file"""
        logger.info(f"Loading Parquet file: {file_path}")
        try:
            # This requires the pyarrow or fastparquet package
            return pd.read_parquet(file_path)
        except Exception as e:
            logger.error(f"Error loading Parquet file: {str(e)}", exc_info=True)
            raise
    
    def _extract_text(self, file_path: str) -> pd.DataFrame:
        """Extract data from a text file"""
        logger.info(f"Loading text file: {file_path}")
        try:
            # Try to infer delimiter for text files
            return pd.read_csv(file_path, sep=None, engine='python')
        except Exception as e:
            logger.error(f"Error loading text file: {str(e)}", exc_info=True)
            # If delimiter inference fails, return as single text column
            with open(file_path, 'r') as f:
                content = f.readlines()
            return pd.DataFrame({'text': content})
            
class DatabaseConnector(DataConnector):
    """Base class for database connectors"""
    def __init__(self, connection_params: Dict[str, Any]):
        super().__init__()
        self.connection_params = connection_params
        
    def extract_data(self, query: str) -> pd.DataFrame:
        """Extract data using a SQL query"""
        logger.info(f"Executing query: {query[:100]}...")
        # This is a stub - would be implemented with actual database connection
        logger.warning("Database connector is currently stubbed")
        return pd.DataFrame()  # Empty DataFrame as placeholder
        
class APIConnector(DataConnector):
    """Base class for API connectors"""
    def __init__(self, api_credentials: Dict[str, Any]):
        super().__init__()
        self.api_credentials = api_credentials
        
    def extract_data(self, endpoint: str) -> pd.DataFrame:
        """Extract data from an API endpoint"""
        logger.info(f"Fetching data from API endpoint: {endpoint}")
        # This is a stub - would be implemented with actual API calls
        logger.warning("API connector is currently stubbed")
        return pd.DataFrame()  # Empty DataFrame as placeholder
        
def get_connector(source_type: str, **kwargs) -> DataConnector:
    """Factory function to get the appropriate connector based on source type"""
    if source_type in ['csv', 'json', 'excel', 'parquet', 'txt', 'file']:
        return FileConnector()
    elif source_type == 'database':
        if 'connection_params' not in kwargs:
            raise ValueError("Database connector requires connection_params")
        return DatabaseConnector(kwargs['connection_params'])
    elif source_type == 'api':
        if 'api_credentials' not in kwargs:
            raise ValueError("API connector requires api_credentials")
        return APIConnector(kwargs['api_credentials'])
    else:
        raise ValueError(f"Unsupported source type: {source_type}")