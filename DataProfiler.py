import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union
import os
from DataConnector import get_connector

logger = logging.getLogger(__name__)

class DataProfiler:
    """Class to profile dataset characteristics and extract meaningful insights"""
    
    def __init__(self):
        self.data = None
        self.profile = {}
        logger.info("DataProfiler initialized")
    
    def load_data(self, data_source: str, data_type: str = "csv") -> pd.DataFrame:
        """Load data from various sources"""
        logger.info(f"Loading data from {data_source} of type {data_type}")
        
        try:
            connector = get_connector(data_type)
            self.data = connector.extract_data(data_source)
            
            # Basic validation
            if self.data is None or len(self.data) == 0:
                raise ValueError("No data was loaded")
                
            logger.info(f"Successfully loaded data with shape {self.data.shape}")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}", exc_info=True)
            raise
    
    def generate_profile(self) -> Dict[str, Any]:
        """Generate a comprehensive profile of the loaded data"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data first.")
        
        logger.info("Generating data profile")
        try:
            self.profile = {
                "basic_stats": self._get_basic_stats(),
                "column_types": self._detect_column_types(),
                "missing_data": self._analyze_missing_data(),
                "distribution_info": self._analyze_distributions(),
                "potential_relationships": self._detect_relationships(),
                "data_quality_issues": self._detect_quality_issues()
            }
            
            logger.info("Profile generation complete")
            return self.profile
            
        except Exception as e:
            logger.error(f"Error generating profile: {str(e)}", exc_info=True)
            raise
    
    def _get_basic_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the dataset"""
        logger.debug("Getting basic statistics")
        return {
            "row_count": len(self.data),
            "column_count": len(self.data.columns),
            "memory_usage": self.data.memory_usage(deep=True).sum(),
            "columns": list(self.data.columns)
        }
    
    def _detect_column_types(self) -> Dict[str, str]:
        """Detect and categorize column types beyond standard dtypes"""
        logger.debug("Detecting column types")
        column_types = {}
        
        for col in self.data.columns:
            # Start with pandas dtype
            dtype = str(self.data[col].dtype)
            
            # Enhanced type detection
            if dtype == "object":
                if self._is_categorical(self.data[col]):
                    column_types[col] = "categorical"
                elif self._is_datetime(self.data[col]):
                    column_types[col] = "datetime"
                elif self._is_text(self.data[col]):
                    column_types[col] = "text"
                else:
                    column_types[col] = "object"
            elif "int" in dtype or "float" in dtype:
                if self._is_binary(self.data[col]):
                    column_types[col] = "binary"
                elif self._is_categorical(self.data[col]):
                    column_types[col] = "categorical"
                else:
                    column_types[col] = "numeric"
            elif "datetime" in dtype:
                column_types[col] = "datetime"
            else:
                column_types[col] = dtype
                
        return column_types
    
    # Helper methods for type detection
    def _is_categorical(self, series: pd.Series) -> bool:
        """Check if a column appears to be categorical"""
        unique_count = series.nunique()
        return unique_count < min(10, len(series) * 0.05)
    
    def _is_datetime(self, series: pd.Series) -> bool:
        """Check if a column appears to contain datetime values"""
        try:
            pd.to_datetime(series)
            return True
        except:
            return False
            
    def _is_text(self, series: pd.Series) -> bool:
        """Check if a column contains longer text"""
        if series.dtype != "object":
            return False
        # Get non-null values
        values = series.dropna()
        if len(values) == 0:
            return False
        # Check if average length > 100 chars
        return values.astype(str).str.len().mean() > 100
    
    def _is_binary(self, series: pd.Series) -> bool:
        """Check if a column contains only binary values"""
        return series.nunique() == 2
    
    # Additional analysis methods
    def _analyze_missing_data(self) -> Dict[str, Any]:
        """Analyze missing data patterns"""
        logger.debug("Analyzing missing data")
        result = {}
        
        # Basic missing stats
        missing_counts = self.data.isnull().sum()
        missing_percentages = (missing_counts / len(self.data)) * 100
        
        result["missing_counts"] = missing_counts.to_dict()
        result["missing_percentages"] = missing_percentages.to_dict()
        
        # Missing pattern analysis
        if missing_counts.sum() > 0:
            # Check for columns that are missing together
            # This would be more complex in a real implementation
            missing_together = []
            
            # Create missing data heatmap
            missing_matrix = self.data.isnull().astype(int)
            
            # Identify correlated missingness (simplified approach)
            if len(self.data.columns) > 1:
                corr_matrix = missing_matrix.corr()
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if corr_matrix.iloc[i, j] > 0.7:  # Threshold for correlated missingness
                            missing_together.append({
                                "col1": corr_matrix.columns[i],
                                "col2": corr_matrix.columns[j],
                                "correlation": float(corr_matrix.iloc[i, j])
                            })
            
            result["columns_missing_together"] = missing_together
            
        return result
    
    def _analyze_distributions(self) -> Dict[str, Dict[str, Any]]:
        """Analyze numerical and categorical distributions"""
        logger.debug("Analyzing distributions")
        distributions = {}
        
        for col, col_type in self.profile.get("column_types", {}).items():
            if col_type == "numeric":
                # Handle potential inf values
                clean_data = self.data[col].replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(clean_data) > 0:
                    distributions[col] = {
                        "min": float(clean_data.min()),
                        "max": float(clean_data.max()),
                        "mean": float(clean_data.mean()),
                        "median": float(clean_data.median()),
                        "std": float(clean_data.std()),
                        "skew": float(clean_data.skew()),
                        "histogram_data": self._create_histogram(clean_data)
                    }
            elif col_type in ["categorical", "binary"]:
                value_counts = self.data[col].value_counts()
                
                if len(value_counts) > 0:
                    distributions[col] = {
                        "value_counts": value_counts.to_dict(),
                        "top_value": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                        "entropy": self._calculate_entropy(self.data[col])
                    }
                    
        return distributions
    
    def _create_histogram(self, series: pd.Series) -> Dict[str, List[Any]]:
        """Create histogram data for a numerical column"""
        try:
            hist, bin_edges = np.histogram(series.dropna(), bins='auto')
            return {
                "counts": hist.tolist(),
                "bin_edges": bin_edges.tolist()
            }
        except Exception as e:
            logger.warning(f"Could not create histogram: {str(e)}")
            return {
                "counts": [],
                "bin_edges": []
            }
    
    def _calculate_entropy(self, series: pd.Series) -> float:
        """Calculate entropy of a categorical column"""
        try:
            vc = series.value_counts(normalize=True, dropna=True)
            if len(vc) <= 1:
                return 0.0
            return float(-(vc * np.log2(vc)).sum())
        except Exception as e:
            logger.warning(f"Could not calculate entropy: {str(e)}")
            return 0.0
    
    def _detect_relationships(self) -> Dict[str, Any]:
        """Detect potential relationships between columns"""
        logger.debug("Detecting relationships")
        relationships = {}
        
        # Simplified correlation analysis
        numeric_cols = [col for col, type in self.profile.get("column_types", {}).items() 
                      if type == "numeric"]
        
        if len(numeric_cols) > 1:
            try:
                # Use a more robust correlation method
                corr_matrix = self.data[numeric_cols].corr(method='spearman').abs()
                
                # Find highly correlated pairs
                highly_correlated = []
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        if corr_matrix.iloc[i, j] > 0.7:  # Threshold for strong correlation
                            highly_correlated.append({
                                "col1": numeric_cols[i],
                                "col2": numeric_cols[j],
                                "correlation": float(corr_matrix.iloc[i, j])
                            })
                
                relationships["high_correlations"] = highly_correlated
                
                # Add correlation matrix (simplified for larger datasets)
                if len(numeric_cols) <= 20:  # Only include full matrix for smaller datasets
                    relationships["correlation_matrix"] = corr_matrix.to_dict()
            except Exception as e:
                logger.warning(f"Error in correlation analysis: {str(e)}")
                relationships["high_correlations"] = []
        else:
            relationships["high_correlations"] = []
            
        # Add categorical relationship detection (Chi-square test) - stubbed for now
        categorical_cols = [col for col, type in self.profile.get("column_types", {}).items() 
                          if type in ["categorical", "binary"]]
        
        if len(categorical_cols) > 1:
            relationships["categorical_associations"] = []  # Stub for Chi-square tests
            
        return relationships
    
    def _detect_quality_issues(self) -> List[Dict[str, Any]]:
        """Detect potential data quality issues"""
        logger.debug("Detecting data quality issues")
        issues = []
        
        # Check for columns with high percentage of missing values
        for col, pct in self.profile.get("missing_data", {}).get("missing_percentages", {}).items():
            if pct > 10:
                issues.append({
                    "type": "high_missing_values",
                    "column": col,
                    "description": f"Column has {pct:.1f}% missing values"
                })
        
        # Check for potential outliers in numeric columns
        for col, col_type in self.profile.get("column_types", {}).items():
            if col_type == "numeric":
                try:
                    # Simple IQR-based outlier detection
                    q1 = self.data[col].quantile(0.25)
                    q3 = self.data[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outlier_count = ((self.data[col] < lower_bound) | 
                                    (self.data[col] > upper_bound)).sum()
                    
                    if outlier_count > 0:
                        pct_outliers = (outlier_count / len(self.data)) * 100
                        if pct_outliers > 1:  # Only report if > 1% are outliers
                            issues.append({
                                "type": "potential_outliers",
                                "column": col,
                                "description": f"Column has {pct_outliers:.1f}% potential outliers"
                            })
                except Exception as e:
                    logger.warning(f"Error detecting outliers in {col}: {str(e)}")
        
        # Check for duplicate rows
        duplicate_count = self.data.duplicated().sum()
        if duplicate_count > 0:
            pct_duplicates = (duplicate_count / len(self.data)) * 100
            if pct_duplicates > 0.5:  # Only report if > 0.5% are duplicates
                issues.append({
                    "type": "duplicate_rows",
                    "column": "entire_dataset",
                    "description": f"Dataset contains {pct_duplicates:.1f}% duplicate rows"
                })
        
        # Check for unexpected values in categorical columns
        for col, col_type in self.profile.get("column_types", {}).items():
            if col_type in ["categorical", "binary"]:
                # This is simplified - in a real implementation, we'd have domain-specific rules
                # For example, checking "Gender" column for values other than expected ones
                pass
                
        # Check for inconsistent data types within columns
        for col, col_type in self.profile.get("column_types", {}).items():
            if col_type == "object":
                # This would check for mixed types in object columns
                # Simplified for now
                pass
                
        return issues
                
    def get_column_suggestions(self) -> Dict[str, Any]:
        """Generate suggested analysis for specific columns"""
        # This is a stub method that would be implemented in a full version
        suggestions = {
            "target_column_candidates": [],
            "problematic_columns": [],
            "informative_columns": []
        }
        
        # Target column candidates would be numeric or categorical columns
        # with good distribution and low missing values
        
        # Problematic columns would be those with high missing values,
        # high outlier percentages, or other quality issues
        
        # Informative columns would be those with high correlation to
        # potential target columns
        
        return suggestions
        
    def add_external_data(self, external_data: pd.DataFrame, join_column: str) -> None:
        """Add external data to the current dataset for enhanced analysis
        
        This is a stub method that would be implemented in a full version.
        It would handle merging external data with the current dataset.
        """
        logger.warning("add_external_data method is not fully implemented")
        # Would implement data joining logic here