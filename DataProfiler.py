import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import os
from DataConnector import get_connector
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

class DataProfiler:
    """Class to profile dataset characteristics and extract meaningful insights"""
    
    def __init__(self):
        self.data = None
        self.profile = {}
        logger.info("DataProfiler initialized")
        # Common geographical patterns
        self.geo_patterns = {
            'country': r'country|nation|countries',
            'state': r'state|province|region',
            'city': r'city|town|location|municipality',
            'postal': r'zip|postal|postcode|zipcode',
            'latitude': r'lat|latitude',
            'longitude': r'lon|lng|long|longitude',
            'address': r'address|street|avenue|road'
        }
        # Common units of measurement
        self.unit_patterns = {
            'currency': r'price|cost|revenue|sales|income|profit|expense|budget|value|amount|paid|payment|salary|wage|dollar|euro|pound|yen|rupee|yuan',
            'weight': r'kg|kilogram|gram|pound|lb|ounce|oz|weight',
            'distance': r'meter|km|mile|foot|feet|inch|yard|distance',
            'time': r'seconds|second|minute|hour|day|month|year|time|duration',
            'temperature': r'celsius|fahrenheit|kelvin|temperature|degree',
            'percentage': r'percent|ratio|rate|proportion'
        }
        # Common time series patterns
        self.time_series_patterns = {
            'daily': r'daily|day|date',
            'weekly': r'weekly|week',
            'monthly': r'monthly|month',
            'quarterly': r'quarter|quarterly',
            'yearly': r'yearly|annual|year'
        }
    
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
                "data_quality_issues": self._detect_quality_issues(),
                "metadata": self._extract_metadata(),
                "temporal_patterns": self._detect_temporal_patterns(),
                "geographical_data": self._identify_geographical_data(),
                "hierarchical_relationships": self._detect_hierarchical_relationships()
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
            "columns": list(self.data.columns),
            "data_shape": self.data.shape
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
                    if self._is_ordinal(self.data[col]):
                        column_types[col] = "ordinal"
                    else:
                        column_types[col] = "categorical"
                elif self._is_datetime(self.data[col]):
                    column_types[col] = "datetime"
                elif self._is_text(self.data[col]):
                    column_types[col] = "text"
                elif self._is_email(self.data[col]):
                    column_types[col] = "email"
                elif self._is_url(self.data[col]):
                    column_types[col] = "url"
                elif self._is_ip_address(self.data[col]):
                    column_types[col] = "ip_address"
                else:
                    column_types[col] = "object"
            elif "int" in dtype or "float" in dtype:
                if self._is_binary(self.data[col]):
                    column_types[col] = "binary"
                elif self._is_categorical(self.data[col]):
                    if self._is_ordinal(self.data[col]):
                        column_types[col] = "ordinal"
                    else:
                        column_types[col] = "categorical"
                elif self._is_continuous(self.data[col]):
                    column_types[col] = "continuous"
                else:
                    column_types[col] = "numeric"
            elif "datetime" in dtype:
                column_types[col] = "datetime"
            else:
                column_types[col] = dtype
                
        return column_types
    
    # Enhanced helper methods for type detection
    def _is_categorical(self, series: pd.Series) -> bool:
        """Check if a column appears to be categorical"""
        unique_count = series.nunique()
        # Check if number of unique values is small relative to the total
        threshold = min(10, len(series) * 0.05)
        return unique_count < threshold
    
    def _is_ordinal(self, series: pd.Series) -> bool:
        """Check if a categorical column appears to be ordinal"""
        # Look for common ordinal patterns like 'low/medium/high', 'small/large', etc.
        if series.dtype == "object":
            # Common ordinal terms
            ordinal_terms = ['low', 'medium', 'high', 'small', 'large', 
                           'poor', 'fair', 'good', 'excellent',
                           'strongly disagree', 'disagree', 'neutral', 'agree', 'strongly agree']
            
            # Convert to lowercase strings for checking
            values = series.dropna().astype(str).str.lower()
            # Check if at least half of the values match ordinal terms
            matching = sum(values.isin(ordinal_terms))
            return matching / len(values) > 0.5 if len(values) > 0 else False
        
        return False
    
    def _is_continuous(self, series: pd.Series) -> bool:
        """Check if a numeric column appears to be continuous rather than discrete"""
        # If it's not numeric, it's not continuous
        if not pd.api.types.is_numeric_dtype(series):
            return False
            
        # Drop NA values for analysis
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return False
            
        # Check if it has decimal points
        if pd.api.types.is_float_dtype(series):
            # Check if there are actually decimal values (not just .0)
            decimal_values = clean_series - clean_series.astype(int)
            return (decimal_values.abs() > 1e-10).any()
            
        # For integers, check if the set of unique values is sparse
        # (continuous data should have many unique values relatively close together)
        unique_values = clean_series.nunique()
        return unique_values > 10 and unique_values > 0.1 * len(clean_series)
    
    def _is_datetime(self, series: pd.Series) -> bool:
        """Check if a column appears to contain datetime values"""
        # If it's already a datetime dtype, return True
        if pd.api.types.is_datetime64_dtype(series):
            return True
            
        # If not a string type, it's not a datetime
        if series.dtype != "object":
            return False
            
        # Get non-null values
        values = series.dropna()
        if len(values) == 0:
            return False
            
        # Check a sample of values
        sample = values.sample(min(100, len(values)))
        try:
            pd.to_datetime(sample)
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
        unique_values = series.dropna().unique()
        if len(unique_values) > 2:
            return False
        if len(unique_values) <= 1:
            return False
            
        # Check common binary patterns
        binary_pairs = [
            {0, 1}, {'0', '1'}, 
            {'yes', 'no'}, {'y', 'n'},
            {'true', 'false'}, {True, False},
            {'t', 'f'}, {'pass', 'fail'},
            {'male', 'female'}, {'m', 'f'}
        ]
        
        # Normalize values for checking
        if pd.api.types.is_numeric_dtype(series):
            unique_set = set(unique_values)
        else:
            # Convert to lowercase strings
            unique_set = set(str(val).lower() for val in unique_values)
            
        # Check if the unique values match any common binary pattern
        for pair in binary_pairs:
            if unique_set == pair or unique_set == {str(v).lower() for v in pair}:
                return True
                
        return False
    
    def _is_email(self, series: pd.Series) -> bool:
        """Check if a column contains email addresses"""
        if series.dtype != "object":
            return False
            
        # Get non-null values
        values = series.dropna()
        if len(values) == 0:
            return False
            
        # Check a sample for email pattern
        sample = values.sample(min(100, len(values))).astype(str)
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        matches = sample.str.match(email_pattern)
        return matches.mean() > 0.7  # If >70% match the pattern
    
    def _is_url(self, series: pd.Series) -> bool:
        """Check if a column contains URLs"""
        if series.dtype != "object":
            return False
            
        # Get non-null values
        values = series.dropna()
        if len(values) == 0:
            return False
            
        # Check a sample for URL pattern
        sample = values.sample(min(100, len(values))).astype(str)
        url_pattern = r'^(http|https)://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$'
        matches = sample.str.match(url_pattern)
        return matches.mean() > 0.7  # If >70% match the pattern
    
    def _is_ip_address(self, series: pd.Series) -> bool:
        """Check if a column contains IP addresses"""
        if series.dtype != "object":
            return False
            
        # Get non-null values
        values = series.dropna()
        if len(values) == 0:
            return False
            
        # Check a sample for IP pattern
        sample = values.sample(min(100, len(values))).astype(str)
        ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        matches = sample.str.match(ipv4_pattern)
        return matches.mean() > 0.7  # If >70% match the pattern
    
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
        
        # Advanced missing patterns
        if missing_counts.sum() > 0:
            # Check for columns that tend to be missing together
            missing_matrix = self.data.isnull().astype(int)
            
            # Identify correlated missingness
            if len(self.data.columns) > 1:
                corr_matrix = missing_matrix.corr()
                missing_together = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if corr_matrix.iloc[i, j] > 0.7:  # Threshold for correlated missingness
                            missing_together.append({
                                "col1": corr_matrix.columns[i],
                                "col2": corr_matrix.columns[j],
                                "correlation": float(corr_matrix.iloc[i, j])
                            })
                
                result["columns_missing_together"] = missing_together
                
                # Identify potentially meaningful missingness
                # (e.g., certain groups have more missing values)
                if missing_together and len(missing_together) < len(self.data.columns):
                    result["meaningful_missingness"] = self._find_meaningful_missingness(missing_matrix)
        
        return result
    
    def _find_meaningful_missingness(self, missing_matrix: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find patterns where missingness may be related to values in other columns"""
        meaningful_patterns = []
        
        # For each column with missing values
        for col in self.data.columns:
            if self.data[col].isnull().sum() == 0:
                continue
                
            missing_idx = self.data[col].isnull()
            non_missing_idx = ~missing_idx
            
            # Check relationships with categorical/binary columns
            for cat_col, col_type in self.profile.get("column_types", {}).items():
                if cat_col == col or col_type not in ["categorical", "binary", "ordinal"]:
                    continue
                    
                # Calculate missingness rate by category
                if len(missing_idx) > 0 and len(non_missing_idx) > 0:
                    try:
                        # Group by category and calculate missing rate
                        missing_by_category = {}
                        categories = self.data[cat_col].dropna().unique()
                        
                        for category in categories:
                            category_rows = self.data[cat_col] == category
                            missing_rate = missing_idx[category_rows].mean()
                            non_missing_rate = missing_idx[~category_rows].mean()
                            
                            # If there's a significant difference in missingness rates
                            if abs(missing_rate - non_missing_rate) > 0.2:
                                missing_by_category[str(category)] = {
                                    "missing_rate": float(missing_rate),
                                    "overall_rate": float(missing_idx.mean()),
                                    "difference": float(missing_rate - non_missing_rate)
                                }
                        
                        if missing_by_category:
                            meaningful_patterns.append({
                                "missing_column": col,
                                "related_to": cat_col,
                                "patterns": missing_by_category
                            })
                    except:
                        # Skip if there's an error in calculation
                        continue
        
        return meaningful_patterns
    
    def _analyze_distributions(self) -> Dict[str, Dict[str, Any]]:
        """Analyze numerical and categorical distributions"""
        logger.debug("Analyzing distributions")
        distributions = {}
        
        for col, col_type in self.profile.get("column_types", {}).items():
            if col_type in ["numeric", "continuous"]:
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
                        "kurtosis": float(clean_data.kurtosis() if hasattr(clean_data, 'kurtosis') else 0),
                        "histogram_data": self._create_histogram(clean_data),
                        "quantiles": {
                            "25%": float(clean_data.quantile(0.25)),
                            "50%": float(clean_data.quantile(0.5)),
                            "75%": float(clean_data.quantile(0.75)),
                            "90%": float(clean_data.quantile(0.9)),
                            "95%": float(clean_data.quantile(0.95)),
                            "99%": float(clean_data.quantile(0.99))
                        }
                    }
            elif col_type in ["categorical", "binary", "ordinal"]:
                value_counts = self.data[col].value_counts()
                
                if len(value_counts) > 0:
                    distributions[col] = {
                        "value_counts": value_counts.to_dict(),
                        "top_value": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                        "entropy": self._calculate_entropy(self.data[col]),
                        "unique_count": int(self.data[col].nunique()),
                        "is_imbalanced": self._check_categorical_imbalance(value_counts)
                    }
                    
        return distributions
    
    def _check_categorical_imbalance(self, value_counts: pd.Series) -> Dict[str, Any]:
        """Check if a categorical variable is imbalanced"""
        if len(value_counts) <= 1:
            return {"status": False}
            
        # Calculate normalized frequencies
        total = value_counts.sum()
        frequencies = value_counts / total
        
        # Check imbalance metrics
        max_freq = frequencies.max()
        min_freq = frequencies.min()
        imbalance_ratio = max_freq / min_freq if min_freq > 0 else float('inf')
        
        return {
            "status": imbalance_ratio > 10,  # Consider imbalanced if ratio > 10
            "ratio": float(imbalance_ratio),
            "dominant_class": str(frequencies.idxmax()),
            "dominant_percentage": float(max_freq * 100)
        }
    
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
        
        # Enhanced correlation analysis for numeric columns
        numeric_cols = [col for col, type in self.profile.get("column_types", {}).items() 
                      if type in ["numeric", "continuous"]]
        
        if len(numeric_cols) > 1:
            try:
                # Use Spearman correlation as it works for non-linear relationships
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
            
        # Categorical association detection (Chi-square test or Cramer's V)
        categorical_cols = [col for col, type in self.profile.get("column_types", {}).items() 
                          if type in ["categorical", "binary", "ordinal"]]
        
        if len(categorical_cols) > 1:
            categorical_associations = self._analyze_categorical_associations(categorical_cols)
            relationships["categorical_associations"] = categorical_associations
            
        # Detect potential key columns (unique identifiers)
        potential_keys = []
        for col, col_type in self.profile.get("column_types", {}).items():
            # Skip columns with missing values
            if self.data[col].isnull().any():
                continue
                
            unique_ratio = self.data[col].nunique() / len(self.data)
            if unique_ratio > 0.9:  # If > 90% of values are unique
                potential_keys.append({
                    "column": col,
                    "unique_ratio": float(unique_ratio),
                    "is_unique": self.data[col].is_unique
                })
                
        relationships["potential_keys"] = potential_keys
        
        # Detect potential functional dependencies
        relationships["functional_dependencies"] = self._detect_functional_dependencies()
            
        return relationships
    
    def _analyze_categorical_associations(self, categorical_cols: List[str]) -> List[Dict[str, Any]]:
        """Analyze associations between categorical variables"""
        associations = []
        
        for i in range(len(categorical_cols)):
            for j in range(i+1, len(categorical_cols)):
                col1 = categorical_cols[i]
                col2 = categorical_cols[j]
                
                try:
                    # Create a contingency table
                    contingency = pd.crosstab(self.data[col1], self.data[col2])
                    
                    # Calculate Cramer's V (normalized measure of association)
                    n = contingency.sum().sum()
                    chi2 = 0
                    
                    # Only proceed if we have enough data
                    if n > 0 and min(contingency.shape) > 1:
                        # Expected frequencies under independence
                        row_sums = contingency.sum(axis=1).values
                        col_sums = contingency.sum(axis=0).values
                        expected = np.outer(row_sums, col_sums) / n
                        
                        # Calculate chi-square statistic
                        chi2 = ((contingency.values - expected) ** 2 / expected).sum()
                        
                        # Calculate Cramer's V
                        phi2 = chi2 / n
                        r, k = contingency.shape
                        cramers_v = np.sqrt(phi2 / min(k-1, r-1)) if min(k-1, r-1) > 0 else 0
                        
                        # Only include strong associations
                        if cramers_v > 0.3:  # Threshold for medium-strong association
                            associations.append({
                                "col1": col1,
                                "col2": col2,
                                "cramers_v": float(cramers_v),
                                "strength": "strong" if cramers_v > 0.5 else "medium"
                            })
                except Exception as e:
                    logger.warning(f"Error calculating association between {col1} and {col2}: {str(e)}")
                    continue
        
        return associations
    
    def _detect_functional_dependencies(self) -> List[Dict[str, Any]]:
        """Detect potential functional dependencies between columns"""
        dependencies = []
        
        # Get columns to check (excluding potential key columns)
        cols_to_check = [col for col in self.data.columns 
                        if self.data[col].nunique() / len(self.data) < 0.9]
        
        # For each potential determinant column
        for col1 in cols_to_check:
            # For each potential dependent column
            for col2 in cols_to_check:
                if col1 == col2:
                    continue
                    
                try:
                    # Group by col1 and check if col2 values are consistent within groups
                    consistency = self.data.groupby(col1)[col2].nunique().max()
                    
                    # If col1 values consistently map to a single col2 value
                    if consistency == 1:
                        dependencies.append({
                            "determinant": col1,
                            "dependent": col2,
                            "confidence": "high"
                        })
                except Exception as e:
                    logger.warning(f"Error detecting functional dependency: {str(e)}")
                    continue
        
        return dependencies
    
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
            if col_type in ["numeric", "continuous"]:
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
                                "description": f"Column has {pct_outliers:.1f}% potential outliers",
                                "count": int(outlier_count)
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
                    "description": f"Dataset contains {pct_duplicates:.1f}% duplicate rows",
                    "count": int(duplicate_count)
                })
        
        # Check for inconsistent data types within columns
        for col, col_type in self.profile.get("column_types", {}).items():
            if col_type == "object":
                try:
                    # For object columns, check if they contain mixed types
                    sample = self.data[col].dropna().astype(str).sample(min(1000, len(self.data)))
                    
                    # Try to convert to numeric
                    numeric_conversion = pd.to_numeric(sample, errors='coerce')
                    pct_numeric = (~numeric_conversion.isna()).mean() * 100
                    
                    # Try to convert to datetime
                    datetime_conversion = pd.to_datetime(sample, errors='coerce')
                    pct_datetime = (~datetime_conversion.isna()).mean() * 100
                    
                    # If it's partially numeric or datetime, it might be mixed
                    if 10 < pct_numeric < 90 or 10 < pct_datetime < 90:
                        issues.append({
                            "type": "mixed_types",
                            "column": col,
                            "description": f"Column may contain mixed data types: {pct_numeric:.1f}% numeric, {pct_datetime:.1f}% datetime"
                        })
                except:
                    continue
        
        # Check for low variance columns
        for col, col_type in self.profile.get("column_types", {}).items():
            if col_type in ["numeric", "continuous"]:
                try:
                    # Calculate coefficient of variation
                    mean = self.data[col].mean()
                    std = self.data[col].std()
                    
                    if mean != 0 and not np.isnan(mean) and not np.isnan(std):
                        cv = abs(std / mean)
                        
                        if cv < 0.1:  # Very low variation
                            issues.append({
                                "type": "low_variance",
                                "column": col,
                                "description": f"Column has very low variance (CV = {cv:.3f})"
                            })
                except:
                    continue
                    
        return issues
    
    def _extract_metadata(self) -> Dict[str, Any]:
        """Extract metadata from column names and content"""
        logger.debug("Extracting metadata")
        metadata = {}
        
        # Extract domain vocabulary from column names
        domain_terms = self._extract_domain_vocabulary()
        metadata["domain_terms"] = domain_terms
        
        # Identify units of measurement
        units = self._identify_units_of_measurement()
        metadata["units_of_measurement"] = units
        
        # Dataset classification
        metadata["dataset_type"] = self._classify_dataset_type()
        
        return metadata
    
    def _extract_domain_vocabulary(self) -> Dict[str, List[str]]:
        """Extract domain-specific terminology from column names"""
        domain_categories = {
            'financial': ['price', 'cost', 'revenue', 'profit', 'expense', 'income', 'sales', 'tax', 'interest', 'loan', 'budget', 'debit', 'credit'],
            'customer': ['customer', 'client', 'account', 'satisfaction', 'loyalty', 'retention', 'churn', 'acquisition', 'segment', 'persona'],
            'product': ['product', 'item', 'inventory', 'stock', 'warehouse', 'category', 'brand', 'model', 'version', 'feature'],
            'marketing': ['campaign', 'promotion', 'discount', 'coupon', 'advertisement', 'channel', 'conversion', 'engagement', 'click', 'impression'],
            'healthcare': ['patient', 'diagnosis', 'treatment', 'prescription', 'doctor', 'hospital', 'clinic', 'symptom', 'disease', 'medication'],
            'education': ['student', 'course', 'grade', 'score', 'test', 'exam', 'class', 'teacher', 'school', 'university', 'degree'],
            'human_resources': ['employee', 'salary', 'position', 'title', 'department', 'hire', 'terminate', 'performance', 'review', 'compensation'],
            'ecommerce': ['order', 'cart', 'checkout', 'shipping', 'delivery', 'payment', 'return', 'refund', 'review', 'rating']
        }
        
        results = {}
        
        # Check column names against domain terms
        all_columns = ' '.join(self.data.columns).lower()
        for domain, terms in domain_categories.items():
            matches = [term for term in terms if term in all_columns]
            if matches:
                results[domain] = matches
        
        return results
    
    def _identify_units_of_measurement(self) -> Dict[str, List[Dict[str, str]]]:
        """Identify columns that likely contain measurements with units"""
        units = {}
        
        # Check column names against unit patterns
        for unit_type, pattern in self.unit_patterns.items():
            matches = []
            for col in self.data.columns:
                if re.search(pattern, col.lower()):
                    matches.append({
                        "column": col,
                        "likely_unit": unit_type
                    })
            if matches:
                units[unit_type] = matches
        
        return units
    
    def _classify_dataset_type(self) -> Dict[str, float]:
        """Classify the dataset into common types (time series, transactional, etc.)"""
        classification = {}
        
        # Check for time series characteristics
        datetime_cols = [col for col, type in self.profile.get("column_types", {}).items() 
                       if type == "datetime"]
        if datetime_cols:
            classification["time_series"] = 0.8
            
            # Check if sorted by datetime
            for dt_col in datetime_cols:
                try:
                    is_sorted = self.data[dt_col].is_monotonic_increasing
                    if is_sorted:
                        classification["time_series"] = 0.95
                        break
                except:
                    continue
        
        # Check for transactional data characteristics
        transaction_indicators = ['order', 'transaction', 'purchase', 'sale', 'payment']
        has_transaction_terms = any(term in ' '.join(self.data.columns).lower() for term in transaction_indicators)
        has_date = any(col for col, type in self.profile.get("column_types", {}).items() if type == "datetime")
        has_id = any("id" in col.lower() for col in self.data.columns)
        
        if has_transaction_terms and has_date and has_id:
            classification["transactional"] = 0.9
            
        # Check for customer/user data
        user_indicators = ['user', 'customer', 'client', 'member', 'subscriber']
        has_user_terms = any(term in ' '.join(self.data.columns).lower() for term in user_indicators)
        
        if has_user_terms:
            classification["user_data"] = 0.85
            
        # Check for survey/questionnaire data
        likert_indicators = ['agree', 'disagree', 'satisfaction', 'rating', 'score']
        has_likert_terms = any(term in ' '.join(self.data.columns).lower() for term in likert_indicators)
        
        categorical_cols = [col for col, type in self.profile.get("column_types", {}).items() 
                          if type in ["categorical", "ordinal"]]
        many_categorical = len(categorical_cols) > len(self.data.columns) * 0.7
        
        if has_likert_terms and many_categorical:
            classification["survey_data"] = 0.8
            
        return classification
    
    def _detect_temporal_patterns(self) -> Dict[str, Any]:
        """Detect temporal patterns in datetime columns"""
        logger.debug("Detecting temporal patterns")
        patterns = {}
        
        datetime_cols = [col for col, type in self.profile.get("column_types", {}).items() 
                       if type == "datetime"]
        
        for dt_col in datetime_cols:
            try:
                # Convert to datetime if not already
                dt_series = pd.to_datetime(self.data[dt_col])
                
                # Check if sorted
                is_chronological = dt_series.is_monotonic_increasing
                
                # Get time span
                min_date = dt_series.min()
                max_date = dt_series.max()
                time_span = (max_date - min_date).days
                
                # Check frequency
                date_diffs = dt_series.sort_values().diff().dropna()
                
                # Empty result if no data
                if len(date_diffs) == 0:
                    continue
                    
                # Most common difference
                most_common_diff = date_diffs.dt.total_seconds().value_counts().idxmax()
                
                # Determine frequency type
                frequency = "irregular"
                if abs(most_common_diff - 86400) < 60:  # Within a minute of a day
                    frequency = "daily"
                elif abs(most_common_diff - 604800) < 3600:  # Within an hour of a week
                    frequency = "weekly"
                elif abs(most_common_diff - 2592000) < 86400:  # Within a day of a month
                    frequency = "monthly"
                elif abs(most_common_diff - 31536000) < 86400:  # Within a day of a year
                    frequency = "yearly"
                elif most_common_diff < 86400:  # Less than a day
                    frequency = "intraday"
                
                # Check for seasonality in time series data
                seasonality = {"detected": False}
                if is_chronological and len(dt_series) > 10:
                    # Check for basic seasonal patterns
                    by_month = dt_series.dt.month.value_counts().sort_index()
                    by_day_of_week = dt_series.dt.dayofweek.value_counts().sort_index()
                    by_hour = dt_series.dt.hour.value_counts().sort_index()
                    
                    # If some months/days/hours have significantly more entries
                    month_variation = by_month.std() / by_month.mean() if by_month.mean() > 0 else 0
                    day_variation = by_day_of_week.std() / by_day_of_week.mean() if by_day_of_week.mean() > 0 else 0
                    hour_variation = by_hour.std() / by_hour.mean() if by_hour.mean() > 0 else 0
                    
                    if month_variation > 0.2:
                        seasonality = {
                            "detected": True,
                            "type": "monthly",
                            "strength": float(month_variation)
                        }
                    elif day_variation > 0.2:
                        seasonality = {
                            "detected": True,
                            "type": "weekly",
                            "strength": float(day_variation)
                        }
                    elif hour_variation > 0.2:
                        seasonality = {
                            "detected": True,
                            "type": "daily",
                            "strength": float(hour_variation)
                        }
                
                patterns[dt_col] = {
                    "is_chronological": bool(is_chronological),
                    "min_date": str(min_date),
                    "max_date": str(max_date),
                    "time_span_days": int(time_span),
                    "frequency": frequency,
                    "seasonality": seasonality,
                    "complete": self._check_datetime_completeness(dt_series, frequency)
                }
            except Exception as e:
                logger.warning(f"Error detecting temporal patterns in {dt_col}: {str(e)}")
                continue
        
        return patterns
    
    def _check_datetime_completeness(self, dt_series: pd.Series, frequency: str) -> Dict[str, Any]:
        """Check if a datetime series is complete without gaps"""
        try:
            # Sort the series
            dt_series = dt_series.sort_values().dropna()
            
            if len(dt_series) < 2:
                return {"status": True, "gap_percentage": 0.0}
                
            # Create a complete date range at the given frequency
            if frequency == "daily":
                complete_range = pd.date_range(start=dt_series.min(), end=dt_series.max(), freq='D')
            elif frequency == "weekly":
                complete_range = pd.date_range(start=dt_series.min(), end=dt_series.max(), freq='W')
            elif frequency == "monthly":
                complete_range = pd.date_range(start=dt_series.min(), end=dt_series.max(), freq='MS')
            elif frequency == "yearly":
                complete_range = pd.date_range(start=dt_series.min(), end=dt_series.max(), freq='YS')
            elif frequency == "intraday":
                # For intraday, use hourly frequency as approximation
                complete_range = pd.date_range(start=dt_series.min(), end=dt_series.max(), freq='H')
            else:
                # For irregular data, can't determine completeness
                return {"status": "unknown", "reason": "irregular frequency"}
            
            # Check how many dates are missing
            existing_dates = set(dt_series.dt.floor('D') if frequency == "intraday" else dt_series)
            expected_dates = set(complete_range)
            
            missing_dates = expected_dates - existing_dates
            missing_percentage = len(missing_dates) / len(expected_dates) * 100
            
            return {
                "status": missing_percentage < 5,  # Consider complete if < 5% missing
                "gap_percentage": float(missing_percentage),
                "missing_count": len(missing_dates)
            }
        except Exception as e:
            logger.warning(f"Error checking datetime completeness: {str(e)}")
            return {"status": "unknown", "reason": str(e)}
    
    def _identify_geographical_data(self) -> Dict[str, List[Dict[str, str]]]:
        """Identify columns that likely contain geographical data"""
        logger.debug("Identifying geographical data")
        geo_data = {}
        
        # Check column names against geo patterns
        for geo_type, pattern in self.geo_patterns.items():
            matches = []
            for col in self.data.columns:
                if re.search(pattern, col.lower()):
                    matches.append({
                        "column": col,
                        "confidence": "high" if re.match(f".*({pattern}).*", col.lower()) else "medium"
                    })
            if matches:
                geo_data[geo_type] = matches
        
        # Check for columns with known geographical values
        for col, col_type in self.profile.get("column_types", {}).items():
            if col_type in ["categorical", "object", "text"]:
                try:
                    # Geographical pattern matching would be more sophisticated in practice
                    # Here we use a simplified approach checking for common country/state names
                    if self._contains_geo_entities(self.data[col]):
                        geo_type = self._determine_geo_type(self.data[col])
                        if geo_type and geo_type not in geo_data:
                            geo_data[geo_type] = []
                        if geo_type:
                            geo_data[geo_type].append({
                                "column": col,
                                "confidence": "medium",
                                "detected_by": "content_analysis"
                            })
                except:
                    continue
        
        # Check for coordinate pairs
        lat_cols = []
        lon_cols = []
        
        for geo_type, columns in geo_data.items():
            if geo_type == "latitude":
                lat_cols.extend([c["column"] for c in columns])
            elif geo_type == "longitude":
                lon_cols.extend([c["column"] for c in columns])
        
        # If we found both lat and lon columns, mark them as a coordinate pair
        if lat_cols and lon_cols:
            geo_data["coordinate_pairs"] = []
            for lat_col in lat_cols:
                for lon_col in lon_cols:
                    geo_data["coordinate_pairs"].append({
                        "latitude_column": lat_col,
                        "longitude_column": lon_col,
                        "confidence": "high"
                    })
        
        return geo_data
    
    def _contains_geo_entities(self, series: pd.Series) -> bool:
        """Check if a column contains geographical entities"""
        # In practice, this would use more sophisticated methods like NER
        # For this implementation, we'll use a simplified check against common terms
        
        common_countries = ['usa', 'us', 'united states', 'uk', 'canada', 'australia', 'germany', 
                          'france', 'japan', 'china', 'india', 'brazil', 'mexico']
        common_states = ['california', 'texas', 'florida', 'new york', 'illinois', 'pennsylvania', 
                       'ohio', 'michigan', 'georgia', 'north carolina']
                       
        # Sample the series to avoid processing the entire column
        sample = series.dropna().astype(str).str.lower().sample(min(100, len(series)))
        
        # Check for matches
        for term in common_countries + common_states:
            if any(sample.str.contains(f"\\b{term}\\b")):
                return True
                
        return False
    
    def _determine_geo_type(self, series: pd.Series) -> Optional[str]:
        """Determine what type of geographical data a column contains"""
        # In practice, this would use more sophisticated pattern matching
        sample = series.dropna().astype(str).str.lower().sample(min(100, len(series)))
        
        # Check for common country names
        common_countries = ['usa', 'us', 'united states', 'uk', 'canada', 'australia', 'germany']
        if any(sample.str.contains('|'.join([f"\\b{c}\\b" for c in common_countries]))):
            return "country"
            
        # Check for common state names
        common_states = ['california', 'texas', 'florida', 'new york', 'illinois']
        if any(sample.str.contains('|'.join([f"\\b{s}\\b" for s in common_states]))):
            return "state"
            
        # Check for zip code pattern
        if sample.str.match(r'^\d{5}(-\d{4})?$').any():
            return "postal"
            
        # If uncertain, return None
        return None
    
    def _detect_hierarchical_relationships(self) -> Dict[str, Any]:
        """Detect potential hierarchical relationships in the data"""
        logger.debug("Detecting hierarchical relationships")
        hierarchies = {}
        
        # Find categorical columns
        categorical_cols = [col for col, type in self.profile.get("column_types", {}).items() 
                          if type in ["categorical", "ordinal"]]
        
        # Look for geographical hierarchies
        geo_hierarchies = self._find_geo_hierarchies()
        if geo_hierarchies:
            hierarchies["geographical"] = geo_hierarchies
            
        # Look for temporal hierarchies
        datetime_cols = [col for col, type in self.profile.get("column_types", {}).items() 
                       if type == "datetime"]
        
        if datetime_cols:
            temp_hierarchies = self._find_temporal_hierarchies(datetime_cols)
            if temp_hierarchies:
                hierarchies["temporal"] = temp_hierarchies
                
        # Look for general hierarchical relationships
        # This checks if one column's values are consistently nested within another's
        if len(categorical_cols) > 1:
            general_hierarchies = []
            
            for i in range(len(categorical_cols)):
                for j in range(len(categorical_cols)):
                    if i == j:
                        continue
                        
                    col1 = categorical_cols[i]  # Potential parent
                    col2 = categorical_cols[j]  # Potential child
                    
                    try:
                        # Check if col2 values are nested within col1
                        grouped = self.data.groupby(col1)[col2].nunique()
                        
                        # Calculate the average number of unique child values per parent
                        avg_children_per_parent = grouped.mean()
                        
                        # For a hierarchy, we expect:
                        # 1. Multiple child values per parent (>1 on average)
                        # 2. Fewer unique parent values than child values
                        if avg_children_per_parent > 1 and self.data[col1].nunique() < self.data[col2].nunique():
                            # Calculate consistency ratio - child values should consistently appear with same parent
                            consistency = self._calculate_hierarchy_consistency(col1, col2)
                            
                            # Only include if reasonably consistent
                            if consistency > 0.7:
                                general_hierarchies.append({
                                    "parent": col1,
                                    "child": col2,
                                    "avg_children_per_parent": float(avg_children_per_parent),
                                    "consistency": float(consistency),
                                    "confidence": "high" if consistency > 0.9 else "medium"
                                })
                    except Exception as e:
                        logger.warning(f"Error checking hierarchy between {col1} and {col2}: {str(e)}")
                        continue
            
            if general_hierarchies:
                hierarchies["general"] = general_hierarchies
        
        return hierarchies
    
    def _find_geo_hierarchies(self) -> List[Dict[str, Any]]:
        """Find geographical hierarchies like country > state > city"""
        hierarchies = []
        
        # Check if we have geographical columns
        geo_data = self.profile.get("geographical_data", {})
        
        # Define potential hierarchy levels from highest to lowest
        geo_levels = ["country", "state", "city", "postal", "address"]
        
        # Find columns for each level
        level_columns = {}
        for level in geo_levels:
            if level in geo_data:
                level_columns[level] = [item["column"] for item in geo_data[level]]
        
        # Check for pairs of columns at adjacent levels
        for i in range(len(geo_levels) - 1):
            parent_level = geo_levels[i]
            child_level = geo_levels[i + 1]
            
            if parent_level in level_columns and child_level in level_columns:
                for parent_col in level_columns[parent_level]:
                    for child_col in level_columns[child_level]:
                        try:
                            # Check if child values are nested under parent
                            consistency = self._calculate_hierarchy_consistency(parent_col, child_col)
                            
                            if consistency > 0.6:  # Lower threshold for geographical hierarchies
                                hierarchies.append({
                                    "type": "geographical",
                                    "parent_level": parent_level,
                                    "parent_column": parent_col,
                                    "child_level": child_level,
                                    "child_column": child_col,
                                    "consistency": float(consistency)
                                })
                        except Exception as e:
                            logger.warning(f"Error checking geo hierarchy: {str(e)}")
                            continue
        
        return hierarchies
    
    def _find_temporal_hierarchies(self, datetime_cols: List[str]) -> List[Dict[str, Any]]:
        """Find temporal hierarchies like year > quarter > month > day"""
        hierarchies = []
        
        for dt_col in datetime_cols:
            try:
                # Convert to datetime
                dt_series = pd.to_datetime(self.data[dt_col])
                
                # Check if we have enough different years/months/days
                year_count = dt_series.dt.year.nunique()
                month_count = dt_series.dt.month.nunique()
                day_count = dt_series.dt.day.nunique()
                
                # If we have multiple levels of the hierarchy
                if year_count > 1 and (month_count > 1 or day_count > 1):
                    hierarchy = {
                        "type": "temporal",
                        "datetime_column": dt_col,
                        "levels": []
                    }
                    
                    # Add levels that are present
                    if year_count > 1:
                        hierarchy["levels"].append("year")
                    if month_count > 1:
                        hierarchy["levels"].append("month")
                    if day_count > 1:
                        hierarchy["levels"].append("day")
                    
                    hierarchies.append(hierarchy)
            except Exception as e:
                logger.warning(f"Error checking temporal hierarchy in {dt_col}: {str(e)}")
                continue
        
        return hierarchies
    
    def _calculate_hierarchy_consistency(self, parent_col: str, child_col: str) -> float:
        """Calculate how consistently child values appear with the same parent"""
        # Group by child and count unique parents
        child_parent_count = self.data.groupby(child_col)[parent_col].nunique()
        
        # If most children have just one parent, it's a consistent hierarchy
        consistency = (child_parent_count == 1).mean()
        
        return consistency
                
    def get_column_suggestions(self) -> Dict[str, Any]:
        """Generate suggested analysis for specific columns"""
        suggestions = {
            "target_column_candidates": [],
            "problematic_columns": [],
            "informative_columns": [],
            "recommended_transforms": []
        }
        
        # Target column candidates
        # Look for columns that might be good prediction targets
        for col, col_type in self.profile.get("column_types", {}).items():
            if col_type in ["numeric", "continuous", "categorical", "binary"]:
                # Skip columns with too many missing values
                missing_pct = self.profile.get("missing_data", {}).get("missing_percentages", {}).get(col, 0)
                if missing_pct > 20:
                    continue
                    
                # Check distribution quality
                if col in self.profile.get("distribution_info", {}):
                    dist_info = self.profile["distribution_info"][col]
                    
                    confidence = 0.5  # Base confidence
                    
                    # For categorical targets
                    if col_type in ["categorical", "binary"]:
                        # Check for class imbalance
                        if "is_imbalanced" in dist_info:
                            imbalance = dist_info["is_imbalanced"]
                            if isinstance(imbalance, dict) and imbalance.get("status", False):
                                confidence -= 0.2  # Penalize imbalanced targets
                    
                    # For numeric targets
                    elif col_type in ["numeric", "continuous"]:
                        # Prefer targets with reasonable distributions
                        if "skew" in dist_info and abs(dist_info["skew"]) > 2:
                            confidence -= 0.1  # Penalize highly skewed targets
                            
                        # Check for outliers
                        outlier_issues = [issue for issue in self.profile.get("data_quality_issues", [])
                                        if issue.get("type") == "potential_outliers" and issue.get("column") == col]
                        if outlier_issues:
                            confidence -= 0.1  # Penalize targets with outliers
                    
                    # Check relationships - good targets often have relationships with other variables
                    related_cols = []
                    
                    # Check correlations for numeric columns
                    for corr in self.profile.get("potential_relationships", {}).get("high_correlations", []):
                        if corr.get("col1") == col:
                            related_cols.append({"column": corr.get("col2"), "correlation": corr.get("correlation")})
                        elif corr.get("col2") == col:
                            related_cols.append({"column": corr.get("col1"), "correlation": corr.get("correlation")})
                    
                    # Check categorical associations
                    for assoc in self.profile.get("potential_relationships", {}).get("categorical_associations", []):
                        if assoc.get("col1") == col:
                            related_cols.append({"column": assoc.get("col2"), "association": assoc.get("cramers_v")})
                        elif assoc.get("col2") == col:
                            related_cols.append({"column": assoc.get("col1"), "association": assoc.get("cramers_v")})
                    
                    if related_cols:
                        confidence += 0.2  # Boost confidence if the column has relationships
                    
                    # Add to target candidates if reasonable confidence
                    if confidence > 0.4:
                        suggestions["target_column_candidates"].append({
                            "column": col,
                            "type": col_type,
                            "confidence": float(confidence),
                            "related_columns": related_cols
                        })
        
        # Sort target candidates by confidence
        suggestions["target_column_candidates"].sort(key=lambda x: x["confidence"], reverse=True)
        
        # Problematic columns
        # Add columns with data quality issues
        for issue in self.profile.get("data_quality_issues", []):
            if issue.get("column") == "entire_dataset":
                continue
                
            col = issue.get("column")
            if col is None:
                continue
                
            # Check if already in the list
            existing = next((item for item in suggestions["problematic_columns"] if item["column"] == col), None)
            
            if existing:
                existing["issues"].append({
                    "type": issue.get("type"),
                    "description": issue.get("description")
                })
            else:
                suggestions["problematic_columns"].append({
                    "column": col,
                    "type": self.profile.get("column_types", {}).get(col),
                    "issues": [{
                        "type": issue.get("type"),
                        "description": issue.get("description")
                    }]
                })
        
        # Informative columns
        # Find columns that have good relationships with target candidates
        if suggestions["target_column_candidates"]:
            target_col = suggestions["target_column_candidates"][0]["column"]
            
            # Find columns related to the top target
            for col, col_type in self.profile.get("column_types", {}).items():
                if col == target_col:
                    continue
                    
                # Skip problematic columns
                if any(item["column"] == col for item in suggestions["problematic_columns"]):
                    continue
                    
                # Check relationships with target
                relationship_strength = 0.0
                relationship_type = "none"
                
                # Check correlations for numeric columns
                for corr in self.profile.get("potential_relationships", {}).get("high_correlations", []):
                    if (corr.get("col1") == col and corr.get("col2") == target_col) or \
                       (corr.get("col2") == col and corr.get("col1") == target_col):
                        relationship_strength = corr.get("correlation", 0.0)
                        relationship_type = "correlation"
                        break
                
                # Check categorical associations
                if relationship_type == "none":
                    for assoc in self.profile.get("potential_relationships", {}).get("categorical_associations", []):
                        if (assoc.get("col1") == col and assoc.get("col2") == target_col) or \
                           (assoc.get("col2") == col and assoc.get("col1") == target_col):
                            relationship_strength = assoc.get("cramers_v", 0.0)
                            relationship_type = "association"
                            break
                
                # Add to informative columns if good relationship
                if relationship_strength > 0.3:
                    suggestions["informative_columns"].append({
                        "column": col,
                        "type": col_type,
                        "relationship_with_target": relationship_type,
                        "relationship_strength": float(relationship_strength)
                    })
        
        # Sort informative columns by relationship strength
        suggestions["informative_columns"].sort(key=lambda x: x["relationship_strength"], reverse=True)
        
        # Recommended transforms
        # Suggest transformations for problematic columns
        for col, col_type in self.profile.get("column_types", {}).items():
            transforms = []
            
            # Check for skewed numeric columns
            if col_type in ["numeric", "continuous"] and col in self.profile.get("distribution_info", {}):
                skew = self.profile["distribution_info"][col].get("skew", 0)
                
                if abs(skew) > 1:
                    # Suggest log transform for right-skewed data
                    if skew > 1:
                        transforms.append({
                            "transform": "log",
                            "reason": f"Column is right-skewed (skew = {skew:.2f})",
                            "confidence": 0.8 if skew > 2 else 0.6
                        })
                    # Suggest square transform for left-skewed data
                    elif skew < -1:
                        transforms.append({
                            "transform": "square",
                            "reason": f"Column is left-skewed (skew = {skew:.2f})",
                            "confidence": 0.7 if skew < -2 else 0.5
                        })
            
            # Suggest encoding for categorical columns
            if col_type == "categorical" and col in self.profile.get("distribution_info", {}):
                cardinality = self.profile["distribution_info"][col].get("unique_count", 0)
                
                if cardinality <= 10:
                    transforms.append({
                        "transform": "one-hot encoding",
                        "reason": f"Low cardinality categorical column ({cardinality} values)",
                        "confidence": 0.9
                    })
                else:
                    transforms.append({
                        "transform": "label encoding",
                        "reason": f"High cardinality categorical column ({cardinality} values)",
                        "confidence": 0.7
                    })
            
            # Suggest ordinal encoding for ordinal columns
            if col_type == "ordinal":
                transforms.append({
                    "transform": "ordinal encoding",
                    "reason": "Ordinal categorical column",
                    "confidence": 0.9
                })
            
            # Suggest normalizing for numeric columns used in distance-based models
            if col_type in ["numeric", "continuous"]:
                transforms.append({
                    "transform": "standardization",
                    "reason": "Numeric column may benefit from standardization for machine learning",
                    "confidence": 0.7
                })
                
            # Suggest datetime decomposition for datetime columns
            if col_type == "datetime":
                transforms.append({
                    "transform": "datetime decomposition",
                    "reason": "Extract year, month, day, day of week from datetime",
                    "confidence": 0.8
                })
            
            # Add transformations if any found
            if transforms:
                suggestions["recommended_transforms"].append({
                    "column": col,
                    "type": col_type,
                    "transforms": transforms
                })
        
        return suggestions
        
    def add_external_data(self, external_data: pd.DataFrame, join_column: str) -> None:
        """Add external data to the current dataset for enhanced analysis
        
        This version implements joining with external datasets.
        """
        logger.info(f"Adding external data with join column: {join_column}")
        
        try:
            # Validate inputs
            if self.data is None:
                raise ValueError("No primary data loaded. Call load_data first.")
                
            if external_data is None or len(external_data) == 0:
                raise ValueError("External data is empty")
                
            if join_column not in self.data.columns:
                raise ValueError(f"Join column '{join_column}' not found in primary dataset")
                
            if join_column not in external_data.columns:
                raise ValueError(f"Join column '{join_column}' not found in external dataset")
            
            # Perform merge
            merged_data = self.data.merge(
                external_data,
                on=join_column,
                how='left',
                suffixes=('', '_ext')
            )
            
            # Log merge results
            new_cols = [col for col in merged_data.columns if col not in self.data.columns]
            logger.info(f"Merged external data. Added {len(new_cols)} new columns.")
            
            # Check for potential duplicate columns (same name with suffix)
            duplicate_cols = [col for col in merged_data.columns if col.endswith('_ext')]
            if duplicate_cols:
                logger.warning(f"Found {len(duplicate_cols)} potential duplicate columns after merge.")
            
            # Update the internal data
            self.data = merged_data
            
            # Mark profile as outdated (needs to be regenerated)
            self.profile = {}
            
        except Exception as e:
            logger.error(f"Error adding external data: {str(e)}", exc_info=True)
            raise
    
    def find_data_linkages(self, other_dataset: pd.DataFrame) -> Dict[str, Any]:
        """Find potential linkages between this dataset and another one"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data first.")
            
        if other_dataset is None or len(other_dataset) == 0:
            raise ValueError("Other dataset is empty")
            
        logger.info(f"Finding linkages with external dataset of shape {other_dataset.shape}")
        
        linkages = {
            "exact_column_matches": [],
            "similar_column_names": [],
            "potential_key_columns": [],
            "shared_values": [],
            "recommended_joins": []
        }
        
        # Find exact column name matches
        shared_columns = set(self.data.columns).intersection(set(other_dataset.columns))
        linkages["exact_column_matches"] = list(shared_columns)
        
        # Find similar column names
        for col1 in self.data.columns:
            for col2 in other_dataset.columns:
                # Skip exact matches
                if col1 == col2:
                    continue
                    
                # Simple string similarity (could use more sophisticated methods)
                similarity = self._column_name_similarity(col1, col2)
                if similarity > 0.8:
                    linkages["similar_column_names"].append({
                        "primary_column": col1,
                        "other_column": col2,
                        "similarity": float(similarity)
                    })
        
        # Find potential key columns (columns with unique values that exist in both datasets)
        for col in shared_columns:
            # Check if column has high unique ratio in both datasets
            unique_ratio1 = self.data[col].nunique() / len(self.data)
            unique_ratio2 = other_dataset[col].nunique() / len(other_dataset)
            
            # Check if values overlap between datasets
            values1 = set(self.data[col].dropna().unique())
            values2 = set(other_dataset[col].dropna().unique())
            overlap = values1.intersection(values2)
            
            overlap_ratio1 = len(overlap) / len(values1) if len(values1) > 0 else 0
            overlap_ratio2 = len(overlap) / len(values2) if len(values2) > 0 else 0
            
            if (unique_ratio1 > 0.5 or unique_ratio2 > 0.5) and (overlap_ratio1 > 0.1 or overlap_ratio2 > 0.1):
                linkages["potential_key_columns"].append({
                    "column": col,
                    "primary_unique_ratio": float(unique_ratio1),
                    "other_unique_ratio": float(unique_ratio2),
                    "value_overlap_count": len(overlap),
                    "value_overlap_ratio": float((overlap_ratio1 + overlap_ratio2) / 2)
                })
        
        # Find columns with shared values (even if names differ)
        for col1 in self.data.columns:
            col1_type = self.profile.get("column_types", {}).get(col1)
            if col1_type not in ["categorical", "binary", "ordinal"]:
                continue
                
            values1 = set(str(x) for x in self.data[col1].dropna().unique())
            
            for col2 in other_dataset.columns:
                # Skip columns we've already matched
                if col2 in shared_columns or any(m["other_column"] == col2 for m in linkages["similar_column_names"]):
                    continue
                    
                values2 = set(str(x) for x in other_dataset[col2].dropna().unique())
                
                # Check value overlap
                overlap = values1.intersection(values2)
                if len(overlap) > 3:  # Require at least a few shared values
                    overlap_ratio = len(overlap) / min(len(values1), len(values2))
                    
                    if overlap_ratio > 0.3:  # At least 30% overlap
                        linkages["shared_values"].append({
                            "primary_column": col1,
                            "other_column": col2,
                            "shared_value_count": len(overlap),
                            "shared_value_ratio": float(overlap_ratio)
                        })
        
        # Generate recommended joins based on findings
        recommended_joins = []
        
        # First, check potential key columns (these are best for joins)
        if linkages["potential_key_columns"]:
            # Sort by overlap ratio
            sorted_keys = sorted(linkages["potential_key_columns"], 
                              key=lambda x: x["value_overlap_ratio"], 
                              reverse=True)
                              
            for key in sorted_keys[:2]:  # Top 2 recommendations
                recommended_joins.append({
                    "join_column": key["column"],
                    "join_type": "inner" if key["value_overlap_ratio"] > 0.7 else "left",
                    "confidence": key["value_overlap_ratio"],
                    "reason": f"Good key column with {key['value_overlap_count']} shared values"
                })
        
        # Next, check shared values between differently named columns
        if not recommended_joins and linkages["shared_values"]:
            # Sort by shared value ratio
            sorted_shared = sorted(linkages["shared_values"], 
                                key=lambda x: x["shared_value_ratio"], 
                                reverse=True)
                                
            for shared in sorted_shared[:1]:  # Top recommendation
                recommended_joins.append({
                    "primary_column": shared["primary_column"],
                    "other_column": shared["other_column"],
                    "join_type": "left",
                    "confidence": shared["shared_value_ratio"] * 0.8,  # Lower confidence since names differ
                    "reason": f"Columns share {shared['shared_value_count']} values despite different names"
                })
        
        linkages["recommended_joins"] = recommended_joins
        
        return linkages
    
    def _column_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between column names"""
        # Convert to lowercase and split into words
        words1 = set(name1.lower().replace('_', ' ').replace('-', ' ').split())
        words2 = set(name2.lower().replace('_', ' ').replace('-', ' ').split())
        
        # Calculate Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
        
    def extract_notebook_insights(self, notebook_path: str) -> Dict[str, Any]:
        """Extract insights from a Jupyter notebook"""
        # This would be implemented in the real system
        # For now, return a placeholder structure
        return {
            "analysis_types": [],
            "columns_used": [],
            "preprocessing_steps": [],
            "visualizations": [],
            "models": []
        }
    
    def analyze_cross_dataset(self, datasets: List[pd.DataFrame], common_dimension: str) -> Dict[str, Any]:
        """Analyze relationships across multiple datasets"""
        # This would be implemented in the real system
        # For now, return a placeholder structure
        return {
            "common_dimensions": [],
            "compatibility_scores": [],
            "join_strategies": [],
            "potential_insights": []
        }