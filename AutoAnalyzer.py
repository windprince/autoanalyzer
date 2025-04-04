from DataProfiler import DataProfiler
from AnalysisRecommender import AnalysisRecommender
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import os
import tempfile

logger = logging.getLogger(__name__)

class AutoAnalyzer:
    def __init__(self):
        self.profiler = DataProfiler()
        self.recommender = AnalysisRecommender()
        self.supported_file_types = {
            'csv': self._load_csv, 
            'json': self._load_json,
            'xlsx': self._load_excel,
            'txt': self._load_text,
            'parquet': self._load_parquet
        }
        
        # Register analysis techniques
        self.analysis_registry = {
            "distribution_analysis": self._run_distribution_analysis,
            "correlation_analysis": self._run_correlation_analysis,
            "outlier_detection": self._run_outlier_detection,
            "hypothesis_testing": self._run_hypothesis_testing,
            "anova_analysis": self._run_anova_analysis,
            "regression_analysis": self._run_regression_analysis,
            "classification": self._run_classification_analysis,
            "time_series_forecasting": self._run_time_series_analysis,
            "optimization_analysis": self._run_optimization_analysis,
            "what_if_analysis": self._run_what_if_analysis,
            "root_cause_analysis": self._run_root_cause_analysis,
            "comparative_analysis": self._run_comparative_analysis
        }
        
        # Set current dataset
        self.current_data = None
        self.current_profile = None
        
        logger.info("AutoAnalyzer initialized")
        
    def analyze_data(self, data_source, data_type="csv", objective="general"):
        """Main method to analyze data and get recommendations"""
        logger.info(f"Analyzing data from {data_source} of type {data_type}")
        
        # Step 1: Load the data
        self.current_data = self.profiler.load_data(data_source, data_type)
        
        # Step 2: Profile the data
        self.current_profile = self.profiler.generate_profile()
        
        # Step 3: Get recommendations based on objective
        recommendations = self.recommender.recommend_analyses(self.current_profile, objective)
        
        # Step 4: Return comprehensive results
        result = {
            "data_summary": self._create_data_summary(self.current_profile),
            "analysis_recommendations": recommendations,
            "data_quality_alerts": self.current_profile.get("data_quality_issues", []),
            "profile": self.current_profile,  # Full profile for advanced users
            "dataset_path": data_source  # Include the path for future reference
        }
        
        logger.info(f"Analysis complete. Found {len(recommendations)} recommendations and {len(self.current_profile.get('data_quality_issues', []))} quality issues")
        return result
        
    def run_specific_analysis(self, analysis_id, dataset_path, parameters=None):
        """Run a specific analysis technique on the provided dataset"""
        if parameters is None:
            parameters = {}
            
        logger.info(f"Running {analysis_id} analysis on {dataset_path}")
        
        # Check if the analysis is supported
        if analysis_id not in self.analysis_registry:
            raise ValueError(f"Unsupported analysis technique: {analysis_id}")
            
        # Load the data if needed (if different from current dataset)
        if not hasattr(self, 'current_data') or self.current_data is None:
            file_ext = dataset_path.split('.')[-1].lower()
            self.current_data = self.profiler.load_data(dataset_path, file_ext)
            self.current_profile = self.profiler.generate_profile()
            
        # Run the specific analysis
        analysis_func = self.analysis_registry[analysis_id]
        results = analysis_func(parameters)
        
        return results
    
    def _load_csv(self, file_path):
        """Load CSV file"""
        logger.debug(f"Loading CSV file: {file_path}")
        return self.profiler.load_data(file_path, "csv")
    
    def _load_json(self, file_path):
        """Load JSON file"""
        logger.debug(f"Loading JSON file: {file_path}")
        return self.profiler.load_data(file_path, "json")
    
    def _load_excel(self, file_path):
        """Load Excel file"""
        logger.debug(f"Loading Excel file: {file_path}")
        # Stub: Would implement Excel loading
        logger.warning("Excel loading is currently stubbed - will use CSV loader")
        return self.profiler.load_data(file_path, "csv")
    
    def _load_text(self, file_path):
        """Load text file"""
        logger.debug(f"Loading text file: {file_path}")
        # Stub: Would implement text loading
        logger.warning("Text loading is currently stubbed - will use CSV loader")
        return self.profiler.load_data(file_path, "csv")
    
    def _load_parquet(self, file_path):
        """Load Parquet file"""
        logger.debug(f"Loading Parquet file: {file_path}")
        # Stub: Would implement Parquet loading
        logger.warning("Parquet loading is currently stubbed - will use CSV loader")
        return self.profiler.load_data(file_path, "csv")
    
    def _create_data_summary(self, profile):
        """Create a human-readable summary of the dataset"""
        basic = profile.get("basic_stats", {})
        
        summary = {
            "dataset_size": f"{basic.get('row_count', 0)} rows, {basic.get('column_count', 0)} columns",
            "key_stats": []
        }
        
        # Add interesting stats about the data
        column_types = profile.get("column_types", {})
        type_counts = {
            "numeric": 0,
            "categorical": 0, 
            "binary": 0,
            "text": 0,
            "datetime": 0,
            "other": 0
        }
        
        for col_type in column_types.values():
            if col_type in type_counts:
                type_counts[col_type] += 1
            else:
                type_counts["other"] += 1
        
        summary["column_composition"] = type_counts
        
        # Note missing data
        missing_data = profile.get("missing_data", {}).get("missing_percentages", {})
        if any(pct > 5 for pct in missing_data.values()):
            missing_cols = [col for col, pct in missing_data.items() if pct > 5]
            summary["key_stats"].append(
                f"{len(missing_cols)} columns have >5% missing values"
            )
        
        # Note potential relationships
        relationships = profile.get("potential_relationships", {})
        if "high_correlations" in relationships:
            if len(relationships["high_correlations"]) > 0:
                summary["key_stats"].append(
                    f"Found {len(relationships['high_correlations'])} strong relationships between variables"
                )
        
        return summary
        
    def _run_distribution_analysis(self, parameters):
        """Run distribution analysis on selected columns"""
        logger.info("Running distribution analysis")
        
        # Get parameters or use defaults
        columns = parameters.get('columns', None)
        if columns is None:
            # Use all numeric and categorical columns if none specified
            columns = [col for col, type in self.current_profile["column_types"].items() 
                      if type in ["numeric", "categorical", "binary"]]
        
        # Limit to first 10 columns if too many
        if len(columns) > 10:
            columns = columns[:10]
            
        results = {
            "analysis_type": "distribution_analysis",
            "distributions": {},
            "summary_stats": {},
            "visualizations": {}
        }
        
        # Process each column
        for col in columns:
            if col not in self.current_data.columns:
                continue
                
            col_type = self.current_profile["column_types"].get(col)
            
            if col_type == "numeric":
                # Get distribution data
                clean_data = self.current_data[col].replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(clean_data) > 0:
                    # Summary statistics
                    results["summary_stats"][col] = {
                        "min": float(clean_data.min()),
                        "max": float(clean_data.max()),
                        "mean": float(clean_data.mean()),
                        "median": float(clean_data.median()),
                        "std": float(clean_data.std()),
                        "skew": float(clean_data.skew()),
                        "quartiles": [
                            float(clean_data.quantile(0.25)),
                            float(clean_data.quantile(0.5)),
                            float(clean_data.quantile(0.75))
                        ]
                    }
                    
                    # Create histogram data
                    hist, bin_edges = np.histogram(clean_data, bins='auto')
                    results["distributions"][col] = {
                        "type": "histogram",
                        "counts": hist.tolist(),
                        "bin_edges": bin_edges.tolist(),
                        "distribution_test": self._test_distribution(clean_data)
                    }
                    
                    # Generate and save visualization
                    fig_path = self._create_distribution_plot(clean_data, col)
                    if fig_path:
                        results["visualizations"][col] = fig_path
                    
            elif col_type in ["categorical", "binary"]:
                value_counts = self.current_data[col].value_counts()
                
                if len(value_counts) > 0:
                    # Get top 10 categories if more than 10
                    if len(value_counts) > 10:
                        top_counts = value_counts.nlargest(10)
                        other_count = value_counts.sum() - top_counts.sum()
                        counts_dict = top_counts.to_dict()
                        if other_count > 0:
                            counts_dict["Other"] = float(other_count)
                    else:
                        counts_dict = value_counts.to_dict()
                        
                    # Convert keys to strings for JSON serialization
                    counts_dict = {str(k): float(v) for k, v in counts_dict.items()}
                    
                    results["distributions"][col] = {
                        "type": "categorical",
                        "value_counts": counts_dict,
                        "entropy": float(self._calculate_entropy(self.current_data[col]))
                    }
                    
                    # Generate and save visualization
                    fig_path = self._create_categorical_plot(self.current_data[col], col)
                    if fig_path:
                        results["visualizations"][col] = fig_path
                        
        return results
        
    def _test_distribution(self, data):
        """Test what distribution the data most closely follows"""
        try:
            # Test for normality
            shapiro_test = stats.shapiro(data.sample(min(5000, len(data))))
            normal_p_value = shapiro_test[1]
            
            # Other distribution tests
            distribution_tests = {
                "normal": normal_p_value,
                "lognormal": stats.normaltest(np.log1p(data.replace(0, np.nan).dropna()))[1] if (data > 0).any() else 0,
                "exponential": stats.kstest(data, 'expon')[1] if (data >= 0).all() else 0,
                "uniform": stats.kstest(data, 'uniform', args=(data.min(), data.max()))[1]
            }
            
            best_fit = max(distribution_tests.items(), key=lambda x: x[1])
            
            return {
                "best_fit": best_fit[0],
                "p_value": float(best_fit[1]),
                "is_normal": normal_p_value > 0.05,
                "test_results": {k: float(v) for k, v in distribution_tests.items()}
            }
        except Exception as e:
            logger.warning(f"Error in distribution testing: {str(e)}")
            return {"best_fit": "unknown", "error": str(e)}
            
    def _create_distribution_plot(self, data, column_name):
        """Create and save a distribution plot for numeric data"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Create histogram with KDE
            sns.histplot(data, kde=True)
            
            # Add vertical lines for mean and median
            plt.axvline(data.mean(), color='r', linestyle='--', label=f'Mean: {data.mean():.2f}')
            plt.axvline(data.median(), color='g', linestyle='-', label=f'Median: {data.median():.2f}')
            
            plt.title(f'Distribution of {column_name}')
            plt.xlabel(column_name)
            plt.ylabel('Frequency')
            plt.legend()
            
            # Save to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                plt.savefig(tmp.name)
                plt.close()
                return tmp.name
                
        except Exception as e:
            logger.warning(f"Error creating distribution plot: {str(e)}")
            plt.close()
            return None
            
    def _create_categorical_plot(self, data, column_name):
        """Create and save a bar plot for categorical data"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Get value counts and limit to top 10 if needed
            value_counts = data.value_counts()
            if len(value_counts) > 10:
                value_counts = value_counts.nlargest(10)
                
            # Create bar plot
            sns.barplot(x=value_counts.index.astype(str), y=value_counts.values)
            
            plt.title(f'Distribution of {column_name}')
            plt.xlabel(column_name)
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                plt.savefig(tmp.name)
                plt.close()
                return tmp.name
                
        except Exception as e:
            logger.warning(f"Error creating categorical plot: {str(e)}")
            plt.close()
            return None
            
    def _calculate_entropy(self, series):
        """Calculate entropy of a categorical column"""
        try:
            vc = series.value_counts(normalize=True, dropna=True)
            if len(vc) <= 1:
                return 0.0
            return float(-(vc * np.log2(vc)).sum())
        except Exception as e:
            logger.warning(f"Could not calculate entropy: {str(e)}")
            return 0.0
            
    def _run_correlation_analysis(self, parameters):
        """Run correlation analysis on the dataset"""
        logger.info("Running correlation analysis")
        
        # Get parameters or use defaults
        columns = parameters.get('columns', None)
        correlation_method = parameters.get('method', 'spearman')
        threshold = parameters.get('threshold', 0.7)
        
        if columns is None:
            # Use all numeric columns if none specified
            columns = [col for col, type in self.current_profile["column_types"].items() 
                     if type == "numeric"]
                     
        # We need at least 2 columns for correlation
        if len(columns) < 2:
            return {
                "analysis_type": "correlation_analysis",
                "error": "Need at least 2 numeric columns for correlation analysis",
                "columns_found": len(columns)
            }
            
        results = {
            "analysis_type": "correlation_analysis",
            "correlation_pairs": [],
            "correlation_matrix": {},
            "visualizations": {}
        }
        
        try:
            # Create correlation matrix
            corr_matrix = self.current_data[columns].corr(method=correlation_method)
            
            # Find highly correlated pairs
            for i in range(len(columns)):
                for j in range(i+1, len(columns)):
                    corr_value = abs(corr_matrix.iloc[i, j])
                    if corr_value >= threshold:
                        results["correlation_pairs"].append({
                            "col1": columns[i],
                            "col2": columns[j],
                            "correlation": float(corr_value),
                            "direction": "positive" if corr_matrix.iloc[i, j] > 0 else "negative"
                        })
            
            # Convert the matrix to a nested dictionary
            results["correlation_matrix"] = corr_matrix.to_dict()
            
            # Generate and save heatmap visualization
            fig_path = self._create_correlation_heatmap(corr_matrix)
            if fig_path:
                results["visualizations"]["heatmap"] = fig_path
                
            # Generate scatterplots for top correlations
            if results["correlation_pairs"]:
                # Sort by absolute correlation
                sorted_pairs = sorted(results["correlation_pairs"], 
                                    key=lambda x: abs(x["correlation"]), 
                                    reverse=True)
                                    
                # Create scatterplots for top 5 pairs
                for i, pair in enumerate(sorted_pairs[:5]):
                    fig_path = self._create_scatter_plot(
                        self.current_data[pair["col1"]],
                        self.current_data[pair["col2"]],
                        pair["col1"],
                        pair["col2"]
                    )
                    if fig_path:
                        results["visualizations"][f"scatter_{i}"] = fig_path
                        
        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            results["error"] = str(e)
            
        return results
        
    def _create_correlation_heatmap(self, corr_matrix):
        """Create and save a correlation heatmap"""
        try:
            plt.figure(figsize=(10, 8))
            
            # Create heatmap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', mask=mask, 
                      vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
            
            plt.title('Correlation Matrix')
            plt.tight_layout()
            
            # Save to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                plt.savefig(tmp.name)
                plt.close()
                return tmp.name
                
        except Exception as e:
            logger.warning(f"Error creating correlation heatmap: {str(e)}")
            plt.close()
            return None
            
    def _create_scatter_plot(self, x_data, y_data, x_label, y_label):
        """Create and save a scatter plot with regression line"""
        try:
            plt.figure(figsize=(8, 6))
            
            # Create scatter plot with regression line
            sns.regplot(x=x_data, y=y_data, scatter_kws={'alpha':0.5})
            
            plt.title(f'Relationship between {x_label} and {y_label}')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.tight_layout()
            
            # Save to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                plt.savefig(tmp.name)
                plt.close()
                return tmp.name
                
        except Exception as e:
            logger.warning(f"Error creating scatter plot: {str(e)}")
            plt.close()
            return None
            
    def _run_outlier_detection(self, parameters):
        """Run outlier detection on the dataset"""
        logger.info("Running outlier detection")
        
        # Get parameters or use defaults
        columns = parameters.get('columns', None)
        method = parameters.get('method', 'iqr')  # 'iqr' or 'isolation_forest'
        contamination = parameters.get('contamination', 0.05)
        
        if columns is None:
            # Use all numeric columns if none specified
            columns = [col for col, type in self.current_profile["column_types"].items() 
                     if type == "numeric"]
                     
        if not columns:
            return {
                "analysis_type": "outlier_detection",
                "error": "No numeric columns found for outlier detection"
            }
            
        results = {
            "analysis_type": "outlier_detection",
            "outliers_by_column": {},
            "outlier_indices": [],
            "visualizations": {}
        }
        
        try:
            # Outlier detection for each column using IQR method
            if method == 'iqr':
                for col in columns:
                    # Handle potential inf values
                    clean_data = self.current_data[col].replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if len(clean_data) > 0:
                        q1 = clean_data.quantile(0.25)
                        q3 = clean_data.quantile(0.75)
                        iqr = q3 - q1
                        
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        
                        # Find outliers
                        outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
                        
                        # Store results
                        results["outliers_by_column"][col] = {
                            "count": len(outliers),
                            "percentage": float(len(outliers) / len(clean_data) * 100),
                            "bounds": {
                                "lower": float(lower_bound),
                                "upper": float(upper_bound)
                            },
                            "extreme_values": {
                                "min": float(outliers.min()) if not outliers.empty else None,
                                "max": float(outliers.max()) if not outliers.empty else None
                            },
                            "indices": outliers.index.tolist()
                        }
                        
                        # Update overall outlier indices
                        results["outlier_indices"].extend(outliers.index.tolist())
                        
                        # Create boxplot
                        fig_path = self._create_boxplot(clean_data, col)
                        if fig_path:
                            results["visualizations"][col] = fig_path
                
                # Remove duplicates from overall outlier indices
                results["outlier_indices"] = list(set(results["outlier_indices"]))
            
            # Multivariate outlier detection using Isolation Forest
            elif method == 'isolation_forest':
                # Prepare clean data for all columns
                clean_data = self.current_data[columns].replace([np.inf, -np.inf], np.nan)
                clean_data = clean_data.dropna()
                
                if len(clean_data) > 0:
                    # Standardize the data
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(clean_data)
                    
                    # Fit Isolation Forest
                    iso_forest = IsolationForest(contamination=contamination, random_state=42)
                    outlier_labels = iso_forest.fit_predict(scaled_data)
                    
                    # -1 indicates outlier, 1 indicates inlier
                    outlier_indices = clean_data.index[outlier_labels == -1].tolist()
                    
                    # Store results
                    results["method"] = "isolation_forest"
                    results["outlier_indices"] = outlier_indices
                    results["outlier_count"] = len(outlier_indices)
                    results["outlier_percentage"] = float(len(outlier_indices) / len(clean_data) * 100)
                    
                    # Create a PCA plot if we have enough columns
                    if len(columns) >= 2:
                        fig_path = self._create_pca_outlier_plot(scaled_data, outlier_labels)
                        if fig_path:
                            results["visualizations"]["pca"] = fig_path
                            
        except Exception as e:
            logger.error(f"Error in outlier detection: {str(e)}")
            results["error"] = str(e)
            
        return results
        
    def _create_boxplot(self, data, column_name):
        """Create and save a boxplot for outlier visualization"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Create boxplot
            sns.boxplot(x=data)
            
            plt.title(f'Boxplot of {column_name} with Outliers')
            plt.xlabel(column_name)
            plt.tight_layout()
            
            # Save to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                plt.savefig(tmp.name)
                plt.close()
                return tmp.name
                
        except Exception as e:
            logger.warning(f"Error creating boxplot: {str(e)}")
            plt.close()
            return None
            
    def _create_pca_outlier_plot(self, scaled_data, outlier_labels):
        """Create and save a PCA plot to visualize multivariate outliers"""
        try:
            # Apply PCA to reduce to 2 dimensions
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            
            # Plot normal points and outliers with different colors
            plt.scatter(
                pca_result[outlier_labels == 1, 0],
                pca_result[outlier_labels == 1, 1],
                c='blue', label='Normal', alpha=0.5
            )
            plt.scatter(
                pca_result[outlier_labels == -1, 0],
                pca_result[outlier_labels == -1, 1],
                c='red', label='Outlier', alpha=0.7
            )
            
            plt.title('PCA Visualization of Outliers')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.legend()
            plt.tight_layout()
            
            # Save to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                plt.savefig(tmp.name)
                plt.close()
                return tmp.name
                
        except Exception as e:
            logger.warning(f"Error creating PCA outlier plot: {str(e)}")
            plt.close()
            return None
            
    def _run_hypothesis_testing(self, parameters):
        """Run hypothesis testing on columns"""
        logger.info("Running hypothesis testing")
        return {"analysis_type": "hypothesis_testing", "status": "Not fully implemented"}
        
    def _run_anova_analysis(self, parameters):
        """Run ANOVA analysis"""
        logger.info("Running ANOVA analysis")
        return {"analysis_type": "anova_analysis", "status": "Not fully implemented"}
        
    def _run_regression_analysis(self, parameters):
        """Run regression analysis"""
        logger.info("Running regression analysis")
        return {"analysis_type": "regression_analysis", "status": "Not fully implemented"}
        
    def _run_classification_analysis(self, parameters):
        """Run classification analysis"""
        logger.info("Running classification analysis")
        return {"analysis_type": "classification", "status": "Not fully implemented"}
        
    def _run_time_series_analysis(self, parameters):
        """Run time series analysis"""
        logger.info("Running time series analysis")
        return {"analysis_type": "time_series_forecasting", "status": "Not fully implemented"}
        
    def _run_optimization_analysis(self, parameters):
        """Run optimization analysis"""
        logger.info("Running optimization analysis")
        return {"analysis_type": "optimization_analysis", "status": "Not fully implemented"}
        
    def _run_what_if_analysis(self, parameters):
        """Run what-if analysis"""
        logger.info("Running what-if analysis")
        return {"analysis_type": "what_if_analysis", "status": "Not fully implemented"}
        
    def _run_root_cause_analysis(self, parameters):
        """Run root cause analysis"""
        logger.info("Running root cause analysis")
        return {"analysis_type": "root_cause_analysis", "status": "Not fully implemented"}
        
    def _run_comparative_analysis(self, parameters):
        """Run comparative analysis"""
        logger.info("Running comparative analysis")
        return {"analysis_type": "comparative_analysis", "status": "Not fully implemented"}