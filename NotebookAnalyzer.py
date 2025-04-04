import json
import re
import os
import pandas as pd
import numpy as np
import ast
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import Counter

logger = logging.getLogger(__name__)

class NotebookAnalyzer:
    """Analyzes Jupyter notebooks to extract analytical patterns and workflows"""
    
    def __init__(self):
        self.analyzed_notebooks = 0
        self.patterns_by_dataset_type = {}
        self.common_workflows = {}
        self.library_usage = Counter()
        self.visualization_techniques = Counter()
        self.preprocessing_steps = Counter()
        self.analysis_techniques = Counter()
        
        # Common library imports to track
        self.tracked_libraries = {
            'pandas': ['pd'],
            'numpy': ['np'],
            'matplotlib': ['plt', 'matplotlib'],
            'seaborn': ['sns'],
            'scipy': ['stats', 'scipy'],
            'scikit-learn': ['sklearn'],
            'statsmodels': ['sm'],
            'prophet': ['Prophet'],
            'xgboost': ['xgb'],
            'lightgbm': ['lgb'],
            'tensorflow': ['tf'],
            'pytorch': ['torch'],
            'nltk': ['nltk'],
            'gensim': ['gensim'],
            'spacy': ['spacy']
        }
        
        # Common analysis patterns to identify
        self.analysis_patterns = {
            'distribution_analysis': [
                r'\.hist\(',
                r'\.plot\(kind=[\'"]hist',
                r'sns\.distplot\(',
                r'sns\.histplot\(',
                r'plt\.hist\('
            ],
            'correlation_analysis': [
                r'\.corr\(',
                r'sns\.heatmap\(',
                r'\.corrwith\(',
                r'pearsonr\(',
                r'spearmanr\('
            ],
            'time_series_analysis': [
                r'\.resample\(',
                r'seasonal_decompose\(',
                r'ARIMA\(',
                r'SARIMAX\(',
                r'Prophet\(',
                r'\.rolling\('
            ],
            'regression_analysis': [
                r'LinearRegression\(',
                r'LogisticRegression\(',
                r'Ridge\(',
                r'Lasso\(',
                r'ElasticNet\(',
                r'\.fit\(',
                r'\.predict\('
            ],
            'classification_analysis': [
                r'RandomForestClassifier\(',
                r'SVC\(',
                r'KNeighborsClassifier\(',
                r'DecisionTreeClassifier\(',
                r'GradientBoostingClassifier\(',
                r'confusion_matrix\(',
                r'classification_report\(',
                r'roc_auc_score\('
            ],
            'clustering_analysis': [
                r'KMeans\(',
                r'DBSCAN\(',
                r'AgglomerativeClustering\(',
                r'silhouette_score\('
            ],
            'dimensionality_reduction': [
                r'PCA\(',
                r'TSNE\(',
                r'TruncatedSVD\('
            ],
            'text_analysis': [
                r'CountVectorizer\(',
                r'TfidfVectorizer\(',
                r'word_tokenize\(',
                r'stopwords\(',
                r'\.apply\(lambda.*token'
            ]
        }
        
        # Common preprocessing patterns
        self.preprocessing_patterns = {
            'missing_value_handling': [
                r'\.fillna\(',
                r'\.dropna\(',
                r'\.isna\(',
                r'\.isnull\(',
                r'Imputer\(',
                r'SimpleImputer\('
            ],
            'outlier_detection': [
                r'\.quantile\(',
                r'IQR',
                r'boxplot\(',
                r'IsolationForest\(',
                r'LocalOutlierFactor\('
            ],
            'feature_scaling': [
                r'StandardScaler\(',
                r'MinMaxScaler\(',
                r'RobustScaler\(',
                r'normalize\('
            ],
            'feature_engineering': [
                r'\.apply\(',
                r'\.map\(',
                r'get_dummies\(',
                r'OneHotEncoder\(',
                r'LabelEncoder\(',
                r'PolynomialFeatures\('
            ],
            'data_type_conversion': [
                r'\.astype\(',
                r'to_datetime\(',
                r'to_numeric\('
            ]
        }
        
        # Common visualization patterns
        self.visualization_patterns = {
            'bar_plot': [
                r'\.plot\(kind=[\'"]bar',
                r'plt\.bar\(',
                r'sns\.barplot\('
            ],
            'line_plot': [
                r'\.plot\(',
                r'plt\.plot\(',
                r'sns\.lineplot\('
            ],
            'scatter_plot': [
                r'\.plot\(kind=[\'"]scatter',
                r'plt\.scatter\(',
                r'sns\.scatterplot\('
            ],
            'box_plot': [
                r'\.plot\(kind=[\'"]box',
                r'plt\.boxplot\(',
                r'sns\.boxplot\('
            ],
            'heatmap': [
                r'sns\.heatmap\('
            ],
            'pair_plot': [
                r'sns\.pairplot\('
            ],
            'facet_grid': [
                r'sns\.FacetGrid\('
            ]
        }
    
    def analyze_notebook(self, notebook_path: str, dataset_type: Optional[str] = None) -> Dict[str, Any]:
        """Analyze a single Jupyter notebook"""
        logger.info(f"Analyzing notebook: {notebook_path}")
        
        try:
            # Load the notebook
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook_content = json.load(f)
            
            # Extract cells
            cells = notebook_content.get('cells', [])
            
            # Process the notebook
            result = self._process_notebook(cells, notebook_path)
            
            # Store patterns by dataset type if provided
            if dataset_type:
                if dataset_type not in self.patterns_by_dataset_type:
                    self.patterns_by_dataset_type[dataset_type] = []
                self.patterns_by_dataset_type[dataset_type].append(result)
            
            # Update global counters
            self._update_global_counters(result)
            
            self.analyzed_notebooks += 1
            logger.info(f"Successfully analyzed notebook: {notebook_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing notebook {notebook_path}: {str(e)}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def _process_notebook(self, cells: List[Dict], notebook_path: str) -> Dict[str, Any]:
        """Process cells from a notebook"""
        result = {
            "imports": [],
            "dataframes": [],
            "columns_used": set(),
            "preprocessing_steps": [],
            "analysis_techniques": [],
            "visualization_techniques": [],
            "model_types": [],
            "workflow_sequence": [],
            "source_path": notebook_path
        }
        
        # Extract code from all code cells
        code_cells = [cell['source'] for cell in cells if cell['cell_type'] == 'code']
        full_code = '\n'.join([''.join(cell) if isinstance(cell, list) else cell for cell in code_cells])
        
        # Extract imports
        result["imports"] = self._extract_imports(full_code)
        
        # Extract dataframes and columns
        dataframes, columns = self._extract_dataframes_and_columns(full_code)
        result["dataframes"] = list(dataframes)
        result["columns_used"] = list(columns)
        
        # Extract preprocessing steps
        result["preprocessing_steps"] = self._extract_patterns(full_code, self.preprocessing_patterns)
        
        # Extract analysis techniques
        result["analysis_techniques"] = self._extract_patterns(full_code, self.analysis_patterns)
        
        # Extract visualization techniques
        result["visualization_techniques"] = self._extract_patterns(full_code, self.visualization_patterns)
        
        # Extract model types
        result["model_types"] = self._extract_model_types(full_code)
        
        # Determine workflow sequence
        result["workflow_sequence"] = self._determine_workflow_sequence(
            result["preprocessing_steps"], 
            result["analysis_techniques"],
            result["visualization_techniques"]
        )
        
        return result
    
    def _extract_imports(self, code: str) -> List[str]:
        """Extract library imports from code"""
        imports = []
        
        # Match import statements
        import_patterns = [
            r'^import\s+(\w+)(?:\s+as\s+(\w+))?',
            r'^from\s+(\w+)(?:\.\w+)?\s+import'
        ]
        
        for pattern in import_patterns:
            matches = re.finditer(pattern, code, re.MULTILINE)
            for match in matches:
                if match.group(1):
                    imports.append(match.group(1))
        
        return imports
    
    def _extract_dataframes_and_columns(self, code: str) -> Tuple[Set[str], Set[str]]:
        """Extract dataframe names and column references"""
        dataframes = set()
        columns = set()
        
        # Find dataframe creation patterns
        df_creation_patterns = [
            r'(\w+)\s*=\s*pd\.read_csv\(',
            r'(\w+)\s*=\s*pd\.read_excel\(',
            r'(\w+)\s*=\s*pd\.read_parquet\(',
            r'(\w+)\s*=\s*pd\.DataFrame\('
        ]
        
        for pattern in df_creation_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                if match.group(1):
                    dataframes.add(match.group(1))
        
        # Find column references for identified dataframes
        for df in dataframes:
            # Match patterns like df['column'] or df.column
            col_patterns = [
                rf'{df}\[[\'"](\w+?)[\'"]\]',
                rf'{df}\.(\w+)\b(?!\()'  # Avoid matching methods
            ]
            
            for pattern in col_patterns:
                matches = re.finditer(pattern, code)
                for match in matches:
                    if match.group(1) and not match.group(1).startswith('_'):
                        columns.add(match.group(1))
        
        return dataframes, columns
    
    def _extract_patterns(self, code: str, pattern_dict: Dict[str, List[str]]) -> List[str]:
        """Extract patterns from code based on regex patterns"""
        found_patterns = []
        
        for pattern_name, patterns in pattern_dict.items():
            for regex in patterns:
                if re.search(regex, code):
                    found_patterns.append(pattern_name)
                    break  # Found one pattern for this category, move to next
        
        return found_patterns
    
    def _extract_model_types(self, code: str) -> List[str]:
        """Extract model types used in the notebook"""
        models = []
        
        # Common model patterns
        model_patterns = {
            'linear_regression': r'LinearRegression\(',
            'logistic_regression': r'LogisticRegression\(',
            'decision_tree': r'DecisionTreeClassifier\(|DecisionTreeRegressor\(',
            'random_forest': r'RandomForestClassifier\(|RandomForestRegressor\(',
            'svm': r'SVC\(|SVR\(',
            'knn': r'KNeighborsClassifier\(|KNeighborsRegressor\(',
            'gradient_boosting': r'GradientBoostingClassifier\(|GradientBoostingRegressor\(|XGBClassifier\(|XGBRegressor\(',
            'neural_network': r'MLPClassifier\(|MLPRegressor\(|keras|Sequential\(|Dense\(',
            'clustering': r'KMeans\(|DBSCAN\(|AgglomerativeClustering\(',
            'time_series': r'ARIMA\(|SARIMAX\(|Prophet\('
        }
        
        for model_name, pattern in model_patterns.items():
            if re.search(pattern, code):
                models.append(model_name)
        
        return models
    
    def _determine_workflow_sequence(self, 
                                    preprocessing: List[str], 
                                    analysis: List[str], 
                                    visualization: List[str]) -> List[str]:
        """Create a simplified workflow sequence"""
        workflow = []
        
        # Add preprocessing steps
        if preprocessing:
            workflow.append("data_preparation")
            
            # Add specific preprocessing categories
            for step in preprocessing:
                category = step.split('_')[0]  # e.g., 'missing' from 'missing_value_handling'
                if category and f"{category}_processing" not in workflow:
                    workflow.append(f"{category}_processing")
        
        # Add exploratory analysis if visualization comes before modeling
        if visualization:
            workflow.append("exploratory_analysis")
        
        # Add modeling steps
        modeling_techniques = [
            "regression_analysis", 
            "classification_analysis", 
            "clustering_analysis",
            "dimensionality_reduction",
            "time_series_analysis"
        ]
        
        for technique in analysis:
            if technique in modeling_techniques:
                workflow.append("modeling")
                break
        
        # Add specific analysis types
        for technique in analysis:
            if technique not in workflow:
                workflow.append(technique)
        
        # Add visualization if it wasn't added earlier
        if visualization and "exploratory_analysis" not in workflow:
            workflow.append("results_visualization")
        
        return workflow
    
    def _update_global_counters(self, result: Dict[str, Any]) -> None:
        """Update global counters based on notebook analysis"""
        # Update library usage
        for lib in result["imports"]:
            self.library_usage[lib] += 1
        
        # Update preprocessing counters
        for step in result["preprocessing_steps"]:
            self.preprocessing_steps[step] += 1
        
        # Update analysis technique counters
        for technique in result["analysis_techniques"]:
            self.analysis_techniques[technique] += 1
        
        # Update visualization technique counters
        for vis in result["visualization_techniques"]:
            self.visualization_techniques[vis] += 1
    
    def analyze_directory(self, directory_path: str, dataset_type_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Analyze all notebooks in a directory"""
        logger.info(f"Analyzing notebooks in directory: {directory_path}")
        
        results = {
            "analyzed_count": 0,
            "failed_count": 0,
            "notebooks": []
        }
        
        # Get all notebook files
        notebook_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.ipynb'):
                    notebook_files.append(os.path.join(root, file))
        
        # Process each notebook
        for notebook_path in notebook_files:
            dataset_type = None
            if dataset_type_mapping:
                # Extract filename or pattern to determine dataset type
                filename = os.path.basename(notebook_path)
                for pattern, dtype in dataset_type_mapping.items():
                    if pattern in filename:
                        dataset_type = dtype
                        break
                        
            result = self.analyze_notebook(notebook_path, dataset_type)
            
            if "error" in result:
                results["failed_count"] += 1
            else:
                results["analyzed_count"] += 1
                results["notebooks"].append({
                    "path": notebook_path,
                    "dataset_type": dataset_type,
                    "preprocessing": result["preprocessing_steps"],
                    "analysis": result["analysis_techniques"],
                    "visualization": result["visualization_techniques"],
                    "models": result["model_types"],
                    "workflow": result["workflow_sequence"]
                })
        
        # Generate summary statistics
        if results["analyzed_count"] > 0:
            results["summary"] = self._generate_summary()
        
        logger.info(f"Completed analysis of {results['analyzed_count']} notebooks " 
                  f"({results['failed_count']} failed)")
        return results
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from analyzed notebooks"""
        summary = {
            "total_notebooks": self.analyzed_notebooks,
            "top_libraries": dict(self.library_usage.most_common(10)),
            "top_preprocessing": dict(self.preprocessing_steps.most_common(10)),
            "top_analyses": dict(self.analysis_techniques.most_common(10)),
            "top_visualizations": dict(self.visualization_techniques.most_common(10)),
            "dataset_type_patterns": {}
        }
        
        # Generate patterns by dataset type
        for dataset_type, notebooks in self.patterns_by_dataset_type.items():
            type_summary = {
                "count": len(notebooks),
                "preprocessing": Counter(),
                "analysis": Counter(),
                "visualization": Counter()
            }
            
            # Collect patterns
            for notebook in notebooks:
                for step in notebook["preprocessing_steps"]:
                    type_summary["preprocessing"][step] += 1
                for technique in notebook["analysis_techniques"]:
                    type_summary["analysis"][technique] += 1
                for vis in notebook["visualization_techniques"]:
                    type_summary["visualization"][vis] += 1
            
            # Convert to dictionaries
            type_summary["preprocessing"] = dict(type_summary["preprocessing"].most_common(5))
            type_summary["analysis"] = dict(type_summary["analysis"].most_common(5))
            type_summary["visualization"] = dict(type_summary["visualization"].most_common(5))
            
            summary["dataset_type_patterns"][dataset_type] = type_summary
        
        # Generate common workflows
        workflows = Counter()
        for dataset_type, notebooks in self.patterns_by_dataset_type.items():
            for notebook in notebooks:
                workflow_str = " → ".join(notebook["workflow_sequence"])
                workflows[workflow_str] += 1
        
        summary["common_workflows"] = dict(workflows.most_common(5))
        
        return summary
    
    def extract_templates(self) -> Dict[str, Any]:
        """Extract analysis templates based on dataset types"""
        templates = {}
        
        for dataset_type, notebooks in self.patterns_by_dataset_type.items():
            # Skip if too few examples
            if len(notebooks) < 2:
                continue
                
            # Collect all steps and their frequencies
            preprocessing = Counter()
            analysis = Counter()
            visualization = Counter()
            models = Counter()
            
            for notebook in notebooks:
                for step in notebook["preprocessing_steps"]:
                    preprocessing[step] += 1
                for technique in notebook["analysis_techniques"]:
                    analysis[technique] += 1
                for vis in notebook["visualization_techniques"]:
                    visualization[vis] += 1
                for model in notebook["model_types"]:
                    models[model] += 1
            
            # Create template with most common techniques
            templates[dataset_type] = {
                "preprocessing": [item for item, _ in preprocessing.most_common(3)],
                "analysis": [item for item, _ in analysis.most_common(3)],
                "visualization": [item for item, _ in visualization.most_common(3)],
                "recommended_models": [item for item, _ in models.most_common(2)],
                "based_on_notebooks": len(notebooks)
            }
            
            # Add example notebook paths
            templates[dataset_type]["example_notebooks"] = [
                notebook["source_path"] for notebook in notebooks[:2]
            ]
        
        return templates
    
    def recommend_analysis(self, data_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend analysis techniques based on data profile and learned patterns"""
        recommendations = {
            "preprocessing": [],
            "analysis": [],
            "visualization": [],
            "models": [],
            "confidence": 0.0,
            "explanation": []
        }
        
        # Determine dataset type
        dataset_type = self._determine_dataset_type(data_profile)
        recommendations["detected_type"] = dataset_type
        
        # Check if we have templates for this dataset type
        templates = self.extract_templates()
        if dataset_type in templates:
            template = templates[dataset_type]
            recommendations["preprocessing"] = template["preprocessing"]
            recommendations["analysis"] = template["analysis"]
            recommendations["visualization"] = template["visualization"]
            recommendations["models"] = template["recommended_models"]
            recommendations["confidence"] = 0.8  # High confidence when we have a matching template
            recommendations["explanation"].append(
                f"Recommendations based on analysis patterns from {template['based_on_notebooks']} similar notebooks"
            )
        else:
            # Fallback to general recommendations based on data characteristics
            recommendations.update(self._generate_fallback_recommendations(data_profile))
            recommendations["confidence"] = 0.5  # Medium confidence for fallback recommendations
            recommendations["explanation"].append(
                "Recommendations based on general data characteristics (no exact template match)"
            )
        
        # Add data-specific explanations
        self._add_recommendation_explanations(recommendations, data_profile)
        
        return recommendations
    
    def _determine_dataset_type(self, data_profile: Dict[str, Any]) -> str:
        """Determine the most likely dataset type based on profile"""
        # Check if the profile already includes a dataset type classification
        if "metadata" in data_profile and "dataset_type" in data_profile["metadata"]:
            dataset_types = data_profile["metadata"]["dataset_type"]
            if dataset_types:
                # Return the dataset type with highest score
                return max(dataset_types.items(), key=lambda x: x[1])[0]
        
        # Check for time series characteristics
        temporal_patterns = data_profile.get("temporal_patterns", {})
        if temporal_patterns and len(temporal_patterns) > 0:
            return "time_series"
        
        # Check column distribution
        column_types = data_profile.get("column_types", {})
        type_counts = {}
        for col_type in column_types.values():
            type_counts[col_type] = type_counts.get(col_type, 0) + 1
        
        # Check categorical vs numerical balance
        total_cols = len(column_types)
        if total_cols > 0:
            categorical_count = type_counts.get("categorical", 0) + type_counts.get("binary", 0)
            numerical_count = type_counts.get("numeric", 0) + type_counts.get("continuous", 0)
            
            if categorical_count / total_cols > 0.7:
                return "categorical_heavy"
            elif numerical_count / total_cols > 0.7:
                return "numerical_heavy"
        
        # Check for geographical data
        if "geographical_data" in data_profile and data_profile["geographical_data"]:
            return "geospatial"
            
        # Default to generic type
        return "generic"
    
    def _generate_fallback_recommendations(self, data_profile: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate recommendations when no template matches"""
        recommendations = {
            "preprocessing": [],
            "analysis": [],
            "visualization": [],
            "models": []
        }
        
        # Add missing value handling if needed
        missing_data = data_profile.get("missing_data", {}).get("missing_percentages", {})
        if any(pct > 5 for pct in missing_data.values()):
            recommendations["preprocessing"].append("missing_value_handling")
        
        # Add outlier detection if needed
        has_outliers = False
        for issue in data_profile.get("data_quality_issues", []):
            if issue.get("type") == "potential_outliers":
                has_outliers = True
                break
        
        if has_outliers:
            recommendations["preprocessing"].append("outlier_detection")
        
        # Add feature scaling for numerical data
        column_types = data_profile.get("column_types", {})
        numerical_cols = [col for col, type in column_types.items() 
                         if type in ["numeric", "continuous"]]
        
        if numerical_cols:
            recommendations["preprocessing"].append("feature_scaling")
            
            # Add correlation analysis for numerical data
            recommendations["analysis"].append("correlation_analysis")
            recommendations["visualization"].append("heatmap")
            
            # Add distribution analysis
            recommendations["analysis"].append("distribution_analysis")
            recommendations["visualization"].append("box_plot")
        
        # Add encoding for categorical data
        categorical_cols = [col for col, type in column_types.items() 
                           if type in ["categorical", "binary", "ordinal"]]
        
        if categorical_cols:
            recommendations["preprocessing"].append("feature_engineering")
            
            # Add categorical analysis
            recommendations["visualization"].append("bar_plot")
        
        # Recommend models based on column types
        datetime_cols = [col for col, type in column_types.items() if type == "datetime"]
        if datetime_cols:
            recommendations["analysis"].append("time_series_analysis")
            recommendations["models"].append("time_series")
        elif categorical_cols and numerical_cols:
            target_candidates = data_profile.get("column_suggestions", {}).get("target_column_candidates", [])
            if target_candidates:
                target_type = next((t["type"] for t in target_candidates), None)
                if target_type in ["categorical", "binary"]:
                    recommendations["analysis"].append("classification_analysis")
                    recommendations["models"].append("random_forest")
                else:
                    recommendations["analysis"].append("regression_analysis")
                    recommendations["models"].append("linear_regression")
        
        return recommendations
    
    def _add_recommendation_explanations(self, recommendations: Dict[str, Any], data_profile: Dict[str, Any]) -> None:
        """Add explanations for why each technique is recommended"""
        # Explain preprocessing recommendations
        for technique in recommendations["preprocessing"]:
            if technique == "missing_value_handling":
                missing_data = data_profile.get("missing_data", {}).get("missing_percentages", {})
                if missing_data:
                    max_missing = max(missing_data.values())
                    recommendations["explanation"].append(
                        f"Missing value handling recommended because up to {max_missing:.1f}% of values are missing in some columns"
                    )
            
            elif technique == "outlier_detection":
                for issue in data_profile.get("data_quality_issues", []):
                    if issue.get("type") == "potential_outliers":
                        recommendations["explanation"].append(
                            f"Outlier detection recommended for column '{issue.get('column')}': {issue.get('description')}"
                        )
                        break
            
            elif technique == "feature_scaling":
                recommendations["explanation"].append(
                    "Feature scaling recommended for numerical columns to normalize ranges"
                )
                
            elif technique == "feature_engineering":
                recommendations["explanation"].append(
                    "Feature engineering (encoding) recommended for categorical columns"
                )
        
        # Explain analysis recommendations
        for technique in recommendations["analysis"]:
            if technique == "time_series_analysis":
                datetime_cols = [col for col, type in data_profile.get("column_types", {}).items() 
                               if type == "datetime"]
                recommendations["explanation"].append(
                    f"Time series analysis recommended because dataset contains datetime column(s): {', '.join(datetime_cols[:3])}"
                )
            
            elif technique == "correlation_analysis":
                high_correlations = data_profile.get("potential_relationships", {}).get("high_correlations", [])
                if high_correlations:
                    recommendations["explanation"].append(
                        f"Correlation analysis recommended because dataset has {len(high_correlations)} strong correlations between variables"
                    )
            
            elif technique == "classification_analysis":
                target_candidates = data_profile.get("column_suggestions", {}).get("target_column_candidates", [])
                if target_candidates:
                    categorical_targets = [t["column"] for t in target_candidates 
                                         if t["type"] in ["categorical", "binary"]]
                    if categorical_targets:
                        recommendations["explanation"].append(
                            f"Classification analysis recommended for potential categorical target(s): {', '.join(categorical_targets[:2])}"
                        )
            
            elif technique == "regression_analysis":
                target_candidates = data_profile.get("column_suggestions", {}).get("target_column_candidates", [])
                if target_candidates:
                    numeric_targets = [t["column"] for t in target_candidates 
                                     if t["type"] in ["numeric", "continuous"]]
                    if numeric_targets:
                        recommendations["explanation"].append(
                            f"Regression analysis recommended for potential numeric target(s): {', '.join(numeric_targets[:2])}"
                        )
        
        # Explain visualization recommendations
        if "bar_plot" in recommendations["visualization"]:
            categorical_cols = [col for col, type in data_profile.get("column_types", {}).items() 
                               if type in ["categorical", "binary"]]
            if categorical_cols:
                recommendations["explanation"].append(
                    "Bar plots recommended for visualizing categorical variables"
                )
                
        if "box_plot" in recommendations["visualization"]:
            recommendations["explanation"].append(
                "Box plots recommended for examining distributions and identifying outliers"
            )
            
        if "heatmap" in recommendations["visualization"]:
            recommendations["explanation"].append(
                "Heatmap recommended for visualizing correlation matrix"
            )
    
    def build_analysis_pipeline(self, data_profile: Dict[str, Any], objective: str = "general") -> Dict[str, Any]:
        """Build a complete analysis pipeline based on profile and recommendations"""
        # Get recommendations
        recommendations = self.recommend_analysis(data_profile)
        
        # Adjust based on objective
        self._adjust_for_objective(recommendations, objective)
        
        # Create pipeline steps
        pipeline = {
            "name": f"Analysis pipeline for {objective} objective",
            "objective": objective,
            "confidence": recommendations["confidence"],
            "steps": []
        }
        
        # Add data loading step
        pipeline["steps"].append({
            "name": "data_loading",
            "description": "Load and validate the dataset",
            "code_template": "df = pd.read_csv('${file_path}')",
            "parameters": ["file_path"]
        })
        
        # Add preprocessing steps
        self._add_preprocessing_steps(pipeline, recommendations, data_profile)
        
        # Add analysis steps
        self._add_analysis_steps(pipeline, recommendations, data_profile, objective)
        
        # Add visualization steps
        self._add_visualization_steps(pipeline, recommendations, data_profile)
        
        # Add evaluation/interpretation steps
        self._add_evaluation_steps(pipeline, recommendations, objective)
        
        return pipeline
    
    def _adjust_for_objective(self, recommendations: Dict[str, Any], objective: str) -> None:
        """Adjust recommendations based on analysis objective"""
        if objective == "prediction":
            # Focus on predictive modeling
            recommendations["analysis"] = [t for t in recommendations["analysis"] 
                                         if t in ["regression_analysis", "classification_analysis", 
                                                 "time_series_analysis"]]
            if not recommendations["analysis"]:
                recommendations["analysis"] = ["regression_analysis"]
                
            # Add cross-validation
            if "cross_validation" not in recommendations["preprocessing"]:
                recommendations["preprocessing"].append("cross_validation")
                
        elif objective == "segmentation":
            # Focus on clustering and grouping
            if "clustering_analysis" not in recommendations["analysis"]:
                recommendations["analysis"].append("clustering_analysis")
                
            if "dimensionality_reduction" not in recommendations["analysis"]:
                recommendations["analysis"].append("dimensionality_reduction")
                
            # Ensure we have feature scaling
            if "feature_scaling" not in recommendations["preprocessing"]:
                recommendations["preprocessing"].append("feature_scaling")
                
        elif objective == "exploration":
            # Focus on EDA
            if "distribution_analysis" not in recommendations["analysis"]:
                recommendations["analysis"].append("distribution_analysis")
                
            if "correlation_analysis" not in recommendations["analysis"] and any(
                col_type in ["numeric", "continuous"] 
                for col_type in data_profile.get("column_types", {}).values()
            ):
                recommendations["analysis"].append("correlation_analysis")
                
            # Add more visualization
            for viz in ["bar_plot", "box_plot", "scatter_plot"]:
                if viz not in recommendations["visualization"]:
                    recommendations["visualization"].append(viz)
    
    def _add_preprocessing_steps(self, pipeline: Dict[str, Any], 
                               recommendations: Dict[str, Any], 
                               data_profile: Dict[str, Any]) -> None:
        """Add preprocessing steps to the pipeline"""
        # Add initial data overview
        pipeline["steps"].append({
            "name": "data_overview",
            "description": "Get a basic overview of the dataset",
            "code_template": "print(f'Dataset shape: {df.shape}')\ndf.info()\ndf.head()",
            "parameters": []
        })
        
        # Add steps for each recommended preprocessing technique
        for technique in recommendations["preprocessing"]:
            if technique == "missing_value_handling":
                pipeline["steps"].append({
                    "name": "handle_missing_values",
                    "description": "Identify and handle missing values",
                    "code_template": "# Check missing values\nmissing_values = df.isnull().sum()\nprint(missing_values[missing_values > 0])\n\n# Handle missing values\n${strategy}",
                    "parameters": ["strategy"],
                    "options": {
                        "strategy": [
                            "# Drop rows with missing values\ndf = df.dropna()",
                            "# Fill missing values with mean/median/mode\nfor col in df.select_dtypes(include=['number']).columns:\n    df[col] = df[col].fillna(df[col].median())\nfor col in df.select_dtypes(exclude=['number']).columns:\n    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')",
                            "# Use more advanced imputation\nfrom sklearn.impute import SimpleImputer\nnum_imputer = SimpleImputer(strategy='median')\ncat_imputer = SimpleImputer(strategy='most_frequent')\n# Apply to numerical and categorical columns separately"
                        ]
                    }
                })
            
            elif technique == "outlier_detection":
                pipeline["steps"].append({
                    "name": "handle_outliers",
                    "description": "Detect and handle outliers",
                    "code_template": "# Detect outliers using IQR method\nfor col in df.select_dtypes(include=['number']).columns:\n    Q1 = df[col].quantile(0.25)\n    Q3 = df[col].quantile(0.75)\n    IQR = Q3 - Q1\n    lower_bound = Q1 - 1.5 * IQR\n    upper_bound = Q3 + 1.5 * IQR\n    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]\n    print(f'Column {col}: {len(outliers)} outliers detected')\n\n# Handle outliers\n${strategy}",
                    "parameters": ["strategy"],
                    "options": {
                        "strategy": [
                            "# Cap outliers at boundaries\nfor col in df.select_dtypes(include=['number']).columns:\n    Q1 = df[col].quantile(0.25)\n    Q3 = df[col].quantile(0.75)\n    IQR = Q3 - Q1\n    lower_bound = Q1 - 1.5 * IQR\n    upper_bound = Q3 + 1.5 * IQR\n    df[col] = df[col].clip(lower_bound, upper_bound)",
                            "# Remove outlier rows\n# Note: Only do this if outliers are truly invalid data\n# df = df[~((df[col] < lower_bound) | (df[col] > upper_bound))]",
                            "# Use robust methods instead of removing outliers\n# (No action needed here, just use robust methods in analysis)"
                        ]
                    }
                })
            
            elif technique == "feature_scaling":
                pipeline["steps"].append({
                    "name": "scale_features",
                    "description": "Scale numerical features",
                    "code_template": "# Scale numerical features\nfrom sklearn.preprocessing import ${scaler_type}\nscaler = ${scaler_type}()\nnumeric_cols = df.select_dtypes(include=['number']).columns\ndf[numeric_cols] = scaler.fit_transform(df[numeric_cols])",
                    "parameters": ["scaler_type"],
                    "options": {
                        "scaler_type": [
                            "StandardScaler",
                            "MinMaxScaler",
                            "RobustScaler"
                        ]
                    }
                })
            
            elif technique == "feature_engineering":
                categorical_cols = [col for col, type in data_profile.get("column_types", {}).items() 
                                  if type in ["categorical", "binary", "ordinal"]]
                
                if categorical_cols:
                    cols_str = ", ".join([f"'{col}'" for col in categorical_cols[:5]])
                    pipeline["steps"].append({
                        "name": "encode_categorical",
                        "description": "Encode categorical features",
                        "code_template": "# Encode categorical features\n${encoding_strategy}\n",
                        "parameters": ["encoding_strategy"],
                        "options": {
                            "encoding_strategy": [
                                f"# One-hot encoding\ncategorical_cols = [{cols_str}]\ndf = pd.get_dummies(df, columns=categorical_cols, drop_first=True)",
                                f"# Label encoding\nfrom sklearn.preprocessing import LabelEncoder\ncategorical_cols = [{cols_str}]\nle = LabelEncoder()\nfor col in categorical_cols:\n    df[col] = le.fit_transform(df[col].astype(str))"
                            ]
                        }
                    })
            
            elif technique == "cross_validation":
                pipeline["steps"].append({
                    "name": "prepare_validation",
                    "description": "Set up validation framework",
                    "code_template": "# Split data into training and test sets\nfrom sklearn.model_selection import train_test_split\nX = df.drop('${target_column}', axis=1)\ny = df['${target_column}']\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)",
                    "parameters": ["target_column"],
                    "options": {
                        "target_column": [c["column"] for c in data_profile.get("column_suggestions", {}).get("target_column_candidates", [])]
                    }
                })
    
    def _add_analysis_steps(self, pipeline: Dict[str, Any], 
                          recommendations: Dict[str, Any], 
                          data_profile: Dict[str, Any],
                          objective: str) -> None:
        """Add analysis steps to the pipeline"""
        for technique in recommendations["analysis"]:
            if technique == "distribution_analysis":
                pipeline["steps"].append({
                    "name": "analyze_distributions",
                    "description": "Analyze variable distributions",
                    "code_template": "# Analyze distributions\n# Numerical variables\nfor col in df.select_dtypes(include=['number']).columns[:5]:\n    print(f'\\nSummary statistics for {col}:')\n    print(df[col].describe())\n\n# Categorical variables\nfor col in df.select_dtypes(exclude=['number']).columns[:5]:\n    print(f'\\nValue counts for {col}:')\n    print(df[col].value_counts(normalize=True))",
                    "parameters": []
                })
            
            elif technique == "correlation_analysis":
                pipeline["steps"].append({
                    "name": "analyze_correlations",
                    "description": "Analyze correlations between variables",
                    "code_template": "# Calculate correlation matrix\ncorr_matrix = df.select_dtypes(include=['number']).corr(method='${corr_method}')\n\n# Display top correlations\nprint('\\nTop correlations:')\ntop_corr = corr_matrix.unstack().sort_values(ascending=False)\ntop_corr = top_corr[top_corr < 1].head(10)\nprint(top_corr)",
                    "parameters": ["corr_method"],
                    "options": {
                        "corr_method": ["pearson", "spearman"]
                    }
                })
            
            elif technique == "time_series_analysis":
                datetime_cols = [col for col, type in data_profile.get("column_types", {}).items() 
                               if type == "datetime"]
                if datetime_cols:
                    pipeline["steps"].append({
                        "name": "time_series_analysis",
                        "description": "Analyze time series data",
                        "code_template": "# Ensure datetime column is properly formatted\ndf['${date_column}'] = pd.to_datetime(df['${date_column}'])\ndf = df.set_index('${date_column}')\n\n# Resample to regular frequency if needed\n${resample_code}\n\n# Plot time series\ntarget_col = '${target_column}'\nplt.figure(figsize=(12, 6))\nplt.plot(df[target_col])\nplt.title(f'Time Series: {target_col}')\nplt.xlabel('Date')\nplt.ylabel(target_col)\nplt.grid(True)\n\n# Check for stationarity\nfrom statsmodels.tsa.stattools import adfuller\nadf_result = adfuller(df[target_col].dropna())\nprint(f'ADF Statistic: {adf_result[0]}')\nprint(f'p-value: {adf_result[1]}')",
                        "parameters": ["date_column", "target_column", "resample_code"],
                        "options": {
                            "date_column": datetime_cols,
                            "target_column": [col for col, type in data_profile.get("column_types", {}).items() 
                                           if type in ["numeric", "continuous"]],
                            "resample_code": [
                                "# No resampling needed", 
                                "# Resample to daily frequency\ndf = df.resample('D').mean()", 
                                "# Resample to weekly frequency\ndf = df.resample('W').mean()",
                                "# Resample to monthly frequency\ndf = df.resample('M').mean()"
                            ]
                        }
                    })
                    
                    # Add forecasting if that's the objective
                    if objective == "prediction":
                        pipeline["steps"].append({
                            "name": "time_series_forecasting",
                            "description": "Forecast future values",
                            "code_template": "# Time series forecasting\nfrom statsmodels.tsa.arima.model import ARIMA\n\n# Prepare train/test split\ntrain_size = int(len(df) * 0.8)\ntrain, test = df[:train_size], df[train_size:]\nprint(f'Training data size: {len(train)}, Test data size: {len(test)}')\n\n# Fit ARIMA model\ntarget_col = '${target_column}'\nmodel = ARIMA(train[target_col], order=${arima_order})\nmodel_fit = model.fit()\n\n# Forecast\nforecast = model_fit.forecast(steps=len(test))\n\n# Evaluate forecasts\nfrom sklearn.metrics import mean_squared_error, mean_absolute_error\nimport numpy as np\nrmse = np.sqrt(mean_squared_error(test[target_col], forecast))\nmae = mean_absolute_error(test[target_col], forecast)\nprint(f'RMSE: {rmse}')\nprint(f'MAE: {mae}')",
                            "parameters": ["target_column", "arima_order"],
                            "options": {
                                "target_column": [col for col, type in data_profile.get("column_types", {}).items() 
                                               if type in ["numeric", "continuous"]],
                                "arima_order": ["(1, 1, 1)", "(1, 0, 1)", "(2, 1, 2)"]
                            }
                        })
            
            elif technique == "regression_analysis":
                # Find numeric target candidates
                target_candidates = [c["column"] for c in data_profile.get("column_suggestions", {}).get("target_column_candidates", [])
                                   if c["type"] in ["numeric", "continuous"]]
                
                if target_candidates:
                    pipeline["steps"].append({
                        "name": "regression_analysis",
                        "description": "Build regression model",
                        "code_template": "# Regression analysis\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import ${model_type}\nfrom sklearn.metrics import mean_squared_error, r2_score\nfrom sklearn.preprocessing import StandardScaler\n\n# Prepare data\nX = df.drop('${target_column}', axis=1)\ny = df['${target_column}']\n\n# Handle non-numeric features\nX = pd.get_dummies(X)\n\n# Split the data\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\n# Scale features\nscaler = StandardScaler()\nX_train_scaled = scaler.fit_transform(X_train)\nX_test_scaled = scaler.transform(X_test)\n\n# Build and evaluate model\nmodel = ${model_type}()\nmodel.fit(X_train_scaled, y_train)\ny_pred = model.predict(X_test_scaled)\n\n# Evaluate the model\nmse = mean_squared_error(y_test, y_pred)\nr2 = r2_score(y_test, y_pred)\nprint(f'Mean Squared Error: {mse}')\nprint(f'R² Score: {r2}')\n\n# Feature importance (if available)\nif hasattr(model, 'coef_'):\n    coefficients = model.coef_\n    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': coefficients})\n    feature_importance = feature_importance.sort_values('Importance', key=abs, ascending=False)\n    print('\\nFeature Importance:')\n    print(feature_importance.head(10))",
                        "parameters": ["model_type", "target_column"],
                        "options": {
                            "model_type": ["LinearRegression", "Ridge", "Lasso", "ElasticNet"],
                            "target_column": target_candidates
                        }
                    })
            
            elif technique == "classification_analysis":
                # Find categorical target candidates
                target_candidates = [c["column"] for c in data_profile.get("column_suggestions", {}).get("target_column_candidates", [])
                                   if c["type"] in ["categorical", "binary"]]
                
                if target_candidates:
                    pipeline["steps"].append({
                        "name": "classification_analysis",
                        "description": "Build classification model",
                        "code_template": "# Classification analysis\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import ${model_type}\nfrom sklearn.metrics import classification_report, confusion_matrix, accuracy_score\nfrom sklearn.preprocessing import StandardScaler, LabelEncoder\n\n# Prepare data\nX = df.drop('${target_column}', axis=1)\ny = df['${target_column}']\n\n# Encode target if needed\nle = LabelEncoder()\ny = le.fit_transform(y)\n\n# Handle non-numeric features\nX = pd.get_dummies(X)\n\n# Split the data\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n\n# Scale features\nscaler = StandardScaler()\nX_train_scaled = scaler.fit_transform(X_train)\nX_test_scaled = scaler.transform(X_test)\n\n# Build and evaluate model\nmodel = ${model_type}(random_state=42)\nmodel.fit(X_train_scaled, y_train)\ny_pred = model.predict(X_test_scaled)\n\n# Evaluate the model\naccuracy = accuracy_score(y_test, y_pred)\nprint(f'Accuracy: {accuracy:.4f}')\nprint('\\nClassification Report:')\nprint(classification_report(y_test, y_pred, target_names=le.classes_))\nprint('\\nConfusion Matrix:')\nprint(confusion_matrix(y_test, y_pred))\n\n# Feature importance (if available)\nif hasattr(model, 'feature_importances_'):\n    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})\n    feature_importance = feature_importance.sort_values('Importance', ascending=False)\n    print('\\nFeature Importance:')\n    print(feature_importance.head(10))",
                        "parameters": ["model_type", "target_column"],
                        "options": {
                            "model_type": ["RandomForestClassifier", "GradientBoostingClassifier"],
                            "target_column": target_candidates
                        }
                    })
            
            elif technique == "clustering_analysis":
                pipeline["steps"].append({
                    "name": "clustering_analysis",
                    "description": "Perform clustering analysis",
                    "code_template": "# Clustering analysis\nfrom sklearn.cluster import ${algorithm}\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA\n\n# Select features for clustering\nfeatures = df.select_dtypes(include=['number']).columns.tolist()\nX = df[features]\n\n# Scale the features\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\n\n# Apply clustering\nmodel = ${algorithm}(${params})\ndf['cluster'] = model.fit_predict(X_scaled)\n\n# Analyze clusters\ncluster_sizes = df['cluster'].value_counts()\nprint('Cluster sizes:')\nprint(cluster_sizes)\n\n# Calculate cluster profiles\ncluster_profiles = df.groupby('cluster').mean()\nprint('\\nCluster profiles:')\nprint(cluster_profiles)\n\n# Visualize clusters with PCA\npca = PCA(n_components=2)\nX_pca = pca.fit_transform(X_scaled)\n\nplt.figure(figsize=(10, 8))\nplt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='viridis', alpha=0.5)\nplt.title('Cluster Visualization with PCA')\nplt.xlabel('First Principal Component')\nplt.ylabel('Second Principal Component')\nplt.colorbar(label='Cluster')",
                    "parameters": ["algorithm", "params"],
                    "options": {
                        "algorithm": ["KMeans", "DBSCAN", "AgglomerativeClustering"],
                        "params": ["n_clusters=3", "eps=0.5, min_samples=5", "n_clusters=3, linkage='ward'"]
                    }
                })
            
            elif technique == "dimensionality_reduction":
                pipeline["steps"].append({
                    "name": "dimensionality_reduction",
                    "description": "Perform dimensionality reduction",
                    "code_template": "# Dimensionality reduction\nfrom sklearn.decomposition import ${algorithm}\nfrom sklearn.preprocessing import StandardScaler\n\n# Select features\nfeatures = df.select_dtypes(include=['number']).columns.tolist()\nX = df[features]\n\n# Scale the features\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\n\n# Apply dimensionality reduction\nmodel = ${algorithm}(n_components=${n_components})\nX_reduced = model.fit_transform(X_scaled)\n\n# Create DataFrame with reduced dimensions\ncomponents_df = pd.DataFrame(\n    X_reduced, \n    columns=[f'Component_{i+1}' for i in range(${n_components})]\n)\n\n# Show explained variance (for PCA)\nif '${algorithm}' == 'PCA':\n    explained_variance = model.explained_variance_ratio_\n    print('Explained variance ratio:')\n    print(explained_variance)\n    print(f'Total variance explained: {sum(explained_variance):.4f}')\n\n# Visualize first two components\nplt.figure(figsize=(10, 8))\nplt.scatter(components_df['Component_1'], components_df['Component_2'], alpha=0.5)\nplt.title('Data in Reduced Dimension Space')\nplt.xlabel('Component 1')\nplt.ylabel('Component 2')",
                    "parameters": ["algorithm", "n_components"],
                    "options": {
                        "algorithm": ["PCA", "TruncatedSVD"],
                        "n_components": ["2", "3", "5"]
                    }
                })
    
    def _add_visualization_steps(self, pipeline: Dict[str, Any], 
                               recommendations: Dict[str, Any], 
                               data_profile: Dict[str, Any]) -> None:
        """Add visualization steps to the pipeline"""
        # Add a general visualization step
        pipeline["steps"].append({
            "name": "visualize_key_insights",
            "description": "Visualize key insights from the analysis",
            "code_template": "# Set up visualization environment\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nsns.set(style='whitegrid')\nplt.rcParams['figure.figsize'] = (12, 8)\n\n# Create visualizations based on data types\n${visualization_code}",
            "parameters": ["visualization_code"],
            "options": {
                "visualization_code": self._generate_visualization_code(recommendations, data_profile)
            }
        })
    
    def _generate_visualization_code(self, recommendations: Dict[str, Any], 
                                  data_profile: Dict[str, Any]) -> List[str]:
        """Generate visualization code based on recommendations and data profile"""
        visualization_code = []
        
        # Determine which visualizations to include
        vis_techniques = recommendations["visualization"]
        
        # Get column types
        column_types = data_profile.get("column_types", {})
        numerical_cols = [col for col, type in column_types.items() 
                         if type in ["numeric", "continuous"]]
        categorical_cols = [col for col, type in column_types.items() 
                           if type in ["categorical", "binary", "ordinal"]]
        datetime_cols = [col for col, type in column_types.items() 
                        if type == "datetime"]
        
        # Distribution plots for numerical
        if "bar_plot" in vis_techniques and numerical_cols:
            code = "# Distribution of numerical variables\nplt.figure(figsize=(12, 10))\nfor i, col in enumerate(df.select_dtypes(include=['number']).columns[:5]):\n    plt.subplot(3, 2, i+1)\n    sns.histplot(df[col], kde=True)\n    plt.title(f'Distribution of {col}')\nplt.tight_layout()"
            visualization_code.append(code)
        
        # Bar plots for categorical
        if "bar_plot" in vis_techniques and categorical_cols:
            code = "# Bar plots for categorical variables\nplt.figure(figsize=(12, 10))\nfor i, col in enumerate(df.select_dtypes(exclude=['number']).columns[:5]):\n    plt.subplot(3, 2, i+1)\n    value_counts = df[col].value_counts().nlargest(10)\n    sns.barplot(x=value_counts.index, y=value_counts.values)\n    plt.title(f'Counts for {col}')\n    plt.xticks(rotation=45, ha='right')\nplt.tight_layout()"
            visualization_code.append(code)
        
        # Box plots
        if "box_plot" in vis_techniques and numerical_cols:
            code = "# Box plots for numerical variables\nplt.figure(figsize=(12, 10))\nfor i, col in enumerate(df.select_dtypes(include=['number']).columns[:5]):\n    plt.subplot(3, 2, i+1)\n    sns.boxplot(y=df[col])\n    plt.title(f'Box plot of {col}')\nplt.tight_layout()"
            visualization_code.append(code)
        
        # Heatmap for correlations
        if "heatmap" in vis_techniques and len(numerical_cols) > 1:
            code = "# Correlation heatmap\nplt.figure(figsize=(12, 10))\ncorr = df.select_dtypes(include=['number']).corr()\nmask = np.triu(np.ones_like(corr, dtype=bool))\nsns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)\nplt.title('Correlation Matrix')\nplt.tight_layout()"
            visualization_code.append(code)
        
        # Scatter plots for related variables
        if "scatter_plot" in vis_techniques and len(numerical_cols) > 1:
            # Try to find correlated pairs
            corr_pairs = data_profile.get("potential_relationships", {}).get("high_correlations", [])
            if corr_pairs and len(corr_pairs) > 0:
                pair = corr_pairs[0]
                code = f"# Scatter plot for correlated variables\nplt.figure(figsize=(10, 8))\nsns.scatterplot(x=df['{pair['col1']}'], y=df['{pair['col2']}'])\nplt.title(f'Relationship between {pair['col1']} and {pair['col2']}')\nplt.tight_layout()"
                visualization_code.append(code)
            else:
                # Just use first two numeric columns
                if len(numerical_cols) >= 2:
                    code = f"# Scatter plot for numeric variables\nplt.figure(figsize=(10, 8))\nsns.scatterplot(x=df['{numerical_cols[0]}'], y=df['{numerical_cols[1]}'])\nplt.title(f'Relationship between {numerical_cols[0]} and {numerical_cols[1]}')\nplt.tight_layout()"
                    visualization_code.append(code)
        
        # Time series plot
        if datetime_cols and numerical_cols:
            code = f"# Time series plot\nplt.figure(figsize=(12, 6))\ndf_plot = df.copy()\ndf_plot['{datetime_cols[0]}'] = pd.to_datetime(df_plot['{datetime_cols[0]}'])\ndf_plot = df_plot.sort_values('{datetime_cols[0]}')\nplt.plot(df_plot['{datetime_cols[0]}'], df_plot['{numerical_cols[0]}'])\nplt.title(f'Time Series: {numerical_cols[0]} over time')\nplt.xlabel('Date')\nplt.ylabel('{numerical_cols[0]}')\nplt.grid(True)\nplt.tight_layout()"
            visualization_code.append(code)
        
        return visualization_code
    
    def _add_evaluation_steps(self, pipeline: Dict[str, Any], 
                            recommendations: Dict[str, Any], 
                            objective: str) -> None:
        """Add evaluation and interpretation steps to the pipeline"""
        # Add steps based on objective
        if objective == "prediction":
            pipeline["steps"].append({
                "name": "model_evaluation",
                "description": "Evaluate model performance",
                "code_template": "# Model evaluation\n${evaluation_code}",
                "parameters": ["evaluation_code"],
                "options": {
                    "evaluation_code": [
                        "# Classification evaluation\nfrom sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc\nimport numpy as np\nimport matplotlib.pyplot as plt\n\n# Calculate ROC curve and AUC\nfrom sklearn.model_selection import cross_val_predict\ny_pred_proba = cross_val_predict(model, X_train_scaled, y_train, cv=5, method='predict_proba')\nfpr, tpr, _ = roc_curve(y_train, y_pred_proba[:, 1])\nroc_auc = auc(fpr, tpr)\n\n# Plot ROC curve\nplt.figure(figsize=(10, 8))\nplt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')\nplt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')\nplt.xlim([0.0, 1.0])\nplt.ylim([0.0, 1.05])\nplt.xlabel('False Positive Rate')\nplt.ylabel('True Positive Rate')\nplt.title('Receiver Operating Characteristic')\nplt.legend(loc='lower right')\nplt.show()",
                        
                        "# Regression evaluation\nfrom sklearn.model_selection import cross_val_score, cross_val_predict\nimport numpy as np\nimport matplotlib.pyplot as plt\n\n# Cross-validation\ncv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')\nprint(f'Cross-validation R² scores: {cv_scores}')\nprint(f'Mean R² score: {np.mean(cv_scores):.4f}')\n\n# Predict vs. Actual plot\ny_cv_pred = cross_val_predict(model, X_train_scaled, y_train, cv=5)\n\nplt.figure(figsize=(10, 8))\nplt.scatter(y_train, y_cv_pred, alpha=0.5)\nplt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)\nplt.xlabel('Actual')\nplt.ylabel('Predicted')\nplt.title('Actual vs. Predicted Values')\nplt.show()"
                    ]
                }
            })
        
        elif objective == "exploration":
            pipeline["steps"].append({
                "name": "insights_summary",
                "description": "Summarize key insights from exploratory analysis",
                "code_template": "# Summarize key insights\n\n# Numerical summary\nprint('\\nNumerical Summary:')\nprint(df.describe())\n\n# Correlation summary\nif len(df.select_dtypes(include=['number']).columns) > 1:\n    corr = df.select_dtypes(include=['number']).corr()\n    high_corrs = corr.unstack().sort_values(ascending=False)\n    high_corrs = high_corrs[(high_corrs < 1.0) & (high_corrs > 0.5)]\n    print('\\nStrong Positive Correlations:')\n    print(high_corrs.head(10))\n\n# Missing data summary\nmissing = df.isnull().sum()\nmissing = missing[missing > 0]\nif not missing.empty:\n    print('\\nMissing Data Summary:')\n    print(missing)\n\n# Key statistics by group (if categorical variables exist)\ncat_cols = df.select_dtypes(exclude=['number']).columns\nnum_cols = df.select_dtypes(include=['number']).columns\nif len(cat_cols) > 0 and len(num_cols) > 0:\n    print('\\nKey Statistics by Group:')\n    for cat_col in cat_cols[:1]:  # Just first categorical column\n        for num_col in num_cols[:3]:  # First 3 numerical columns\n            print(f'\\n{num_col} by {cat_col}:')\n            print(df.groupby(cat_col)[num_col].describe().round(2))",
                "parameters": []
            })
        
        elif objective == "segmentation":
            pipeline["steps"].append({
                "name": "segment_profiling",
                "description": "Profile the identified segments",
                "code_template": "# Profile the clusters/segments\n\n# Basic cluster statistics\nprint('Cluster Sizes:')\nprint(df['cluster'].value_counts())\n\n# Profile each cluster\nprint('\\nCluster Profiles:')\nfor cluster in sorted(df['cluster'].unique()):\n    print(f'\\nCluster {cluster} Profile:')\n    cluster_data = df[df['cluster'] == cluster]\n    \n    # Numerical features\n    numerical_cols = cluster_data.select_dtypes(include=['number']).columns\n    for col in numerical_cols:\n        if col != 'cluster':\n            print(f'{col}:')\n            print(f'  Mean: {cluster_data[col].mean():.2f} (vs. {df[col].mean():.2f} overall)')\n            print(f'  Median: {cluster_data[col].median():.2f}')\n            \n    # Categorical features (top values)\n    categorical_cols = cluster_data.select_dtypes(exclude=['number']).columns\n    for col in categorical_cols:\n        print(f'{col} distribution:')\n        pct = cluster_data[col].value_counts(normalize=True) * 100\n        for val, p in pct.head(3).items():\n            overall_pct = df[df[col] == val].shape[0] / df.shape[0] * 100\n            print(f'  {val}: {p:.1f}% (vs. {overall_pct:.1f}% overall)')",
                "parameters": []
            })
        
        # Add final summary step for all objectives
        pipeline["steps"].append({
            "name": "final_summary",
            "description": "Final summary and recommendations",
            "code_template": "# Overall summary\nprint('\\n' + '='*50)\nprint('ANALYSIS SUMMARY')\nprint('='*50)\n\n# Dataset summary\nprint(f'Dataset: {df.shape[0]} rows, {df.shape[1]} columns')\n\n# Key findings\nprint('\\nKey Findings:')\nprint('1. <Highlight key finding 1 based on your analysis>')\nprint('2. <Highlight key finding 2 based on your analysis>')\nprint('3. <Highlight key finding 3 based on your analysis>')\n\n# Recommendations\nprint('\\nRecommendations:')\nprint('1. <Add recommendation 1 based on findings>')\nprint('2. <Add recommendation 2 based on findings>')\n\n# Next steps\nprint('\\nSuggested Next Steps:')\nprint('1. <Suggest next analysis step 1>')\nprint('2. <Suggest next analysis step 2>')",
            "parameters": []
        })

# Factory function to create a notebook analyzer
def create_notebook_analyzer():
    return NotebookAnalyzer()