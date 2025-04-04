import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class AnalysisRecommender:
    def __init__(self):
        # Define analysis techniques by category
        self.analysis_techniques = {
            "exploratory": [
                {
                    "id": "distribution_analysis",
                    "name": "Distribution Analysis",
                    "description": "Analyze the distributions of individual variables",
                    "requirements": ["at_least_one_column"],
                    "value": "Understand the shape and characteristics of your data"
                },
                {
                    "id": "correlation_analysis",
                    "name": "Correlation Analysis",
                    "description": "Identify relationships between variables",
                    "requirements": ["multiple_numeric_columns"],
                    "value": "Discover which variables are related to each other"
                },
                {
                    "id": "outlier_detection",
                    "name": "Outlier Detection",
                    "description": "Find unusual data points that may require investigation",
                    "requirements": ["at_least_one_numeric_column"],
                    "value": "Identify anomalies and potential data quality issues"
                }
            ],
            "statistical": [
                {
                    "id": "hypothesis_testing",
                    "name": "Hypothesis Testing",
                    "description": "Test specific hypotheses about your data",
                    "requirements": ["at_least_one_numeric_column"],
                    "value": "Validate assumptions and test theories about your data"
                },
                {
                    "id": "anova_analysis",
                    "name": "ANOVA Analysis",
                    "description": "Compare means across multiple groups",
                    "requirements": ["one_numeric_target", "one_categorical_feature"],
                    "value": "Determine if there are significant differences between groups"
                }
            ],
            "predictive": [
                {
                    "id": "regression_analysis",
                    "name": "Regression Analysis",
                    "description": "Predict numeric values based on other variables",
                    "requirements": ["one_numeric_target", "multiple_predictor_columns"],
                    "value": "Forecast values and understand factors that influence outcomes"
                },
                {
                    "id": "classification",
                    "name": "Classification Analysis",
                    "description": "Predict categories based on other variables",
                    "requirements": ["one_categorical_target", "multiple_predictor_columns"],
                    "value": "Categorize data points and understand what drives category membership"
                },
                {
                    "id": "time_series_forecasting",
                    "name": "Time Series Forecasting",
                    "description": "Predict future values based on historical patterns",
                    "requirements": ["one_datetime_column", "one_numeric_target"],
                    "value": "Project future trends and seasonal patterns"
                }
            ],
            "prescriptive": [
                {
                    "id": "optimization_analysis",
                    "name": "Optimization Analysis",
                    "description": "Find optimal values for decision variables",
                    "requirements": ["numeric_decision_variables", "objective_function"],
                    "value": "Determine the best course of action to maximize results"
                },
                {
                    "id": "what_if_analysis",
                    "name": "What-If Analysis",
                    "description": "Simulate outcomes under different scenarios",
                    "requirements": ["numeric_inputs", "outcome_variable"],
                    "value": "Explore different scenarios and their potential outcomes"
                }
            ],
            "diagnostic": [
                {
                    "id": "root_cause_analysis",
                    "name": "Root Cause Analysis",
                    "description": "Identify factors contributing to specific outcomes",
                    "requirements": ["outcome_variable", "multiple_predictor_columns"],
                    "value": "Understand why certain events or outcomes occur"
                },
                {
                    "id": "comparative_analysis",
                    "name": "Comparative Analysis",
                    "description": "Compare performance across different segments or time periods",
                    "requirements": ["segment_variable", "performance_metric"],
                    "value": "Identify high and low performing areas"
                }
            ]
        }
        
        # Define analysis objectives
        self.analysis_objectives = {
            "general": ["exploratory", "statistical", "predictive"],
            "descriptive": ["exploratory"],
            "diagnostic": ["exploratory", "statistical", "diagnostic"],
            "predictive": ["exploratory", "predictive"],
            "prescriptive": ["predictive", "prescriptive"]
        }
        
        logger.info("AnalysisRecommender initialized with techniques and objectives")
    
    def recommend_analyses(self, data_profile: Dict[str, Any], objective: str = "general") -> List[Dict[str, Any]]:
        """Recommend appropriate analysis techniques based on data profile and objective"""
        recommendations = []
        
        # Get categories to include based on objective
        relevant_categories = self.analysis_objectives.get(objective, self.analysis_objectives["general"])
        logger.info(f"Recommending analyses for objective '{objective}' using categories: {relevant_categories}")
        
        # Check which techniques are appropriate for this dataset
        for category in relevant_categories:
            if category not in self.analysis_techniques:
                continue
                
            for technique in self.analysis_techniques[category]:
                if self._check_requirements(technique["requirements"], data_profile):
                    # Add recommendation with tailored explanation
                    recommendations.append({
                        "id": technique["id"],
                        "name": technique["name"],
                        "description": technique["description"],
                        "value_proposition": technique["value"],
                        "example_insight": self._generate_example_insight(technique["id"], data_profile),
                        "confidence_score": self._calculate_confidence(technique["id"], data_profile, objective)
                    })
        
        # Sort by confidence score
        recommendations.sort(key=lambda x: x["confidence_score"], reverse=True)
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations
    
    def _check_requirements(self, requirements: List[str], profile: Dict[str, Any]) -> bool:
        """Check if the dataset meets the requirements for a technique"""
        for req in requirements:
            if req == "at_least_one_column":
                if profile["basic_stats"]["column_count"] < 1:
                    return False
            elif req == "multiple_numeric_columns":
                numeric_cols = [col for col, type in profile["column_types"].items() 
                              if type == "numeric"]
                if len(numeric_cols) < 2:
                    return False
            elif req == "one_numeric_target":
                # This would require user input in practice
                numeric_cols = [col for col, type in profile["column_types"].items() 
                              if type == "numeric"]
                if len(numeric_cols) < 1:
                    return False
            elif req == "one_categorical_target":
                categorical_cols = [col for col, type in profile["column_types"].items() 
                                  if type in ["categorical", "binary"]]
                if len(categorical_cols) < 1:
                    return False
            elif req == "multiple_predictor_columns":
                if profile["basic_stats"]["column_count"] < 3:  # Need at least target + 2 predictors
                    return False
            elif req == "one_datetime_column":
                datetime_cols = [col for col, type in profile["column_types"].items() 
                              if type == "datetime"]
                if len(datetime_cols) < 1:
                    return False
            elif req == "one_categorical_feature":
                categorical_cols = [col for col, type in profile["column_types"].items() 
                                  if type in ["categorical", "binary"]]
                if len(categorical_cols) < 1:
                    return False
            elif req in ["numeric_decision_variables", "objective_function", 
                       "numeric_inputs", "outcome_variable", 
                       "segment_variable", "performance_metric"]:
                # These are more sophisticated requirements that would need specific user input
                # For MVP, we'll assume they're satisfied if we have enough numeric columns
                numeric_cols = [col for col, type in profile["column_types"].items() 
                              if type == "numeric"]
                if len(numeric_cols) < 2:
                    return False
        
        return True
    
    def _generate_example_insight(self, technique_id: str, profile: Dict[str, Any]) -> str:
        """Generate an example insight for a technique based on the profile"""
        # This would be more sophisticated in a real implementation
        if technique_id == "distribution_analysis":
            # Find a numeric column with interesting distribution
            for col, dist in profile.get("distribution_info", {}).items():
                if "skew" in dist and abs(dist["skew"]) > 1:
                    return f"Column '{col}' has a skewed distribution, which may need transformation"
            
            return "Understand the shape of your key variables"
            
        elif technique_id == "correlation_analysis":
            # Look for correlations
            if "high_correlations" in profile.get("potential_relationships", {}):
                if len(profile["potential_relationships"]["high_correlations"]) > 0:
                    corr = profile["potential_relationships"]["high_correlations"][0]
                    return f"There appears to be a strong relationship between '{corr['col1']}' and '{corr['col2']}'"
            
            return "Discover hidden relationships between your variables"
            
        elif technique_id == "regression_analysis":
            # Find a potential target variable
            numeric_cols = [col for col, type in profile["column_types"].items() 
                          if type == "numeric"]
            if numeric_cols:
                return f"You could predict '{numeric_cols[0]}' based on other variables in your dataset"
            
            return "Forecast key metrics based on other variables"
            
        elif technique_id == "classification":
            categorical_cols = [col for col, type in profile["column_types"].items() 
                              if type in ["categorical", "binary"]]
            if categorical_cols:
                return f"You could classify '{categorical_cols[0]}' based on other variables in your dataset"
            
            return "Categorize items based on their characteristics"
            
        elif technique_id == "time_series_forecasting":
            datetime_cols = [col for col, type in profile["column_types"].items() 
                           if type == "datetime"]
            numeric_cols = [col for col, type in profile["column_types"].items() 
                          if type == "numeric"]
            if datetime_cols and numeric_cols:
                return f"You could forecast '{numeric_cols[0]}' over time using '{datetime_cols[0]}'"
            
            return "Predict future values based on historical trends"
            
        # Default generic insight
        return "This analysis could reveal valuable insights for your data"
    
    def _calculate_confidence(self, technique_id: str, profile: Dict[str, Any], objective: str) -> float:
        """Calculate confidence score for a technique recommendation"""
        # Simplified scoring logic - would be more sophisticated in reality
        base_score = 0.5  # Start with 50% confidence
        
        # Adjust based on technique
        if technique_id == "distribution_analysis":
            # Always useful
            base_score = 0.9
            
        elif technique_id == "correlation_analysis":
            # More confident if we've already detected correlations
            if "high_correlations" in profile.get("potential_relationships", {}):
                if len(profile["potential_relationships"]["high_correlations"]) > 0:
                    base_score = 0.85
                else:
                    base_score = 0.7
            
        elif technique_id == "regression_analysis":
            # More confident with more numeric columns and relationships
            numeric_cols = [col for col, type in profile["column_types"].items() 
                          if type == "numeric"]
            if len(numeric_cols) > 3:
                base_score += 0.2
            
            if "high_correlations" in profile.get("potential_relationships", {}):
                if len(profile["potential_relationships"]["high_correlations"]) > 0:
                    base_score += 0.1
                    
        # Adjust based on objective
        if objective != "general":
            # Check if technique aligns with the specified objective
            for category, techniques in self.analysis_techniques.items():
                if category in self.analysis_objectives.get(objective, []):
                    if any(t["id"] == technique_id for t in techniques):
                        base_score += 0.1
        
        # Cap at 0.95
        return min(0.95, base_score)