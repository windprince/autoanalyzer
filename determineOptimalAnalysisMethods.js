/**
 * Determines the optimal analysis methods based on dataset characteristics
 * and historical patterns extracted from Jupyter notebooks
 */
class OptimalAnalysisMethodDeterminer {
  constructor() {
    this.notebookPatterns = {};
    this.analysisTemplates = {};
    this.datasetTypeRules = {
      'time_series': [
        'decomposition_analysis',
        'seasonal_analysis',
        'trend_analysis',
        'forecasting',
        'arima_sarima_models',
        'prophet_forecasting',
        'stationarity_tests'
      ],
      'transactional': [
        'customer_segmentation',
        'rfm_analysis',
        'market_basket_analysis',
        'cohort_analysis',
        'funnel_analysis',
        'conversion_analysis'
      ],
      'survey_data': [
        'factor_analysis',
        'reliability_tests',
        'sentiment_analysis',
        'cluster_analysis',
        'multiple_correspondence_analysis'
      ],
      'user_data': [
        'behavioral_analysis',
        'retention_analysis',
        'churn_prediction',
        'lifetime_value_analysis',
        'engagement_metrics'
      ],
      'categorical_heavy': [
        'chi_square_tests',
        'association_analysis',
        'logistic_regression',
        'decision_trees',
        'random_forests'
      ],
      'numerical_heavy': [
        'correlation_analysis',
        'regression_models',
        'principal_component_analysis',
        'clustering',
        'anomaly_detection'
      ],
      'geospatial': [
        'spatial_clustering',
        'hotspot_detection',
        'distance_calculations',
        'geospatial_visualization',
        'catchment_area_analysis'
      ]
    };
    this.columnTypeRules = {
      'datetime': ['time_series_decomposition', 'trend_analysis', 'seasonality_detection'],
      'continuous': ['distribution_analysis', 'correlation_analysis', 'regression'],
      'categorical': ['frequency_analysis', 'chi_square_tests', 'classification'],
      'binary': ['logistic_regression', 'classification', 'probability_analysis'],
      'ordinal': ['non_parametric_tests', 'rank_correlation', 'ordinal_regression'],
      'text': ['sentiment_analysis', 'topic_modeling', 'text_summarization', 'NER'],
      'email': ['domain_analysis', 'contact_analysis'],
      'url': ['domain_analysis', 'link_structure_analysis'],
      'ip_address': ['network_analysis', 'geolocation_analysis'],
      'geographical': ['spatial_analysis', 'clustering', 'distance_calculations']
    };
    this.dataQualityRules = {
      'high_missing_values': ['imputation_techniques', 'missing_pattern_analysis'],
      'potential_outliers': ['robust_statistics', 'outlier_detection_and_treatment'],
      'duplicate_rows': ['deduplication_techniques', 'entity_resolution'],
      'mixed_types': ['type_conversion', 'feature_extraction'],
      'low_variance': ['feature_selection', 'feature_engineering']
    };
  }

  /**
   * Learn analysis patterns from Jupyter notebooks
   * @param {Object} notebook The parsed Jupyter notebook
   * @param {string} datasetType The type of dataset used in the notebook
   */
  learnFromNotebook(notebook, datasetType) {
    // Extract analysis workflow
    const workflow = this._extractWorkflow(notebook);
    
    // Associate workflow with dataset characteristics
    if (!this.notebookPatterns[datasetType]) {
      this.notebookPatterns[datasetType] = [];
    }
    
    this.notebookPatterns[datasetType].push(workflow);
    
    // Update templates based on frequency of patterns
    this._updateTemplates(datasetType);
  }
  
  /**
   * Extract the analysis workflow from a notebook
   * @param {Object} notebook The notebook object
   * @returns {Object} The extracted workflow
   */
  _extractWorkflow(notebook) {
    const workflow = {
      preprocessing: [],
      visualization: [],
      analysis: [],
      evaluation: [],
      columnsUsed: new Set(),
      importedLibraries: new Set()
    };
    
    // Simulated extraction logic
    // In real implementation, this would parse notebook cells and code
    
    return workflow;
  }
  
  /**
   * Update analysis templates based on observed patterns
   * @param {string} datasetType Type of the dataset
   */
  _updateTemplates(datasetType) {
    // Count frequency of different techniques
    const patternCounts = {};
    const patterns = this.notebookPatterns[datasetType] || [];
    
    for (const pattern of patterns) {
      // Count preprocessing steps
      for (const step of pattern.preprocessing) {
        patternCounts[step] = (patternCounts[step] || 0) + 1;
      }
      
      // Count analysis techniques
      for (const technique of pattern.analysis) {
        patternCounts[technique] = (patternCounts[technique] || 0) + 1;
      }
      
      // Count visualization types
      for (const viz of pattern.visualization) {
        patternCounts[viz] = (patternCounts[viz] || 0) + 1;
      }
    }
    
    // Create template based on most frequent techniques
    this.analysisTemplates[datasetType] = {
      preprocessing: this._getMostFrequent(patternCounts, 'preprocessing'),
      analysis: this._getMostFrequent(patternCounts, 'analysis'),
      visualization: this._getMostFrequent(patternCounts, 'visualization')
    };
  }
  
  /**
   * Get the most frequent techniques of a certain type
   * @param {Object} counts Frequency counts of different techniques
   * @param {string} prefix Type of technique
   * @returns {Array} The most frequent techniques
   */
  _getMostFrequent(counts, prefix) {
    // Filter techniques of the specified type
    const relevantCounts = Object.entries(counts)
      .filter(([key]) => key.startsWith(prefix))
      .sort((a, b) => b[1] - a[1]);
    
    // Return the top 5 or all if less than 5
    return relevantCounts.slice(0, 5).map(([key]) => key);
  }
  
  /**
   * Determine optimal analysis methods for a dataset based on its profile
   * @param {Object} dataProfile The profile of the dataset
   * @returns {Object} Recommended analysis methods
   */
  determineOptimalMethods(dataProfile) {
    const result = {
      recommended_methods: [],
      explanation: [],
      confidence_scores: {}
    };
    
    // Determine dataset type
    const datasetType = this._determineDatasetType(dataProfile);
    result.dataset_type = datasetType;
    
    // Get methods based on dataset type
    this._addMethodsBasedOnDatasetType(result, datasetType);
    
    // Get methods based on column types
    this._addMethodsBasedOnColumnTypes(result, dataProfile);
    
    // Get methods based on data quality issues
    this._addMethodsBasedOnDataQuality(result, dataProfile);
    
    // Get methods based on notebook patterns
    this._addMethodsBasedOnNotebookPatterns(result, datasetType, dataProfile);
    
    // Sort and deduplicate methods
    result.recommended_methods = [...new Set(result.recommended_methods)]
      .sort((a, b) => (result.confidence_scores[b] || 0) - (result.confidence_scores[a] || 0));
    
    return result;
  }
  
  /**
   * Determine the type of dataset based on its profile
   * @param {Object} dataProfile The profile of the dataset
   * @returns {string} The determined dataset type
   */
  _determineDatasetType(dataProfile) {
    const typeProbabilities = {};
    
    // Check for time series characteristics
    if (dataProfile.metadata && dataProfile.metadata.dataset_type) {
      // Use the dataset classification from the profile
      Object.entries(dataProfile.metadata.dataset_type).forEach(([type, score]) => {
        typeProbabilities[type] = score;
      });
    }
    
    // Determine column type distribution
    const columnTypes = dataProfile.column_types || {};
    const typeDistribution = Object.values(columnTypes).reduce((acc, type) => {
      acc[type] = (acc[type] || 0) + 1;
      return acc;
    }, {});
    
    // Check for numerical vs categorical heavy datasets
    const totalColumns = Object.keys(columnTypes).length;
    const numericalColumns = (typeDistribution['numeric'] || 0) + (typeDistribution['continuous'] || 0);
    const categoricalColumns = (typeDistribution['categorical'] || 0) + (typeDistribution['binary'] || 0) + (typeDistribution['ordinal'] || 0);
    
    if (numericalColumns / totalColumns > 0.7) {
      typeProbabilities['numerical_heavy'] = 0.9;
    }
    
    if (categoricalColumns / totalColumns > 0.7) {
      typeProbabilities['categorical_heavy'] = 0.9;
    }
    
    // Check for geographical data
    if (dataProfile.geographical_data && Object.keys(dataProfile.geographical_data).length > 0) {
      typeProbabilities['geospatial'] = 0.8;
    }
    
    // Return the dataset type with highest probability
    const sortedTypes = Object.entries(typeProbabilities)
      .sort((a, b) => b[1] - a[1]);
    
    return sortedTypes.length > 0 ? sortedTypes[0][0] : 'unknown';
  }
  
  /**
   * Add methods based on the dataset type
   * @param {Object} result The result object to modify
   * @param {string} datasetType The type of dataset
   */
  _addMethodsBasedOnDatasetType(result, datasetType) {
    const methods = this.datasetTypeRules[datasetType] || [];
    
    for (const method of methods) {
      result.recommended_methods.push(method);
      result.confidence_scores[method] = 0.8;
      result.explanation.push(`Method '${method}' recommended for ${datasetType} datasets`);
    }
  }
  
  /**
   * Add methods based on column types in the dataset
   * @param {Object} result The result object to modify
   * @param {Object} dataProfile The profile of the dataset
   */
  _addMethodsBasedOnColumnTypes(result, dataProfile) {
    const columnTypes = dataProfile.column_types || {};
    
    for (const [column, type] of Object.entries(columnTypes)) {
      const methods = this.columnTypeRules[type] || [];
      
      for (const method of methods) {
        result.recommended_methods.push(method);
        result.confidence_scores[method] = (result.confidence_scores[method] || 0) + 0.1;
        result.explanation.push(`Method '${method}' recommended for ${type} column '${column}'`);
      }
    }
  }
  
  /**
   * Add methods based on data quality issues
   * @param {Object} result The result object to modify
   * @param {Object} dataProfile The profile of the dataset
   */
  _addMethodsBasedOnDataQuality(result, dataProfile) {
    const qualityIssues = dataProfile.data_quality_issues || [];
    
    for (const issue of qualityIssues) {
      const issueType = issue.type;
      const methods = this.dataQualityRules[issueType] || [];
      
      for (const method of methods) {
        result.recommended_methods.push(method);
        result.confidence_scores[method] = (result.confidence_scores[method] || 0) + 0.2;
        result.explanation.push(`Method '${method}' recommended for handling ${issueType} in column '${issue.column}'`);
      }
    }
  }
  
  /**
   * Add methods based on notebook patterns
   * @param {Object} result The result object to modify
   * @param {string} datasetType The type of dataset
   * @param {Object} dataProfile The profile of the dataset
   */
  _addMethodsBasedOnNotebookPatterns(result, datasetType, dataProfile) {
    // Check if we have templates for this dataset type
    if (this.analysisTemplates[datasetType]) {
      const template = this.analysisTemplates[datasetType];
      
      // Add preprocessing methods
      for (const method of template.preprocessing) {
        result.recommended_methods.push(method);
        result.confidence_scores[method] = (result.confidence_scores[method] || 0) + 0.5;
        result.explanation.push(`Method '${method}' recommended based on successful notebook patterns`);
      }
      
      // Add analysis methods
      for (const method of template.analysis) {
        result.recommended_methods.push(method);
        result.confidence_scores[method] = (result.confidence_scores[method] || 0) + 0.7;
        result.explanation.push(`Method '${method}' recommended based on successful notebook patterns`);
      }
      
      // Add visualization methods
      for (const method of template.visualization) {
        result.recommended_methods.push(method);
        result.confidence_scores[method] = (result.confidence_scores[method] || 0) + 0.6;
        result.explanation.push(`Visualization '${method}' recommended based on successful notebook patterns`);
      }
    }
  }
  
  /**
   * Find similar datasets and their analysis methods
   * @param {Object} dataProfile The profile of the dataset
   * @param {Array} datasetLibrary Library of dataset profiles
   * @returns {Array} Similar datasets and their analysis methods
   */
  findSimilarDatasets(dataProfile, datasetLibrary) {
    const similarities = [];
    
    for (const libraryDataset of datasetLibrary) {
      const similarity = this._calculateProfileSimilarity(dataProfile, libraryDataset.profile);
      
      if (similarity > 0.6) {  // Only consider datasets with > 60% similarity
        similarities.push({
          datasetName: libraryDataset.name,
          similarity: similarity,
          recommendedMethods: libraryDataset.methods
        });
      }
    }
    
    // Sort by similarity (descending)
    return similarities.sort((a, b) => b.similarity - a.similarity);
  }
  
  /**
   * Calculate similarity between two dataset profiles
   * @param {Object} profile1 First dataset profile
   * @param {Object} profile2 Second dataset profile
   * @returns {number} Similarity score (0-1)
   */
  _calculateProfileSimilarity(profile1, profile2) {
    let totalScore = 0;
    let totalFactors = 0;
    
    // Compare column types
    const columnTypeScore = this._compareColumnTypes(profile1, profile2);
    totalScore += columnTypeScore;
    totalFactors += 1;
    
    // Compare dataset size
    const sizeScore = this._compareSizes(profile1, profile2);
    totalScore += sizeScore;
    totalFactors += 1;
    
    // Compare missing data patterns
    const missingDataScore = this._compareMissingData(profile1, profile2);
    totalScore += missingDataScore;
    totalFactors += 1;
    
    // Compare column relationships
    const relationshipScore = this._compareRelationships(profile1, profile2);
    totalScore += relationshipScore;
    totalFactors += 1;
    
    return totalScore / totalFactors;
  }
  
  /**
   * Compare column types between two profiles
   * @param {Object} profile1 First profile
   * @param {Object} profile2 Second profile
   * @returns {number} Similarity score for column types (0-1)
   */
  _compareColumnTypes(profile1, profile2) {
    const types1 = profile1.column_types || {};
    const types2 = profile2.column_types || {};
    
    // Calculate distribution of each type
    const distribution1 = this._getTypeDistribution(types1);
    const distribution2 = this._getTypeDistribution(types2);
    
    // Calculate similarity between distributions
    let similarity = 0;
    let totalTypes = 0;
    
    for (const type of Object.keys({...distribution1, ...distribution2})) {
      const freq1 = distribution1[type] || 0;
      const freq2 = distribution2[type] || 0;
      
      similarity += 1 - Math.abs(freq1 - freq2);
      totalTypes += 1;
    }
    
    return totalTypes > 0 ? similarity / totalTypes : 0;
  }
  
  /**
   * Get distribution of column types
   * @param {Object} columnTypes Column types object
   * @returns {Object} Distribution of types
   */
  _getTypeDistribution(columnTypes) {
    const typeCounts = {};
    const totalColumns = Object.keys(columnTypes).length;
    
    for (const type of Object.values(columnTypes)) {
      typeCounts[type] = (typeCounts[type] || 0) + 1;
    }
    
    // Convert counts to frequencies
    for (const type of Object.keys(typeCounts)) {
      typeCounts[type] = typeCounts[type] / totalColumns;
    }
    
    return typeCounts;
  }
  
  /**
   * Compare dataset sizes
   * @param {Object} profile1 First profile
   * @param {Object} profile2 Second profile
   * @returns {number} Similarity score for sizes (0-1)
   */
  _compareSizes(profile1, profile2) {
    const rows1 = profile1.basic_stats?.row_count || 0;
    const rows2 = profile2.basic_stats?.row_count || 0;
    const cols1 = profile1.basic_stats?.column_count || 0;
    const cols2 = profile2.basic_stats?.column_count || 0;
    
    // Compare row counts on logarithmic scale
    const rowSimilarity = Math.min(rows1, rows2) / Math.max(rows1, rows2);
    
    // Compare column counts
    const colSimilarity = Math.min(cols1, cols2) / Math.max(cols1, cols2);
    
    return (rowSimilarity + colSimilarity) / 2;
  }
  
  /**
   * Compare missing data patterns
   * @param {Object} profile1 First profile
   * @param {Object} profile2 Second profile
   * @returns {number} Similarity score for missing data (0-1)
   */
  _compareMissingData(profile1, profile2) {
    const missing1 = profile1.missing_data?.missing_percentages || {};
    const missing2 = profile2.missing_data?.missing_percentages || {};
    
    // Calculate average missing percentage
    const avgMissing1 = this._calculateAverage(Object.values(missing1));
    const avgMissing2 = this._calculateAverage(Object.values(missing2));
    
    // Compare average missing percentages
    return 1 - Math.abs(avgMissing1 - avgMissing2) / 100;
  }
  
  /**
   * Calculate average of a list of numbers
   * @param {Array} values List of numbers
   * @returns {number} Average value
   */
  _calculateAverage(values) {
    if (values.length === 0) return 0;
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }
  
  /**
   * Compare column relationships
   * @param {Object} profile1 First profile
   * @param {Object} profile2 Second profile
   * @returns {number} Similarity score for relationships (0-1)
   */
  _compareRelationships(profile1, profile2) {
    const rel1 = profile1.potential_relationships || {};
    const rel2 = profile2.potential_relationships || {};
    
    // Compare number of high correlations
    const corr1 = rel1.high_correlations?.length || 0;
    const corr2 = rel2.high_correlations?.length || 0;
    
    // Compare density of correlations
    const cols1 = profile1.basic_stats?.column_count || 1;
    const cols2 = profile2.basic_stats?.column_count || 1;
    
    const density1 = corr1 / (cols1 * (cols1 - 1) / 2);
    const density2 = corr2 / (cols2 * (cols2 - 1) / 2);
    
    return 1 - Math.abs(density1 - density2);
  }
}

/**
 * Factory function to create an OptimalAnalysisMethodDeterminer
 * @returns {OptimalAnalysisMethodDeterminer} A new determiner instance
 */
function createAnalysisMethodDeterminer() {
  return new OptimalAnalysisMethodDeterminer();
}

/**
 * Legacy function to determine optimal analysis methods
 * @param {Object} dataCharacteristics Data characteristics
 * @param {Object} businessContext Business context
 * @returns {Array} Scored analysis methods
 */
function determineOptimalAnalysisMethods(dataCharacteristics, businessContext) {
  // Create and use the new determiner
  const determiner = createAnalysisMethodDeterminer();
  const results = determiner.determineOptimalMethods(dataCharacteristics);
  
  // Format results to match legacy format
  return results.recommended_methods.map(method => ({
    method,
    score: results.confidence_scores[method] || 0.5
  }));
}

// Export the factory function and legacy function
module.exports = { 
  createAnalysisMethodDeterminer,
  determineOptimalAnalysisMethods
};