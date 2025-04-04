// Main JavaScript for AutoAnalyzer Application

document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const loadingDiv = document.getElementById('loading');
    const resultsDiv = document.getElementById('results');
    
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        console.log("Form submission intercepted");
        
        const fileInput = document.getElementById('file');
        if (!fileInput.files || fileInput.files.length === 0) {
            alert('Please select a file');
            return;
        }
        
        console.log("File selected:", fileInput.files[0].name);
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        // Add analysis objective if selected
        const objectiveSelect = document.getElementById('analysis-objective');
        if (objectiveSelect.value) {
            formData.append('objective', objectiveSelect.value);
            console.log("Objective selected:", objectiveSelect.value);
        }
        
        // Show loading indicator
        loadingDiv.classList.remove('d-none');
        resultsDiv.classList.add('d-none');
        
        console.log("Sending request to /analyze");
        
        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            console.log("Response received:", response.status);
            if (!response.ok) {
                throw new Error(`Analysis failed with status ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Data received:", data);
            window.currentResults = data; // Store results for later use
            displayResults(data);
            loadingDiv.classList.add('d-none');
            resultsDiv.classList.remove('d-none');
        })
        .catch(error => {
            console.error("Error occurred:", error);
            alert('Error: ' + error.message);
            loadingDiv.classList.add('d-none');
        });
    });
    
    function displayResults(data) {
        // Display data summary
        const summaryDiv = document.getElementById('data-summary');
        summaryDiv.innerHTML = `
            <p><strong>Dataset size:</strong> ${data.data_summary.dataset_size}</p>
            <p><strong>Column types:</strong></p>
            <ul>
                ${Object.entries(data.data_summary.column_composition)
                  .filter(([type, count]) => count > 0)
                  .map(([type, count]) => `<li class="${type}-column">${type}: ${count}</li>`)
                  .join('')}
            </ul>
            ${data.data_summary.key_stats.length > 0 ? 
              `<p><strong>Key observations:</strong></p>
               <ul>${data.data_summary.key_stats.map(stat => `<li>${stat}</li>`).join('')}</ul>` : ''}
        `;
        
        // Display quality alerts
        const alertsDiv = document.getElementById('quality-alerts');
        if (data.data_quality_alerts && data.data_quality_alerts.length > 0) {
            alertsDiv.innerHTML = `
                <div class="quality-issues">
                    ${data.data_quality_alerts.map(alert => `
                        <div class="quality-issue">
                            <strong>${alert.column}:</strong> ${alert.description}
                        </div>
                    `).join('')}
                </div>
            `;
        } else {
            alertsDiv.innerHTML = '<p>No quality issues detected.</p>';
        }
        
        // Display recommendations
        const recommendationsDiv = document.getElementById('recommendations');
        if (data.analysis_recommendations && data.analysis_recommendations.length > 0) {
            recommendationsDiv.innerHTML = data.analysis_recommendations.map(rec => `
                <div class="card recommendation-card">
                    <div class="card-body">
                        <h5 class="card-title">${rec.name}</h5>
                        <div class="confidence-indicator">
                            <div class="confidence-level" style="width: ${rec.confidence_score * 100}%"></div>
                        </div>
                        <p class="card-text">${rec.description}</p>
                        <p><strong>Expected insight:</strong> ${rec.example_insight}</p>
                        <p><em>Business value: ${rec.value_proposition}</em></p>
                        <button class="btn btn-sm btn-outline-primary run-analysis" data-analysis-id="${rec.id}">Run this analysis</button>
                    </div>
                </div>
            `).join('');
            
            // Add event listeners to the "Run Analysis" buttons
            document.querySelectorAll('.run-analysis').forEach(button => {
                button.addEventListener('click', function() {
                    const analysisId = this.getAttribute('data-analysis-id');
                    runAnalysis(analysisId);
                });
            });
        } else {
            recommendationsDiv.innerHTML = '<p>No recommendations available for this dataset.</p>';
        }
    }
    
    function runAnalysis(analysisId) {
        // Create analysis panel if it doesn't exist
        if (!document.getElementById('analysis-panel')) {
            const panel = document.createElement('div');
            panel.id = 'analysis-panel';
            panel.className = 'analysis-panel';
            panel.innerHTML = `
                <span class="panel-close">&times;</span>
                <h3>Analysis Results</h3>
                <div id="analysis-content">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p>Running analysis...</p>
                </div>
            `;
            document.body.appendChild(panel);
            
            // Close button functionality
            document.querySelector('.panel-close').addEventListener('click', function() {
                document.getElementById('analysis-panel').classList.remove('open');
            });
        }
        
        // Open the panel
        const panel = document.getElementById('analysis-panel');
        panel.classList.add('open');
        
        // Get current dataset path from previous analysis
        let datasetPath = ""; 
        if (window.currentResults && window.currentResults.dataset_path) {
            datasetPath = window.currentResults.dataset_path;
        }
        
        // Prepare the content area
        const analysisContent = document.getElementById('analysis-content');
        analysisContent.innerHTML = `
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p>Running analysis...</p>
        `;
        
        // Call the backend with the analysis request
        fetch('/run_analysis', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                analysis_id: analysisId,
                dataset_path: datasetPath,
                parameters: {}
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Analysis failed');
            }
            return response.json();
        })
        .then(data => {
            displayAnalysisResults(data, analysisContent);
        })
        .catch(error => {
            analysisContent.innerHTML = `
                <div class="alert alert-danger">
                    <strong>Error:</strong> ${error.message}
                </div>
                <p>Please try again or select a different analysis.</p>
            `;
        });
    }
    
    function displayAnalysisResults(data, container) {
        // Handle different analysis types differently
        let contentHtml = '';
        
        if (data.error) {
            contentHtml = `
                <div class="alert alert-danger">
                    <strong>Error:</strong> ${data.error}
                </div>
            `;
        } else {
            // Common header
            contentHtml = `
                <h4>Analysis: ${data.analysis_type}</h4>
                <p>Analysis complete! Here are your results:</p>
            `;
            
            // Display visualizations if available
            if (data.visualizations && Object.keys(data.visualizations).length > 0) {
                contentHtml += `<div class="visualizations-container">`;
                for (const [key, path] of Object.entries(data.visualizations)) {
                    contentHtml += `
                        <div class="visualization-item">
                            <h5>${key}</h5>
                            <img src="${path}" class="img-fluid analysis-image" alt="${key} visualization">
                        </div>
                    `;
                }
                contentHtml += `</div>`;
            } else {
                contentHtml += `
                    <div class="placeholder-chart mb-4" style="height: 200px; background-color: #f8f9fa; border-radius: 4px; display: flex; align-items: center; justify-content: center;">
                        <span>No visualizations available</span>
                    </div>
                `;
            }
            
            // Analysis-specific content
            if (data.analysis_type === 'distribution_analysis') {
                // Show summary stats
                if (data.summary_stats && Object.keys(data.summary_stats).length > 0) {
                    contentHtml += `<h5>Statistical Summary</h5><div class="row">`;
                    for (const [col, stats] of Object.entries(data.summary_stats)) {
                        contentHtml += `
                            <div class="col-md-6 mb-3">
                                <div class="card">
                                    <div class="card-header">${col}</div>
                                    <div class="card-body">
                                        <table class="table table-sm">
                                            <tbody>
                                                <tr><td>Min</td><td>${stats.min.toFixed(2)}</td></tr>
                                                <tr><td>Max</td><td>${stats.max.toFixed(2)}</td></tr>
                                                <tr><td>Mean</td><td>${stats.mean.toFixed(2)}</td></tr>
                                                <tr><td>Median</td><td>${stats.median.toFixed(2)}</td></tr>
                                                <tr><td>Std Dev</td><td>${stats.std.toFixed(2)}</td></tr>
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>
                        `;
                    }
                    contentHtml += `</div>`;
                }
            } else if (data.analysis_type === 'correlation_analysis') {
                // Show correlation pairs
                if (data.correlation_pairs && data.correlation_pairs.length > 0) {
                    contentHtml += `
                        <h5>Strong Correlations Found</h5>
                        <table class="table">
                            <thead>
                                <tr>
                                    <th>Variables</th>
                                    <th>Correlation</th>
                                    <th>Direction</th>
                                </tr>
                            </thead>
                            <tbody>
                    `;
                    
                    for (const pair of data.correlation_pairs) {
                        contentHtml += `
                            <tr>
                                <td>${pair.col1} / ${pair.col2}</td>
                                <td>${pair.correlation.toFixed(2)}</td>
                                <td>${pair.direction}</td>
                            </tr>
                        `;
                    }
                    
                    contentHtml += `</tbody></table>`;
                } else {
                    contentHtml += `<p>No significant correlations found in the data.</p>`;
                }
            } else if (data.analysis_type === 'outlier_detection') {
                if (data.outliers_by_column && Object.keys(data.outliers_by_column).length > 0) {
                    contentHtml += `
                        <h5>Outliers Detected</h5>
                        <table class="table">
                            <thead>
                                <tr>
                                    <th>Column</th>
                                    <th>Count</th>
                                    <th>Percentage</th>
                                    <th>Bounds (Lower/Upper)</th>
                                </tr>
                            </thead>
                            <tbody>
                    `;
                    
                    for (const [col, info] of Object.entries(data.outliers_by_column)) {
                        contentHtml += `
                            <tr>
                                <td>${col}</td>
                                <td>${info.count}</td>
                                <td>${info.percentage.toFixed(2)}%</td>
                                <td>${info.bounds.lower.toFixed(2)} / ${info.bounds.upper.toFixed(2)}</td>
                            </tr>
                        `;
                    }
                    
                    contentHtml += `</tbody></table>`;
                    
                    if (data.outlier_indices && data.outlier_indices.length > 0) {
                        contentHtml += `
                            <p>Total outliers found: ${data.outlier_indices.length}</p>
                        `;
                    }
                } else {
                    contentHtml += `<p>No significant outliers detected in the data.</p>`;
                }
            } else {
                // Generic display for other analysis types
                contentHtml += `
                    <div class="alert alert-info">
                        <p>${data.status || 'Analysis completed successfully.'}</p>
                    </div>
                `;
            }
            
            // Add export and refinement options
            contentHtml += `
                <div class="mt-4">
                    <button class="btn btn-primary" onclick="alert('Export functionality not implemented yet')">Export Results</button>
                    <button class="btn btn-outline-primary ms-2" onclick="alert('Refinement functionality not implemented yet')">Refine Analysis</button>
                </div>
            `;
        }
        
        container.innerHTML = contentHtml;
    }
    }
});