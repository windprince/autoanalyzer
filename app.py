from flask import Flask, request, jsonify, render_template, url_for
import os
import json
import logging
from AutoAnalyzer import AutoAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')
analyzer = AutoAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Determine file type
    file_ext = file.filename.split('.')[-1].lower()
    supported_extensions = ['csv', 'json', 'xlsx', 'txt', 'parquet']
    
    if file_ext not in supported_extensions:
        return jsonify({
            "error": f"Unsupported file type. Please upload one of: {', '.join(supported_extensions)}"
        }), 400
    
    # Get analysis objective if provided
    objective = request.form.get('objective', 'general')
    
    # Save file temporarily
    temp_path = f"temp_{os.urandom(8).hex()}.{file_ext}"
    file.save(temp_path)
    
    try:
        # Log the analysis request
        logger.info(f"Analyzing file: {file.filename} (type: {file_ext}, objective: {objective})")
        
        # Analyze the data
        results = analyzer.analyze_data(temp_path, file_ext, objective=objective)
        
        # Convert numpy types to Python native types for JSON serialization
        import json
        from json import JSONEncoder
        
        class NumpyEncoder(JSONEncoder):
            def default(self, obj):
                import numpy as np
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
                
        return app.response_class(
            response=json.dumps(results, cls=NumpyEncoder),
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    """Endpoint to run a specific analysis on already uploaded data"""
    try:
        data = request.json
        analysis_id = data.get('analysis_id')
        dataset_path = data.get('dataset_path')
        params = data.get('parameters', {})
        
        if not analysis_id:
            return jsonify({"error": "No analysis ID provided"}), 400
            
        if not dataset_path:
            return jsonify({"error": "No dataset path provided"}), 400
            
        # Run the specific analysis using the analyzer
        results = analyzer.run_specific_analysis(analysis_id, dataset_path, params)
        
        # Convert numpy types to Python native types for JSON serialization
        import json
        from json import JSONEncoder
        
        class NumpyEncoder(JSONEncoder):
            def default(self, obj):
                import numpy as np
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
                
        return app.response_class(
            response=json.dumps(results, cls=NumpyEncoder),
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        logger.error(f"Run analysis error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True)