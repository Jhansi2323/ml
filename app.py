from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

import os

app = Flask(__name__, 
    template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend', 'templates'),
    static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
)

# Load trained ML model
model = pickle.load(open("predictive_maintenance_xgb.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        # Read the CSV file
        data = pd.read_csv(file)
        
        # Print column names for debugging
        print("Columns in uploaded file:", data.columns.tolist())
        
        # Map the expected columns to the actual columns in the dataset
        # These should match the features used during model training
        feature_columns = [
            'Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]',
            'TWF',  # Tool Wear Failure
            'HDF',  # Heat Dissipation Failure
            'PWF',  # Power Failure
            'OSF',  # Overstrain Failure
            'RNF',  # Random Failure
            'Type'  # Product type (L, M, H)
        ]
        
        # Convert Type to numerical values (L=0, M=1, H=2)
        if 'Type' in data.columns:
            data['Type'] = data['Type'].map({'L': 0, 'M': 1, 'H': 2})
        
        # Check if all required columns exist
        missing_columns = [col for col in feature_columns if col not in data.columns]
        if missing_columns:
            return jsonify({
                'error': f'Missing required columns: {missing_columns}',
                'available_columns': data.columns.tolist()
            }), 400
        
        # Select the features
        features = data[feature_columns]
        
        # Make prediction
        predictions = model.predict(features)
        
        # Get basic statistics
        avg_prediction = float(predictions.mean())
        min_pred = float(predictions.min())
        max_pred = float(predictions.max())
        
        # Print prediction statistics for debugging
        print(f"Prediction stats - Min: {min_pred}, Max: {max_pred}, Mean: {avg_prediction}")
        
        # Calculate a dynamic threshold based on the prediction distribution
        import numpy as np
        
        # Calculate percentiles to understand the distribution
        p75 = np.percentile(predictions, 75)
        p95 = np.percentile(predictions, 95)
        
        # Use 90th percentile for more sensitive failure detection
        # This will flag the top 10% of predictions as potential failures
        threshold = np.percentile(predictions, 90)
        
        # Ensure threshold is reasonable compared to the max prediction
        threshold = min(threshold, max_pred * 0.95)  # Don't get too close to max
        threshold = max(threshold, p75 * 1.2)  # At least 20% above 75th percentile
        
        # Get binary predictions based on threshold
        binary_predictions = (predictions > threshold).astype(int)
        
        # Count failures
        failure_count = int(sum(binary_predictions))
        total_samples = len(predictions)
        failure_percentage = float(failure_count / total_samples * 100) if total_samples > 0 else 0
        
        # Get sample of predictions for debugging
        sample_predictions = predictions[:5].tolist()  # First 5 predictions
        
        # Convert all numpy types to native Python types for JSON serialization
        response_data = {
            'status': 'success',
            'prediction': 'Machine Failure Detected' if failure_count > 0 else 'No Failure Detected',
            'failure_count': int(failure_count),
            'total_samples': int(total_samples),
            'failure_percentage': float(failure_percentage),
            'average_prediction': float(avg_prediction),
            'prediction_range': {
                'min': float(min_pred),
                'max': float(max_pred)
            },
            'sample_predictions': [float(x) for x in sample_predictions],
            'threshold_used': float(threshold),
            'percentiles': {
                'p75': float(p75),
                'p95': float(p95),
                'p97_5': float(threshold)
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        error_message = str(e)
        error_traceback = traceback.format_exc()
        print(f"Error in /predict: {error_message}\n{error_traceback}")
        return jsonify({
            'status': 'error',
            'error': error_message,
            'traceback': error_traceback
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
