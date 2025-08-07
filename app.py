from flask import Flask, request, jsonify, render_template
import joblib
import traceback
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np

app = Flask(__name__)

# Load all models and their configurations
MODELS = {
    'primary': {
        'model': joblib.load('primary_features_model.pkl'),
        'label_encoders': joblib.load('primary_features_label_encoders.pkl'),
        'name': 'Primary Features Model',
        'accuracy': '83.41%'
    },
    'secondary': {
        'model': joblib.load('secondary_features_model.pkl') if os.path.exists('secondary_features_model.pkl') else None,
        'label_encoders': joblib.load('secondary_features_label_encoders.pkl') if os.path.exists('secondary_features_label_encoders.pkl') else None,
        'name': 'Secondary Features Model',
        'accuracy': '85%'
    }
}

def clean_ethnicity(ethnicity_str):
    """Cleans the ethnicity string (lowercase and stripped)."""
    return str(ethnicity_str).strip().lower()


def transform_feature(label_encoders, feature_name, value, dataset_type):
    """Helper function to transform features with proper error handling."""
    try:
        value = str(value).strip().lower()
        
        if dataset_type == 'secondary' and feature_name == 'Ethnicity':
            # Special handling for secondary dataset ethnicity
            # First try to convert to int if possible (some encoders might expect this)
            try:
                return int(value)
            except ValueError:
                # If not an integer, try to transform as string
                if value in label_encoders[feature_name].classes_:
                    return label_encoders[feature_name].transform([value])[0]
                # Fallback to first category if not found
                return label_encoders[feature_name].transform([label_encoders[feature_name].classes_[0]])[0]
        
        # Default handling for other features
        if value not in label_encoders[feature_name].classes_:
            # Use the first category as fallback
            value = label_encoders[feature_name].classes_[0]
        return label_encoders[feature_name].transform([value])[0]
    
    except Exception as e:
        print(f"Error transforming {feature_name}: {e}")
        raise ValueError(f"Could not transform {feature_name} value: {value}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/model-info')
def model_info():
    dataset = request.args.get('dataset', 'primary')
    if dataset not in MODELS or not MODELS[dataset]['model']:
        return jsonify({'error': f'Invalid dataset: {dataset}'}), 400
    
    return jsonify({
        'model_name': MODELS[dataset]['name'],
        'model_accuracy': MODELS[dataset]['accuracy']
    })
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        dataset = data.get('dataset', 'primary')
        
        if dataset not in MODELS or not MODELS[dataset]['model']:
            return jsonify({'error': f'Invalid dataset: {dataset}'}), 400
            
        model = MODELS[dataset]['model']
        label_encoders = MODELS[dataset]['label_encoders']

        # Basic validation
        age = int(data['age'])
        if age < 1 or age > 20:
            return jsonify({'error': 'Age must be between 1 and 20'}), 400

        # Transform features with dataset-specific handling
        try:
            gender_str = data['gender'].lower()
            gender = transform_feature(label_encoders, 'Sex', 
                                     'm' if gender_str == 'male' else 'f', 
                                     dataset)

            ethnicity_str = data['ethnicity'].lower()
            ethnicity = transform_feature(label_encoders, 'Ethnicity', ethnicity_str, dataset)

            jaundice = transform_feature(label_encoders, 'Jaundice', 
                                       data['jaundice'].lower(), dataset)
            
            autism = transform_feature(label_encoders, 'Family_mem_with_ASD', 
                                     data['austim'].lower(), dataset)

            # Ensure question responses are valid
            a1 = int(data['a1'])
            a4 = int(data['a4'])
            a10 = int(data['a10'])
            if a1 not in [0, 1] or a4 not in [0, 1] or a10 not in [0, 1]:
                return jsonify({'error': 'Question responses must be 0 or 1'}), 400

            # Calculate percentage of "yes" answers
            yes_count = a1 + a4 + a10
            percentage = round((yes_count / 3) * 100, 2)

            # Prepare features and predict
            features = [[age, gender, ethnicity, jaundice, autism, a1, a4, a10]]
            prediction = model.predict(features)
            
            result = "Likely to have Autism (Positive)" if prediction[0] == 1 else "Not likely to have Autism (Negative)"
            
            return jsonify({
                'result': result, 
                'yes_percentage': percentage,
                'model_name': MODELS[dataset]['name'],
                'model_accuracy': MODELS[dataset]['accuracy']
            })

        except ValueError as ve:
            return jsonify({'error': f'Invalid input data: {str(ve)}'}), 400

    except KeyError as ke:
        return jsonify({'error': f'Missing required field: {str(ke)}'}), 400
    except Exception as e:
        print("Prediction error:", e)
        traceback.print_exc()
        return jsonify({'error': 'Prediction failed. Please check the input data and try again.'}), 500

if __name__ == '__main__':
    app.run(debug=True)