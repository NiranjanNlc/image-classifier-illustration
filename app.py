"""
Flask API for Image Classification using MobileNetV2
"""
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from classifier import ImageClassifier

app = Flask(__name__)
CORS(app)

# Initialize the classifier (loads the model once at startup)
classifier = ImageClassifier()

# Configure upload settings
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'MobileNetV2',
        'ready': classifier.is_ready()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict endpoint - accepts an image and returns classification results
    
    Expected: multipart/form-data with 'image' field containing the image file
    
    Returns:
        JSON with predictions including label, confidence, and related data
    """
    # Check if image was provided
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No image file provided',
            'message': 'Please provide an image file with key "image"'
        }), 400
    
    file = request.files['image']
    
    # Check if file was selected
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected',
            'message': 'Please select an image file to upload'
        }), 400
    
    # Validate file type
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': 'Invalid file type',
            'message': f'Allowed file types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    try:
        # Read image data
        image_data = file.read()
        
        # Get predictions
        result = classifier.predict(image_data)
        
        return jsonify({
            'success': True,
            'predictions': result['predictions'],
            'top_prediction': result['top_prediction'],
            'image_info': {
                'filename': file.filename,
                'size_bytes': len(image_data)
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route('/predict/url', methods=['POST'])
def predict_from_url():
    """
    Predict endpoint for image URLs
    
    Expected: JSON body with 'url' field containing the image URL
    
    Returns:
        JSON with predictions including label, confidence, and related data
    """
    data = request.get_json()
    
    if not data or 'url' not in data:
        return jsonify({
            'success': False,
            'error': 'No URL provided',
            'message': 'Please provide an image URL in the request body'
        }), 400
    
    try:
        import requests
        
        # Download image from URL
        response = requests.get(data['url'], timeout=10)
        response.raise_for_status()
        
        image_data = response.content
        
        # Get predictions
        result = classifier.predict(image_data)
        
        return jsonify({
            'success': True,
            'predictions': result['predictions'],
            'top_prediction': result['top_prediction'],
            'image_info': {
                'url': data['url'],
                'size_bytes': len(image_data)
            }
        })
        
    except requests.exceptions.RequestException as e:
        return jsonify({
            'success': False,
            'error': 'Failed to fetch image from URL',
            'message': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API information"""
    return jsonify({
        'name': 'Image Classifier API',
        'version': '1.0.0',
        'model': 'MobileNetV2 (ImageNet)',
        'endpoints': {
            '/': 'API information (GET)',
            '/health': 'Health check (GET)',
            '/predict': 'Classify image from file upload (POST)',
            '/predict/url': 'Classify image from URL (POST)'
        },
        'usage': {
            '/predict': 'POST multipart/form-data with "image" field',
            '/predict/url': 'POST JSON with "url" field'
        }
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)

