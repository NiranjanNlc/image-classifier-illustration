"""
Tests for the Flask API endpoints
"""
import io
import sys
import os
import pytest
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app


@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_image():
    """Create a sample test image"""
    # Create a simple RGB image
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture
def sample_png_image():
    """Create a sample PNG test image"""
    img = Image.new('RGBA', (300, 300), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes


class TestRootEndpoint:
    """Tests for the root endpoint"""
    
    def test_root_returns_api_info(self, client):
        """Test that root endpoint returns API information"""
        response = client.get('/')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['name'] == 'Image Classifier API'
        assert data['model'] == 'MobileNetV2 (ImageNet)'
        assert 'endpoints' in data
        assert 'usage' in data


class TestHealthEndpoint:
    """Tests for the health check endpoint"""
    
    def test_health_check(self, client):
        """Test health check endpoint returns correct status"""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['status'] == 'healthy'
        assert data['model'] == 'MobileNetV2'
        assert 'ready' in data


class TestPredictEndpoint:
    """Tests for the /predict endpoint"""
    
    def test_predict_without_image(self, client):
        """Test prediction without image returns error"""
        response = client.post('/predict')
        assert response.status_code == 400
        
        data = response.get_json()
        assert data['success'] is False
        assert 'error' in data
    
    def test_predict_with_empty_filename(self, client):
        """Test prediction with empty filename returns error"""
        response = client.post(
            '/predict',
            data={'image': (io.BytesIO(b''), '')},
            content_type='multipart/form-data'
        )
        assert response.status_code == 400
        
        data = response.get_json()
        assert data['success'] is False
    
    def test_predict_with_invalid_file_type(self, client):
        """Test prediction with invalid file type returns error"""
        response = client.post(
            '/predict',
            data={'image': (io.BytesIO(b'test'), 'test.txt')},
            content_type='multipart/form-data'
        )
        assert response.status_code == 400
        
        data = response.get_json()
        assert data['success'] is False
        assert 'Invalid file type' in data['error']
    
    def test_predict_with_valid_jpeg(self, client, sample_image):
        """Test prediction with valid JPEG image"""
        response = client.post(
            '/predict',
            data={'image': (sample_image, 'test.jpg')},
            content_type='multipart/form-data'
        )
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['success'] is True
        assert 'predictions' in data
        assert 'top_prediction' in data
        assert 'image_info' in data
        
        # Check predictions structure
        predictions = data['predictions']
        assert len(predictions) > 0
        assert 'label' in predictions[0]
        assert 'confidence' in predictions[0]
        assert 'confidence_percent' in predictions[0]
        assert 'class_id' in predictions[0]
        
        # Check top prediction
        top_pred = data['top_prediction']
        assert 'label' in top_pred
        assert 'confidence' in top_pred
        assert isinstance(top_pred['confidence'], float)
    
    def test_predict_with_valid_png(self, client, sample_png_image):
        """Test prediction with valid PNG image"""
        response = client.post(
            '/predict',
            data={'image': (sample_png_image, 'test.png')},
            content_type='multipart/form-data'
        )
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['success'] is True
        assert 'predictions' in data
    
    def test_predict_returns_multiple_predictions(self, client, sample_image):
        """Test that prediction returns multiple results"""
        response = client.post(
            '/predict',
            data={'image': (sample_image, 'test.jpg')},
            content_type='multipart/form-data'
        )
        
        data = response.get_json()
        predictions = data['predictions']
        
        # Should return top 5 predictions by default
        assert len(predictions) == 5
        
        # Confidence should be in descending order
        confidences = [p['confidence'] for p in predictions]
        assert confidences == sorted(confidences, reverse=True)


class TestPredictUrlEndpoint:
    """Tests for the /predict/url endpoint"""
    
    def test_predict_url_without_url(self, client):
        """Test prediction without URL returns error"""
        response = client.post('/predict/url', json={})
        assert response.status_code == 400
        
        data = response.get_json()
        assert data['success'] is False
    
    def test_predict_url_with_empty_body(self, client):
        """Test prediction with empty body returns error"""
        response = client.post('/predict/url')
        # Flask returns 415 (Unsupported Media Type) when no JSON body is provided
        assert response.status_code in [400, 415]


class TestImageInfo:
    """Tests for image info in responses"""
    
    def test_image_info_contains_filename(self, client, sample_image):
        """Test that response includes filename"""
        response = client.post(
            '/predict',
            data={'image': (sample_image, 'my_image.jpg')},
            content_type='multipart/form-data'
        )
        
        data = response.get_json()
        assert data['image_info']['filename'] == 'my_image.jpg'
    
    def test_image_info_contains_size(self, client, sample_image):
        """Test that response includes file size"""
        response = client.post(
            '/predict',
            data={'image': (sample_image, 'test.jpg')},
            content_type='multipart/form-data'
        )
        
        data = response.get_json()
        assert 'size_bytes' in data['image_info']
        assert isinstance(data['image_info']['size_bytes'], int)
        assert data['image_info']['size_bytes'] > 0

