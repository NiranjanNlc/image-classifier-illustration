"""
Tests for the ImageClassifier class (PyTorch MobileNetV2)
"""
import io
import sys
import os
import pytest
import numpy as np
import torch
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier import ImageClassifier


@pytest.fixture(scope='module')
def classifier():
    """Create a classifier instance (shared across tests for efficiency)"""
    return ImageClassifier()


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes"""
    img = Image.new('RGB', (300, 300), color='green')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes.getvalue()


@pytest.fixture
def grayscale_image_bytes():
    """Create grayscale image bytes"""
    img = Image.new('L', (200, 200), color=128)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.getvalue()


@pytest.fixture
def rgba_image_bytes():
    """Create RGBA image bytes"""
    img = Image.new('RGBA', (250, 250), color=(255, 0, 0, 128))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.getvalue()


class TestClassifierInitialization:
    """Tests for classifier initialization"""
    
    def test_classifier_loads_model(self, classifier):
        """Test that classifier loads the model successfully"""
        assert classifier.is_ready() is True
    
    def test_classifier_model_not_none(self, classifier):
        """Test that model is not None after loading"""
        assert classifier.model is not None
    
    def test_image_size_constant(self, classifier):
        """Test that image size is set correctly"""
        assert classifier.IMAGE_SIZE == (224, 224)


class TestImagePreprocessing:
    """Tests for image preprocessing"""
    
    def test_preprocess_rgb_image(self, classifier, sample_image_bytes):
        """Test preprocessing of RGB image"""
        processed = classifier.preprocess_image(sample_image_bytes)
        
        assert processed is not None
        # PyTorch uses (batch, channels, height, width) format
        assert processed.shape == (1, 3, 224, 224)
    
    def test_preprocess_grayscale_image(self, classifier, grayscale_image_bytes):
        """Test preprocessing of grayscale image (should convert to RGB)"""
        processed = classifier.preprocess_image(grayscale_image_bytes)
        
        assert processed is not None
        assert processed.shape == (1, 3, 224, 224)
    
    def test_preprocess_rgba_image(self, classifier, rgba_image_bytes):
        """Test preprocessing of RGBA image (should convert to RGB)"""
        processed = classifier.preprocess_image(rgba_image_bytes)
        
        assert processed is not None
        assert processed.shape == (1, 3, 224, 224)
    
    def test_preprocess_resizes_image(self, classifier):
        """Test that preprocessing resizes large images"""
        # Create a large image
        img = Image.new('RGB', (1000, 1000), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        
        processed = classifier.preprocess_image(img_bytes.getvalue())
        
        assert processed.shape == (1, 3, 224, 224)
    
    def test_preprocess_output_dtype(self, classifier, sample_image_bytes):
        """Test that preprocessed image has correct dtype"""
        processed = classifier.preprocess_image(sample_image_bytes)
        
        # PyTorch tensors should be float32
        assert processed.dtype == torch.float32


class TestPrediction:
    """Tests for image prediction"""
    
    def test_predict_returns_dict(self, classifier, sample_image_bytes):
        """Test that predict returns a dictionary"""
        result = classifier.predict(sample_image_bytes)
        
        assert isinstance(result, dict)
    
    def test_predict_contains_predictions(self, classifier, sample_image_bytes):
        """Test that result contains predictions list"""
        result = classifier.predict(sample_image_bytes)
        
        assert 'predictions' in result
        assert isinstance(result['predictions'], list)
    
    def test_predict_contains_top_prediction(self, classifier, sample_image_bytes):
        """Test that result contains top prediction"""
        result = classifier.predict(sample_image_bytes)
        
        assert 'top_prediction' in result
        assert isinstance(result['top_prediction'], dict)
    
    def test_prediction_has_required_fields(self, classifier, sample_image_bytes):
        """Test that each prediction has required fields"""
        result = classifier.predict(sample_image_bytes)
        
        for pred in result['predictions']:
            assert 'class_id' in pred
            assert 'label' in pred
            assert 'confidence' in pred
            assert 'confidence_percent' in pred
    
    def test_top_k_parameter(self, classifier, sample_image_bytes):
        """Test that top_k parameter works correctly"""
        result_5 = classifier.predict(sample_image_bytes, top_k=5)
        result_3 = classifier.predict(sample_image_bytes, top_k=3)
        result_10 = classifier.predict(sample_image_bytes, top_k=10)
        
        assert len(result_5['predictions']) == 5
        assert len(result_3['predictions']) == 3
        assert len(result_10['predictions']) == 10
    
    def test_confidence_is_float(self, classifier, sample_image_bytes):
        """Test that confidence values are floats"""
        result = classifier.predict(sample_image_bytes)
        
        for pred in result['predictions']:
            assert isinstance(pred['confidence'], float)
            assert 0 <= pred['confidence'] <= 1
    
    def test_confidence_in_descending_order(self, classifier, sample_image_bytes):
        """Test that predictions are sorted by confidence"""
        result = classifier.predict(sample_image_bytes)
        
        confidences = [p['confidence'] for p in result['predictions']]
        assert confidences == sorted(confidences, reverse=True)
    
    def test_label_formatting(self, classifier, sample_image_bytes):
        """Test that labels are properly formatted"""
        result = classifier.predict(sample_image_bytes)
        
        for pred in result['predictions']:
            # Labels should not contain underscores (replaced with spaces)
            assert '_' not in pred['label'] or pred['label'].replace('_', ' ').title() == pred['label']


class TestModelInfo:
    """Tests for model information"""
    
    def test_get_model_info(self, classifier):
        """Test getting model information"""
        info = classifier.get_model_info()
        
        assert info is not None
        assert info['name'] == 'MobileNetV2'
        assert info['framework'] == 'PyTorch'
        assert info['input_shape'] == (224, 224)
        assert info['num_classes'] == 1000
        assert info['dataset'] == 'ImageNet'
        assert 'device' in info
    
    def test_model_has_parameters(self, classifier):
        """Test that model reports parameters count"""
        info = classifier.get_model_info()
        
        assert 'parameters' in info
        assert isinstance(info['parameters'], int)
        assert info['parameters'] > 0


class TestEdgeCases:
    """Tests for edge cases"""
    
    def test_very_small_image(self, classifier):
        """Test handling of very small images"""
        img = Image.new('RGB', (10, 10), color='yellow')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        
        result = classifier.predict(img_bytes.getvalue())
        assert result is not None
        assert 'predictions' in result
    
    def test_rectangular_image(self, classifier):
        """Test handling of non-square images"""
        img = Image.new('RGB', (640, 480), color='purple')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        
        result = classifier.predict(img_bytes.getvalue())
        assert result is not None
        assert 'predictions' in result
    
    def test_invalid_image_bytes_raises_error(self, classifier):
        """Test that invalid image data raises an error"""
        with pytest.raises(Exception):
            classifier.predict(b'not an image')

