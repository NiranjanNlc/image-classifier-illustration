"""
Image Classifier using MobileNetV2 (PyTorch) pre-trained on ImageNet
"""
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class ImageClassifier:
    """
    Image classifier using MobileNetV2 pre-trained on ImageNet dataset.
    
    Provides classification of images into 1000 ImageNet categories.
    """
    
    # MobileNetV2 expects 224x224 images
    IMAGE_SIZE = (224, 224)
    
    # ImageNet normalization values
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(self):
        """Initialize the classifier and load the MobileNetV2 model"""
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.labels = None
        self._load_model()
        self._load_labels()
    
    def _load_model(self):
        """Load the MobileNetV2 model with ImageNet weights"""
        try:
            print(f"Loading MobileNetV2 model on {self.device}...")
            
            # Load pre-trained MobileNetV2
            self.weights = MobileNet_V2_Weights.IMAGENET1K_V2
            self.model = mobilenet_v2(weights=self.weights)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Move to appropriate device
            self.model.to(self.device)
            
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _load_labels(self):
        """Load ImageNet class labels"""
        # Use the labels from the weights metadata
        self.labels = self.weights.meta["categories"]
    
    def is_ready(self):
        """Check if the model is loaded and ready"""
        return self.model is not None
    
    def get_transform(self):
        """Get the preprocessing transform pipeline"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.IMAGENET_MEAN,
                std=self.IMAGENET_STD
            )
        ])
    
    def preprocess_image(self, image_data):
        """
        Preprocess image data for MobileNetV2
        
        Args:
            image_data: Raw bytes of the image file
            
        Returns:
            Preprocessed torch tensor ready for model prediction
        """
        # Load image from bytes
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply preprocessing transforms
        transform = self.get_transform()
        img_tensor = transform(image)
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)
        
        # Move to device
        img_tensor = img_tensor.to(self.device)
        
        return img_tensor
    
    def predict(self, image_data, top_k=5):
        """
        Classify an image and return top predictions
        
        Args:
            image_data: Raw bytes of the image file
            top_k: Number of top predictions to return (default: 5)
            
        Returns:
            Dictionary containing:
                - predictions: List of top_k predictions with label and confidence
                - top_prediction: The highest confidence prediction
        """
        if not self.is_ready():
            raise RuntimeError("Model is not loaded")
        
        # Preprocess the image
        processed_image = self.preprocess_image(image_data)
        
        # Make prediction (no gradient computation needed)
        with torch.no_grad():
            outputs = self.model(processed_image)
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(outputs, dim=1)
        
        # Get top k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # Move to CPU and convert to numpy
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        # Format results
        results = []
        for idx, prob in zip(top_indices, top_probs):
            label = self.labels[idx]
            results.append({
                'class_id': f'class_{idx}',
                'label': label.replace('_', ' ').title(),
                'confidence': float(prob),
                'confidence_percent': f"{prob * 100:.2f}%"
            })
        
        return {
            'predictions': results,
            'top_prediction': {
                'label': results[0]['label'],
                'confidence': results[0]['confidence'],
                'confidence_percent': results[0]['confidence_percent']
            }
        }
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.is_ready():
            return None
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            'name': 'MobileNetV2',
            'framework': 'PyTorch',
            'input_shape': self.IMAGE_SIZE,
            'num_classes': 1000,
            'dataset': 'ImageNet',
            'parameters': total_params,
            'device': str(self.device)
        }
