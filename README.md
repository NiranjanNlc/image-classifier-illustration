# Image Classifier API

A Flask REST API for image classification using PyTorch MobileNetV2 pre-trained on ImageNet.

## Features

- **Image Classification**: Classify images into 1000 ImageNet categories
- **Multiple Input Methods**: Upload images directly or provide URLs
- **MobileNetV2 (PyTorch)**: Uses efficient MobileNetV2 architecture for fast inference
- **GPU Support**: Automatically uses CUDA if available
- **RESTful API**: Clean JSON responses with confidence scores

## Installation

### 1. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the API

### Development Server

```bash
python app.py
```

The API will be available at `http://localhost:5000`

### Production (with Gunicorn)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## API Endpoints

### GET `/`
Returns API information and available endpoints.

### GET `/health`
Health check endpoint. Returns model status.

### POST `/predict`
Classify an uploaded image.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `image` field with the image file

**Example with cURL:**
```bash
curl -X POST -F "image=@your_image.jpg" http://localhost:5000/predict
```

**Response:**
```json
{
  "success": true,
  "predictions": [
    {
      "class_id": "n02123045",
      "label": "Tabby",
      "confidence": 0.7823,
      "confidence_percent": "78.23%"
    },
    ...
  ],
  "top_prediction": {
    "label": "Tabby",
    "confidence": 0.7823,
    "confidence_percent": "78.23%"
  },
  "image_info": {
    "filename": "your_image.jpg",
    "size_bytes": 102400
  }
}
```

### POST `/predict/url`
Classify an image from a URL.

**Request:**
- Content-Type: `application/json`
- Body: `{"url": "https://example.com/image.jpg"}`

**Example with cURL:**
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/image.jpg"}' \
  http://localhost:5000/predict/url
```

## Testing

### Run All Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=. --cov-report=html
```

### Run Specific Test File

```bash
pytest tests/test_api.py
pytest tests/test_classifier.py
```

## Supported Image Formats

- JPEG/JPG
- PNG
- GIF
- WebP

## Response Codes

| Code | Description |
|------|-------------|
| 200  | Success |
| 400  | Bad Request (missing image, invalid file type, etc.) |
| 500  | Server Error |

## Configuration

Environment variables:
- `PORT`: Server port (default: 5000)
- `FLASK_DEBUG`: Enable debug mode (default: False)

## Project Structure

```
backend/
├── app.py              # Flask application
├── classifier.py       # ImageClassifier class
├── requirements.txt    # Python dependencies
├── pytest.ini          # Pytest configuration
├── README.md           # This file
└── tests/
    ├── __init__.py
    ├── test_api.py     # API endpoint tests
    └── test_classifier.py  # Classifier tests
```

## Model Information

- **Model**: MobileNetV2
- **Framework**: PyTorch (torchvision)
- **Pre-trained on**: ImageNet (1000 classes)
- **Input size**: 224x224 RGB images
- **Architecture**: Efficient inverted residual structure
- **GPU Support**: Automatically uses CUDA if available

## License

MIT License

