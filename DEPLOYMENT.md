# 🚀 Deployment Guide for Image Classifier API

This guide covers multiple ways to deploy your image classification API so it can be reused by other applications.

---

## Option 1: Render.com (Recommended - Free Tier)

### Steps:
1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/image-classifier-api.git
   git push -u origin main
   ```

2. **Deploy on Render:**
   - Go to [render.com](https://render.com) and sign up
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Render will auto-detect the `Dockerfile`
   - Click "Create Web Service"

3. **Your API will be available at:**
   ```
   https://image-classifier-api.onrender.com
   ```

---

## Option 2: Railway.app (Simple & Fast)

### Steps:
1. Go to [railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway auto-detects Dockerfile and deploys

**Pricing:** $5 free credit/month

---

## Option 3: Hugging Face Spaces (Great for ML Models)

### Steps:
1. Create account at [huggingface.co](https://huggingface.co)
2. Create new Space → Select "Docker" SDK
3. Upload your files or connect GitHub

Create this `app.py` for Gradio interface (optional):
```python
import gradio as gr
from classifier import ImageClassifier

classifier = ImageClassifier()

def classify(image):
    import io
    from PIL import Image
    
    # Convert to bytes
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    image_bytes = buffer.getvalue()
    
    result = classifier.predict(image_bytes)
    return {pred['label']: pred['confidence'] for pred in result['predictions']}

demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="Image Classifier",
    description="Upload an image to classify it using MobileNetV2"
)

demo.launch()
```

---

## Option 4: Google Cloud Run (Scalable)

### Steps:
1. Install Google Cloud CLI
2. Build and push Docker image:
   ```bash
   # Build the image
   docker build -t gcr.io/YOUR_PROJECT_ID/image-classifier .
   
   # Push to Google Container Registry
   docker push gcr.io/YOUR_PROJECT_ID/image-classifier
   
   # Deploy to Cloud Run
   gcloud run deploy image-classifier \
     --image gcr.io/YOUR_PROJECT_ID/image-classifier \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 2Gi
   ```

**Pricing:** Pay per request, generous free tier

---

## Option 5: Docker Hub (Share Your Container)

### Steps:
1. Create account at [hub.docker.com](https://hub.docker.com)
2. Build and push:
   ```bash
   # Build
   docker build -t YOUR_USERNAME/image-classifier:latest .
   
   # Login
   docker login
   
   # Push
   docker push YOUR_USERNAME/image-classifier:latest
   ```

3. Others can now use your API:
   ```bash
   docker run -p 5000:5000 YOUR_USERNAME/image-classifier:latest
   ```

---

## 📡 Using Your Deployed API

Once deployed, other applications can use your API like this:

### Python Example:
```python
import requests

# Replace with your deployed URL
API_URL = "https://your-app.onrender.com"

# Classify an image file
with open("image.jpg", "rb") as f:
    response = requests.post(
        f"{API_URL}/predict",
        files={"image": f}
    )
    
result = response.json()
print(f"Prediction: {result['top_prediction']['label']}")
print(f"Confidence: {result['top_prediction']['confidence_percent']}")
```

### JavaScript Example:
```javascript
const formData = new FormData();
formData.append('image', imageFile);

const response = await fetch('https://your-app.onrender.com/predict', {
    method: 'POST',
    body: formData
});

const result = await response.json();
console.log(`Prediction: ${result.top_prediction.label}`);
```

### cURL Example:
```bash
curl -X POST https://your-app.onrender.com/predict \
  -F "image=@/path/to/image.jpg"
```

### From URL:
```bash
curl -X POST https://your-app.onrender.com/predict/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/image.jpg"}'
```

---

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/predict` | POST | Classify uploaded image |
| `/predict/url` | POST | Classify image from URL |

---

## 💡 Tips

1. **Cold Starts:** Free tiers may have cold starts (30-60 sec first request)
2. **Memory:** PyTorch models need ~1-2GB RAM minimum
3. **CPU vs GPU:** CPU inference is slower but cheaper/simpler
4. **Rate Limiting:** Consider adding rate limiting for production

---

## 🔒 Security (Production)

Add these for production deployments:
- API key authentication
- Rate limiting
- HTTPS only
- Input validation

