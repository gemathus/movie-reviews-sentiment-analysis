from fastapi import FastAPI, HTTPException
import torch
import torch.nn as nn
from pydantic import BaseModel
import numpy as np

# Define the model architecture (same as in training)
class MovieReviewNN(nn.Module):
    def __init__(self, vocab_size):
        super(MovieReviewNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(vocab_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

# Define request model
class ReviewRequest(BaseModel):
    text: str

# Initialize FastAPI
app = FastAPI(title="Movie Review Sentiment Analysis")

# Global variables for model and vectorizer
model = None
vectorizer = None

@app.on_event("startup")
async def load_model():
    global model, vectorizer
    try:
        # Load the saved model and vectorizer
        checkpoint = torch.load("./model-storage/movie_sentiment_model.pt")
        vectorizer = checkpoint['vectorizer']
        
        # Initialize model
        model = MovieReviewNN(len(vectorizer.get_feature_names_out()))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

@app.post("/predict")
async def predict(request: ReviewRequest):
    if not model or not vectorizer:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Vectorize the input
        features = vectorizer.transform([request.text]).toarray()
        features_tensor = torch.FloatTensor(features)
        
        # Get prediction
        with torch.no_grad():
            output = model(features_tensor)
            prediction = output.item()
        
        # Get sentiment and confidence
        sentiment = "positive" if prediction > 0.5 else "negative"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        return {
            "sentiment": sentiment,
            "confidence": float(confidence * 100)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
