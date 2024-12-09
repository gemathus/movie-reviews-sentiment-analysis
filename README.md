# Movie Review Sentiment Analysis

## Usage
**The external IP address of the inference service is 34.123.70.73**

```bash
 curl -X POST "http://34.123.70.73/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was fantastic! The acting was superb and the plot kept me engaged throughout."}'
```
**Output**
```json
{
    "sentiment":"positive",
    "confidence":99.16995763778687
} 
```

## Data

### Dataset

### Data Pipeline

## Model

### Model Training
```text
Loading IMDB dataset...
Preparing datasets...
Starting training...
Epoch 1/5, Batch 0, Loss: 0.6897
Epoch 1/5, Batch 100, Loss: 0.3312
Epoch 1/5, Batch 200, Loss: 0.3329
Epoch 1/5, Batch 300, Loss: 0.2270
Epoch 1 - Avg Loss: 0.3913, Val Loss: 0.2758, Val Accuracy: 89.31%
Epoch 2/5, Batch 0, Loss: 0.1624
Epoch 2/5, Batch 100, Loss: 0.1573
Epoch 2/5, Batch 200, Loss: 0.0902
Epoch 2/5, Batch 300, Loss: 0.0761
Epoch 2 - Avg Loss: 0.1759, Val Loss: 0.2944, Val Accuracy: 88.94%
Epoch 3/5, Batch 0, Loss: 0.0830
Epoch 3/5, Batch 100, Loss: 0.0428
Epoch 3/5, Batch 200, Loss: 0.0284
Epoch 3/5, Batch 300, Loss: 0.1102
Epoch 3 - Avg Loss: 0.0962, Val Loss: 0.4721, Val Accuracy: 87.96%
Epoch 4/5, Batch 0, Loss: 0.0185
Epoch 4/5, Batch 100, Loss: 0.0263
Epoch 4/5, Batch 200, Loss: 0.0128
Epoch 4/5, Batch 300, Loss: 0.0381
Epoch 4 - Avg Loss: 0.0575, Val Loss: 0.5573, Val Accuracy: 87.01%
Epoch 5/5, Batch 0, Loss: 0.0228
Epoch 5/5, Batch 100, Loss: 0.0088
Epoch 5/5, Batch 200, Loss: 0.0960
Epoch 5/5, Batch 300, Loss: 0.0045
Epoch 5 - Avg Loss: 0.0295, Val Loss: 0.6997, Val Accuracy: 87.21%
Saving model and vectorizer...
Model and vectorizer saved to ./model-storage/movie_sentiment_model.pt
```

### Model Inference
