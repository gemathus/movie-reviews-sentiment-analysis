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

### Model Inference
