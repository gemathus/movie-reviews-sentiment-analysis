FROM --platform=linux/amd64 python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY inference.py .

# Ensure model storage exists
RUN mkdir -p ./model-storage

ENV TORCH_CPU_CAPABILITY=default

# Command to run the FastAPI app
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]

