# Use a lightweight Python image
FROM --platform=linux/amd64 python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependency list
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script and supporting files
COPY train.py .

# Create model-storage directory (we still need this for saving the model)
RUN mkdir -p ./model-storage

ENV TORCH_CPU_CAPABILITY=default

# Set the command to run the training script
CMD ["python", "train.py"]

