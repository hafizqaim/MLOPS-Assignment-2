# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

# Create models directory if it doesn't exist
RUN mkdir -p models

# Run training script
CMD ["python", "src/train.py"]
