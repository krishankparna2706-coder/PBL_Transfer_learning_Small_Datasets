# Transfer Learning MVP - Inference API
# Build: docker build -t transfer-learning-mvp .
# Run:   docker run -p 5000:5000 -v $(pwd)/best_model.pth:/app/best_model.pth -v $(pwd)/class_names.json:/app/class_names.json transfer-learning-mvp

FROM python:3.10-slim

WORKDIR /app

# Install system deps if needed (e.g. for Pillow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py data_utils.py model_utils.py app.py ./

# Default port for the API
EXPOSE 5000

# Run the FastAPI app with uvicorn
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
