# We use Python 3.12-slim as a stable, lightweight base image
FROM python:3.12-slim

# We set the working directory inside the container
WORKDIR /app

# We install system-level build tools needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# We copy and install Python dependencies first to leverage Docker layer caching
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# We copy source files into the image
COPY backend/ /app/backend/
COPY data/    /app/data/

# We run the training script at build time so the model is baked into the image
RUN python /app/backend/train.py

# We expose the port that uvicorn will listen on
EXPOSE 8000

# We set environment variables used by both train.py and main.py
ENV DATA_PATH=/app/data/raw/house_prices.csv \
    MODEL_PATH=/app/backend/model.joblib      \
    META_PATH=/app/backend/feature_names.json \
    PYTHONUNBUFFERED=1

# We start the FastAPI application from the backend directory so relative imports resolve
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app/backend"]
