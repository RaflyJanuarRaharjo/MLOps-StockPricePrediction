# Dockerfile
# ============================================================
# Container untuk API Inferensi Model AAPL
# Rafly Januar Raharjo - 235150201111011
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    mlflow==2.10.0 \
    scikit-learn==1.4.0 \
    pandas==2.1.4 \
    numpy==1.26.3 \
    yfinance==0.2.36

# Copy source code
COPY src/ ./src/
COPY models/ ./models/
COPY inference.py .

# Expose port
EXPOSE 8000

# Jalankan FastAPI
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
