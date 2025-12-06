FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for numpy, pandas, sklearn, tensorflow-cpu
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libatlas-base-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY final_psobilstm_model.keras .
COPY scaler.pkl .
COPY templates ./templates
COPY static ./static

EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
