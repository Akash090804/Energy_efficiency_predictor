FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Expose port
EXPOSE 8001

# Run app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]