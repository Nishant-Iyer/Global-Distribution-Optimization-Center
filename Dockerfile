# Base Image
FROM python:3.9-slim

# System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Work Directory
WORKDIR /app

# Install dependencies first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and dataset
COPY setup.py .
COPY gdoc_opt/ ./gdoc_opt/
COPY final_dataset.csv .

# Install the package in editable mode
RUN pip install -e .

# Expose ports
EXPOSE 8000 8501

# Command is overridden in docker-compose.yml
CMD ["python"]
