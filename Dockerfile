FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get install poppler-utils 

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY .env .

# Create upload directory
RUN mkdir doc

# Expose Streamlit port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "copal_multirag.py", "--server.port=8501", "--server.address=0.0.0.0"]
