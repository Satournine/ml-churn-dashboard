# Use official Streamlit base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose port for Streamlit
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]