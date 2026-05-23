FROM python:3.13-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-app.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-app.txt

# Copy only the runtime source files needed for Streamlit
COPY app/ ./app
COPY src/ ./src

# Expose port
EXPOSE 8501

# Run the App
CMD ["streamlit", "run", "app/app_lc.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
