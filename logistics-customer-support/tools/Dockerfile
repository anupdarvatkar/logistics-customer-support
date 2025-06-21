# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local directory to the working directory
COPY . .

# Set environment variables
ENV OCR_AGENT_URL="https://ocr-agent-service-203057862897.us-central1.run.app:8080"
ENV GCS_BUCKET_NAME="logistics_customer_support_bucket"


EXPOSE 8080

CMD ["python", "-m", "mcp_server"]