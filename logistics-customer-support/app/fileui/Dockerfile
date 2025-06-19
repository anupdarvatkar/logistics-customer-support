# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local directory to the working directory
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8080

# Command to run the Streamlit app
CMD ["streamlit", "run", "fileui.py", "--server.port=8080", "--server.address=0.0.0.0"]
