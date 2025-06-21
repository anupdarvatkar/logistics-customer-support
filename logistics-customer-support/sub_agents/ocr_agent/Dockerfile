# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app


COPY . .
COPY ./a2a_common-0.1.0-py3-none-any.whl /app/a2a_common-0.1.0-py3-none-any.whl
# Install any dependencies
ENV PYTHONPATH=/app:$PYTHONPATH
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local directory to the working directory


EXPOSE 8080

CMD ["python", "-m", "a2a_server"]