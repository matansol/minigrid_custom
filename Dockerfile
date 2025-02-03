# # Use an official Python image as a base
# FROM python:3.11

# # Install Fontconfig for font management
# RUN apt-get update && apt-get install -y \
#     fontconfig

# # Set up the virtual environment
# RUN python -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# # Install Python dependencies from requirements.txt
# COPY requirements.txt .
# RUN pip install -r requirements.txt

# # Copy your application code to the container
# COPY . /app
# WORKDIR /app

# # Copy the model files to the container
# COPY models ./models

# # Expose the port your app runs on
# EXPOSE 8000

# # Specify the command to run your application
# CMD ["gunicorn", "-w", "1", "--threads", "3", "--max-requests-jitter", "100", "-k", "gevent", "--timeout", "5000", "-b", "0.0.0.0:8000", "app:app"]



#FastAPI
# Use an official Python image as a base
FROM python:3.11

# Install Fontconfig for font management
RUN apt-get update && apt-get install -y \
    fontconfig

# Set up the virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code to the container
COPY . /app
WORKDIR /app

# Copy the model files (if not already copied above)
COPY models ./models

# Expose the port your app runs on
EXPOSE 8000

# Command to run your application with Gunicorn & Uvicorn
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "app:socket_app"]
