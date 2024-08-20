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
RUN pip install -r requirements.txt

# Copy your application code to the container
COPY . /app
WORKDIR /app

# Specify the command to run your application
CMD ["python", "app.py"]
