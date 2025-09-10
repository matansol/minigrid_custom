FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY tutorial_requirements.txt .
RUN pip install --no-cache-dir -r tutorial_requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
# EXPOSE 8001
EXPOSE 8002

# Command to run the application
# CMD ["python", "tutorial_app.py"] 
CMD ["python", "final_app.py"]