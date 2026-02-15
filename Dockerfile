# Use a Python base image
FROM python:3.9-slim

# Set the directory inside the container
WORKDIR /app

# Copy your requirements file first (to cache layers)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project code
COPY . .

# Command to run your training script
CMD ["python", "train.py"]