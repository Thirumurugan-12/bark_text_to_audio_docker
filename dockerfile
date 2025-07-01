# Use an official Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install OS dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg git && \
    apt-get clean

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Optional: download Bark model weights in advance
# RUN python -c "from bark import SAMPLE_RATE, generate_audio; generate_audio('Hello, world!')"

# Set default command to run the script
CMD ["python", "app.py"]
