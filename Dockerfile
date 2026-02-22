FROM --platform=linux/amd64 python:3.10-slim

RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy the training script
COPY train_model.py .

# Run the training script
ENTRYPOINT ["python", "train_model.py"]
