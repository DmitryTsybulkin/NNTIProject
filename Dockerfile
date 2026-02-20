FROM --platform=linux/amd64 python:3.10-slim

# Accept build arguments for API tokens
ARG HF_TOKEN
ARG WANDB_KEY
ARG USER_UID=1000
ARG USER_NAME=user

# Set them as environment variables
ENV HF_TOKEN=${HF_TOKEN}
ENV WANDB_KEY=${WANDB_KEY}

# Create user with specified UID and name
RUN groupadd -g ${USER_UID} ${USER_NAME} && \
    useradd -m -u ${USER_UID} -g ${USER_NAME} ${USER_NAME}

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script
COPY train_model.py .

# Switch to non-root user
USER ${USER_NAME}

# Run the training script
CMD ["python", "train_model.py"]
