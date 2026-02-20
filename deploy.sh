#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Error: .env file not found!"
    exit 1
fi

# Check if tokens are set
if [ -z "$HF_TOKEN" ] || [ -z "$WANDB_KEY" ]; then
    echo "Error: HF_TOKEN or WANDB_KEY not set in .env file"
    exit 1
fi

# Configuration
DOCKER_REGISTRY="docker.cs.uni-saarland.de"
USER_NAME="dmts00001"  # Replace with your actual LSV username
IMAGE_NAME="nnti-slid-training"
TAG_NAME="latest"

FULL_IMAGE_NAME="${DOCKER_REGISTRY}/${USER_NAME}/${IMAGE_NAME}:${TAG_NAME}"

echo "Building Docker image: ${FULL_IMAGE_NAME}"
echo "Using UID: ${UID}"
echo "Using username: $(id -un)"

# Build the Docker image with secrets
docker build \
    --build-arg USER_UID=${UID} \
    --build-arg USER_NAME=$(id -un) \
    --build-arg HF_TOKEN=${HF_TOKEN} \
    --build-arg WANDB_KEY=${WANDB_KEY} \
    -t ${FULL_IMAGE_NAME} \
    -f Dockerfile \
    .

if [ $? -eq 0 ]; then
    echo "✓ Docker image built successfully!"
    echo ""
    echo "Pushing to registry..."
    docker push ${FULL_IMAGE_NAME}

    if [ $? -eq 0 ]; then
        echo "✓ Image pushed successfully!"
        echo ""
        echo "Image available at: ${FULL_IMAGE_NAME}"
    else
        echo "✗ Failed to push image"
        exit 1
    fi
else
    echo "✗ Failed to build Docker image"
    exit 1
fi

