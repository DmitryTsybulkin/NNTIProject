#!/bin/bash

# Enable Docker BuildKit for faster builds with caching
export DOCKER_BUILDKIT=1

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Error: .env file not found!"
    echo "Please create .env file with your configuration first."
    exit 1
fi

# Check if Docker registry config is set
if [ -z "$DOCKER_REGISTRY" ] || [ -z "$DOCKER_USER_NAME" ] || [ -z "$DOCKER_IMAGE_NAME" ] || [ -z "$DOCKER_TAG" ]; then
    echo "Error: Docker configuration not set in .env file"
    echo "Required: DOCKER_REGISTRY, DOCKER_USER_NAME, DOCKER_IMAGE_NAME, DOCKER_TAG"
    exit 1
fi

# Build full image name
FULL_IMAGE_NAME="${DOCKER_REGISTRY}/${DOCKER_USER_NAME}/${DOCKER_IMAGE_NAME}:${DOCKER_TAG}"

echo "🔨 Building Docker image..."
docker build \
    -t ${FULL_IMAGE_NAME} \
    -f Dockerfile \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Docker image built successfully!"
    echo ""
    echo "🚀 Pushing to registry..."
    docker push ${FULL_IMAGE_NAME}

    if [ $? -eq 0 ]; then
        echo "✓ Image pushed successfully!"
        echo ""
        echo "════════════════════════════════════════════════════════"
        echo "✅ IMAGE SUCCESSFULLY DEPLOYED!"
        echo "════════════════════════════════════════════════════════"
        echo ""
        echo "📍 Image Location:"
        echo "   ${FULL_IMAGE_NAME}"
        echo ""
        echo "🐳 Pull command:"
        echo "   docker pull ${FULL_IMAGE_NAME}"
        echo ""
        echo "📝 For HTCondor, use this in train_model.sub:"
        echo "   docker_image = ${FULL_IMAGE_NAME}"
        echo ""
        echo "════════════════════════════════════════════════════════"
    else
        echo "❌ Failed to push image to registry"
        exit 1
    fi
else
    echo "❌ Failed to build Docker image"
    exit 1
fi

