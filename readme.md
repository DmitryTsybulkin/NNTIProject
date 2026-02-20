# NNTI Project

## Setup:
1. Setup virtual env: `python -m venv venv`
2. Activate virtual env: `source ./venv/bin/activate`
3. Install dependencies: `python3 -m pip install -r requirements.txt`

## API Token Management:
1. Copy the example environment file: `cp .env.example .env`
2. Edit `.env` and add your actual API tokens:
   - Get HuggingFace token from: https://huggingface.co/settings/tokens
   - Get WandB key from: https://wandb.ai/authorize

## Deployment to HTCondor Cluster:

### SSH Connection:
```bash
ssh neuronet_team167@conduit2.hpc.uni-saarland.de
```

### Build and Push Docker Image:

**Option 1: Use the deployment script (recommended):**
```bash
./deploy.sh
```

**Option 2: Manual build:**
```bash
# Load tokens from .env file
export $(cat .env | grep -v '^#' | xargs)

# Build the docker image with tokens embedded
docker build \
  --build-arg USER_UID=$UID \
  --build-arg USER_NAME=$(id -un) \
  --build-arg HF_TOKEN=$HF_TOKEN \
  --build-arg WANDB_KEY=$WANDB_KEY \
  -t docker.cs.uni-saarland.de/[lsv_user_name]/[image_name]:[tag_name] \
  -f Dockerfile \
  .

# Push the image to the docker registry
docker push docker.cs.uni-saarland.de/[lsv_user_name]/[image_name]:[tag_name]
```

## Wandb

How to use wandb: <a href="https://www.youtube.com/watch?v=LQvRhQwDOm0">video</a>

## Dataset

https://huggingface.co/datasets/badrex/nnti-dataset-full

## Task 1