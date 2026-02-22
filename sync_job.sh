#!/bin/bash

if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Error: .env not found"
    exit 1
fi

# 2. Fill the template and create a temporary .sub file
# This replaces ${VAR} with actual values from your .env
envsubst < train_model.sub.template > train_model.sub

# 3. Define SIC Cluster details
SUBMIT_NODE="$SSH_USER@$SSH_HOST" # [cite: 19]
REMOTE_DIR="~/NNTIProject"

echo "📤 Transferring files to $SUBMIT_NODE..."

# 4. Create remote directory and transfer files
ssh $SUBMIT_NODE "mkdir -p $REMOTE_DIR/logs"
scp train_model.sub train_model.py $SUBMIT_NODE:$REMOTE_DIR/

# 5. Clean up the local temporary file (so keys don't sit on your disk)
#rm train_model.sub

echo "✅ Done! Now log in and run:"
echo "ssh $SUBMIT_NODE"
echo "cd $REMOTE_DIR && condor_submit train_model.sub"