#!/bin/bash

# Configuration
# TODO: Replace with your actual Space SSH connection string from the "Dev Mode" modal
# Example: REMOTE_HOST="user@ssh.hf.space" and export SSH_PORT="22"
REMOTE_USER="user"
REMOTE_HOST="ssh.hf.space"
# Note: HF Spaces usually use port 22 or a custom port provided in the modal. 
# We'll assume the user provides the specific connection string or we use a config file.
# For now, we will assume standard HF SSH hostname and let user configure specific port via env var or arg.

REMOTE_BASE="/data"
LOCAL_BASE="./data"

# Ensure local directories exist
mkdir -p "$LOCAL_BASE/inputs"
mkdir -p "$LOCAL_BASE/outputs"

# Usage check
if [ "$1" == "up" ]; then
    echo "🚀 Syncing INPUTS UP to Hugging Face Space..."
    # rsync flags: -a (archive), -v (verbose), -z (compress), -P (progress)
    # Exclude hidden files
    rsync -avzP -e "ssh -o StrictHostKeyChecking=no" "$LOCAL_BASE/inputs/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_BASE/inputs/"
    
elif [ "$1" == "down" ]; then
    echo "⬇️ Syncing OUTPUTS DOWN from Hugging Face Space..."
    rsync -avzP -e "ssh -o StrictHostKeyChecking=no" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_BASE/outputs/" "$LOCAL_BASE/outputs/"
    
else
    echo "Usage: $0 {up|down}"
    echo "  up   : Upload local inputs to cloud"
    echo "  down : Download cloud outputs to local"
    exit 1
fi
