# SAM-Body4D Cloud Deployment Guide (HuggingFace Spaces)

**Objective**: Run SAM-Body4D inference on a cloud GPU via HuggingFace Spaces while developing locally with VS Code.

## 1. Architecture Overview

```
Local (macOS)                     HuggingFace Space (L40S / A100)
┌──────────────────┐              ┌──────────────────────────┐
│  VS Code          │──SSH/Dev──→ │  Docker Container         │
│  (Remote-SSH)     │  Mode       │                           │
│                   │             │  Gradio UI (:7860)        │
│  gcloud CLI       │             │  + headless pipeline      │
│                   │             │                           │
│  sync_data.sh     │──rsync───→  │  /app (code)              │
│  (up/down)        │  over SSH   │  /tmp/checkpoints         │
└────────┬─────────┘              └───────────┬──────────────┘
         │                                     │
         │          ┌─────────────┐            │
         └─────────→│ GCS Bucket  │←───────────┘
                    │ checkpoints │
                    └─────────────┘
```

- **Local (macOS)**: Source of truth for code (GitHub). Review results here.
- **Remote (HF Space)**: GPU compute engine. Runs inference via Gradio UI or headless pipeline.
- **GCS Bucket**: Persistent storage for model checkpoints (~20-30GB).
- **Connection**: Dev Mode SSH for VS Code remote development.

## 2. Prerequisites

1. **HuggingFace Pro Account** ($9/mo) — required for Dev Mode (SSH access).
2. **SSH Key** added to [hf.co/settings/keys](https://huggingface.co/settings/keys).
3. **VS Code** with the "Remote - SSH" extension installed.
4. **HuggingFace CLI** (`hf`) installed locally:
   ```bash
   curl -LsSf https://hf.co/cli/install.sh | bash
   hf auth login
   ```
5. **gcloud CLI** configured for GCS checkpoint storage.
6. **Gated model access** approved on HuggingFace for:
   - [SAM 3](https://huggingface.co/facebook/sam3)
   - [SAM 3D Body](https://huggingface.co/facebook/sam-3d-body-dinov3)

## 3. Dual-Remote Git Setup

This project uses **two git remotes**:

| Remote | URL | Purpose |
|--------|-----|---------|
| `origin` | `git@github.com:dr-cummins/sam-body4d.git` | Source of truth. All development branches and history. |
| `hf` | `git@hf.co:spaces/troutmoose/sam-body4d` | Deployment target. Pushing here triggers a Docker rebuild on HF Spaces. |

```bash
# Check remotes
git remote -v

# Normal development: push to GitHub
git push origin master

# Deploy to HF Space: push to HuggingFace (triggers Docker rebuild)
git push hf master
```

See the [README](README.md#-huggingface-spaces-deployment) for full details on the CI/CD workflow.

## 4. Space Configuration

| Setting | Value |
|---------|-------|
| **Space** | [troutmoose/sam-body4d](https://huggingface.co/spaces/troutmoose/sam-body4d) |
| **SDK** | Docker (for CUDA + FFmpeg support) |
| **Hardware** | NVIDIA L40S (48GB VRAM, $1.80/hr) — upgradable to A100 (80GB, $2.50/hr) |
| **Visibility** | Private |
| **Dev Mode** | Enabled |

## 5. Cloud Infrastructure Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Container definition: CUDA 12.1, Python 3.11, FFmpeg, PyTorch. Dev Mode compatible. |
| `.dockerignore` | Excludes .venv, .git, secrets, checkpoints, outputs from Docker build. |
| `scripts/cloud/sync_data.sh` | Bidirectional rsync over SSH: `up` (send videos) / `down` (get results). |
| `scripts/cloud/remote_inference.py` | Headless batch inference script that runs on the GPU. |
| `scripts/cloud/gcs_sync.py` | Download checkpoints from GCS bucket to container. |
| `scripts/cloud/diagnose.py` | Environment diagnostic: CUDA, PyTorch, FFmpeg, GPU memory checks. |

## 6. Development Workflow

### First-time setup

```bash
# 1. Create the Space (already done)
hf repo create troutmoose/sam-body4d --repo-type space --space-sdk docker --private

# 2. Add the HF remote
git remote add hf git@hf.co:spaces/troutmoose/sam-body4d

# 3. Push code to trigger first build
git push hf master

# 4. Enable Dev Mode from Space Settings on HF website
# 5. Select L40S hardware from Space Settings
```

### Daily workflow

```bash
# 1. Develop locally, commit to GitHub
git add -A && git commit -m "description of change"
git push origin master

# 2. Deploy to HF Space (triggers Docker rebuild)
git push hf master

# 3. Connect via VS Code Remote-SSH once Space is running
#    (SSH details shown in Dev Mode modal on HF website)

# 4. On the remote Space (via SSH):
python scripts/cloud/remote_inference.py    # headless batch processing
# OR access Gradio UI via Space URL in browser

# 5. Sync results back locally
./scripts/cloud/sync_data.sh down

# 6. Pause Space when done to stop billing
```

### Iterative development (Dev Mode)

When Dev Mode is enabled, you can edit code on the Space directly via VS Code SSH without triggering a full Docker rebuild:

1. Connect to Space via VS Code Remote-SSH
2. Edit files in `/app`
3. Click **Refresh** in Dev Mode modal to restart the app
4. Changes are **ephemeral** — commit and push to persist:
   ```bash
   # Inside the Space container
   git add . && git commit -m "changes from dev mode"
   git push   # pushes to HF remote, persists changes
   ```

## 7. Checkpoint Management

Model checkpoints are stored in a GCS bucket and downloaded to the container at startup. This avoids baking ~20-30GB of weights into the Docker image.

```bash
# Upload checkpoints to GCS (one-time, from local machine)
gsutil -m cp -r /path/to/checkpoints/* gs://BUCKET_NAME/sam-body4d/checkpoints/

# Download checkpoints on the Space (run via SSH)
python scripts/cloud/gcs_sync.py
```

### Checkpoint manifest

| Model | Files | Size (approx) |
|-------|-------|---------------|
| SAM-3 | `sam3/sam3.pt` | ~2.5 GB |
| SAM-3D-Body | `sam-3d-body-dinov3/model.ckpt`, `mhr_model.pt`, `model_config.yaml` | ~3 GB |
| MoGe-2 | `moge-2-vitl-normal/model.pt` | ~1.2 GB |
| Diffusion-VAS (amodal) | `diffusion-vas-amodal-segmentation/` | ~5 GB |
| Diffusion-VAS (completion) | `diffusion-vas-content-completion/` | ~5 GB |
| Depth-Anything-V2 | `depth_anything_v2_vitl.pth` | ~1.3 GB |

## 8. GPU Memory & Hardware

Profiling results from H800 (80GB VRAM):

| Scenario | #Humans | Occlusion | Batch Size | 4D Peak VRAM | Runtime |
|----------|---------|-----------|------------|--------------|---------|
| Simple | 1 | No | 64 | 14.49 GB | 1m 10s |
| Multi-human | 5 | No | 64 | 40.87 GB | 2m 55s |
| Multi-human | 5 | Yes | 32 | 35.19 GB | 27m 15s |
| Multi-human | 5 | Yes | 64 | 53.28 GB | 26m 6s |

**L40S (48GB)**: Handles 1-2 humans without occlusion easily. Multi-human (batch_size=32, no occlusion) should fit. Upgrade to A100 for full multi-human + occlusion support.

## 9. Debugging

Because you connect via SSH, debugging works identically to local development:

1. **Live logs**: Run scripts in VS Code terminal on the remote Space.
2. **Log file**: `remote_inference.py` writes to `/data/outputs/inference.log`.
3. **Breakpoints**: Open files in remote VS Code, set breakpoints, press F5.
4. **GPU monitoring**: Run `nvidia-smi` or `htop` in a second terminal.

## 10. Cost Management

| Resource | Cost |
|----------|------|
| HF Pro subscription | $9/month |
| L40S compute | $1.80/hour (only while running) |
| A100 compute | $2.50/hour (upgrade path) |
| GCS storage | ~$0.60/month (~30GB) |

**To minimize costs**: Pause the Space from Settings when not in use. Billing stops immediately on pause.
