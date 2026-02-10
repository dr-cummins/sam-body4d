---
title: SAM-Body4D
emoji: 🏂
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

<!-- <h1 align="center">🏂 SAM-Body4D</h1> -->

# 🏂 SAM-Body4D

[**Mingqi Gao**](https://mingqigao.com), [**Yunqi Miao**](https://yoqim.github.io/), [**Jungong Han**](https://jungonghan.github.io/)

**SAM-Body4D** is a **training-free** method for **temporally consistent** and **robust** 4D human mesh recovery from videos.
By leveraging **pixel-level human continuity** from promptable video segmentation **together with occlusion recovery**, it reliably preserves identity and full-body geometry in challenging in-the-wild scenes.

[ 📄 [`Paper`](https://arxiv.org/pdf/2512.08406)] [ 🌐 [`Project Page`](https://mingqigao.com/projects/sam-body4d/index.html)] [ 📝 [`BibTeX`](#-citation)]


### ✨ Key Features

- **Temporally consistent human meshes across the entire video**
<div align=center>
<img src="./assets/demo1.gif" width="99%"/>
</div>

- **Robust multi-human recovery under heavy occlusions**
<div align=center>
<img src="./assets/demo2.gif" width="99%"/>
</div>

- **Robust 4D reconstruction under camera motion**
<div align=center>
<img src="./assets/demo3.gif" width="99%"/>
</div>

<!-- Training-Free 4D Human Mesh Recovery from Videos, based on [SAM-3](https://github.com/facebookresearch/sam3), [Diffusion-VAS](https://github.com/Kaihua-Chen/diffusion-vas), and [SAM-3D-Body](https://github.com/facebookresearch/sam-3d-body). -->

## 🕹️ Gradio Demo

https://github.com/user-attachments/assets/07e49405-e471-40a0-b491-593d97a95465


## 📊 Resource & Profiling Summary

For detailed GPU/CPU resource usage, peak memory statistics, and runtime profiling, please refer to:

👉 **[resources.md](assets/doc/resources.md)**  


## 🖥️ Installation

#### 1. Create and Activate Environment
```
conda create -n body4d python=3.12 -y
conda activate body4d
```
#### 2. Install PyTorch (choose the version that matches your CUDA), Detectron, and SAM3
```
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps
pip install -e models/sam3
```
If you are using a different CUDA version, please select the matching PyTorch build from the official download page:
https://pytorch.org/get-started/previous-versions/

#### 3. Install Dependencies
```
pip install -e .
```


## 🚀 Run the Demo

#### 1. Setup checkpoints & config (recommended)

We provide an automated setup script that:
- generates `configs/body4d.yaml` from a release template,
- downloads all required checkpoints (existing files will be skipped).

Some checkpoints (**[SAM 3](https://huggingface.co/facebook/sam3)** and **[SAM 3D Body](https://huggingface.co/facebook/sam-3d-body-dinov3)**) require prior access approval on Hugging Face.
Before running the setup script, please make sure you have **accepted access**
on their Hugging Face pages.

If you plan to use these checkpoints, login once:
```bash
huggingface-cli login
```
Then run the setup script:
```bash
python scripts/setup.py --ckpt-root /path/to/checkpoints
```
#### 2. Run
```bash
python app.py
```
#### Manual checkpoint setup (optional)

If you prefer to download checkpoints manually ([SAM 3](https://huggingface.co/facebook/sam3), [SAM 3D Body](https://huggingface.co/facebook/sam-3d-body-dinov3), [MoGe-2](https://huggingface.co/Ruicheng/moge-2-vitl-normal), [Diffusion-VAS](https://github.com/Kaihua-Chen/diffusion-vas?tab=readme-ov-file#download-checkpoints), [Depth-Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true)), please place them under the directory with the following structure:
```
${CKPT_ROOT}/
├── sam3/                                
│   └── sam3.pt
├── sam-3d-body-dinov3/
│   ├── model.ckpt
│   └── assets/
│       └── mhr_model.pt
├── moge-2-vitl-normal/
│   └── model.pt
├── diffusion-vas-amodal-segmentation/
│   └── (directory contents)
├── diffusion-vas-content-completion/
│   └── (directory contents)
└── depth_anything_v2_vitl.pth
```
After placing the files correctly, you can run the setup script again.
Existing files will be detected and skipped automatically.

## 🤖 Auto Run
Run the full end-to-end video pipeline with a single command:
```bash
python scripts/offline_app.py --input_video <path>
```
where the input can be a directory of frames or an .mp4 file. The pipeline automatically detects humans in the initial frame, treats all detected humans as targets, and performs temporally consistent 4D reconstruction over the video.

## 🧭 Getting Started (Cloud)

This section walks through the full workflow for processing a video on the HuggingFace Space, from connecting via SSH to downloading results locally.

### Prerequisites

Before you begin, ensure the following are in place:

1. The HuggingFace Space is **running** (not paused/sleeping) — check the [Space page](https://huggingface.co/spaces/troutmoose/sam-body4d).
2. **Dev Mode** is enabled in Space Settings.
3. Your SSH key is added to [hf.co/settings/keys](https://huggingface.co/settings/keys).
4. Model checkpoints have been downloaded to the server (see [Checkpoint Management](#checkpoint-management) below).

### Step 1: Connect to the Space via SSH

From your local terminal, SSH into the running Space container:

```bash
ssh hf-sam3
```

This uses the SSH config alias. If you haven't set it up, the full command is:

```bash
ssh troutmoose-sam-body4d@ssh.hf.space
```

Once connected you'll land in the `/app` directory, which contains the project code.

### Step 2: Validate the Environment

Run the following checks inside the SSH session to confirm the GPU and dependencies are working:

```bash
# Check GPU is visible and has enough VRAM
nvidia-smi

# Verify PyTorch can see CUDA
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB')"

# Verify FFmpeg is installed
ffmpeg -version | head -1

# Check that checkpoints are present
ls -lh /data/checkpoints/
```

Expected output: CUDA available, GPU name (e.g. NVIDIA L40S), ~48 GB VRAM, FFmpeg version, and checkpoint files listed.

### Step 3: Start the Gradio App

If the app is not already running (it auto-starts on deploy, but not always in Dev Mode):

```bash
cd /app
python3 app.py
```

The Gradio UI will be accessible at the Space's public URL:
`https://troutmoose-sam-body4d.hf.space`

You can also access it from the "App" tab on the Space page.

### Step 4: Upload a Video (GUI)

1. Open the Gradio UI in your browser.
2. In the left column, click **"Upload Video (click to open)"** to expand the upload panel.
3. Click the file upload area and select an `.mp4` video from your local machine.
   - Alternatively, click one of the three **Example Videos** thumbnails at the top to load a pre-loaded example.
4. Once loaded, the **first frame** of the video appears in the "Current Frame" viewer. The **frame slider** and **time display** update to reflect the video duration.

### Step 5: Annotate Targets (GUI)

You need to tell the model which humans in the video to track by clicking on them in the first frame.

1. **Select Point Type**: Choose **Positive** (mark a human) or **Negative** (mark background) from the radio buttons below the frame.
2. **Click on a person** in the Current Frame viewer. A marker appears on the image, and SAM-3 immediately computes a segmentation mask overlay for that person.
   - Add more positive clicks to refine the mask if needed.
   - Add negative clicks to exclude nearby regions that were incorrectly included.
3. Click **"Add Target"** to finalize this person as a tracking target. The target appears in the **Targets** checklist.
4. **Repeat** steps 1-3 for each additional person you want to track. Each target gets a unique color overlay.
5. Use the **frame slider** to scrub through the video and verify your selections look reasonable on different frames. Note: annotations are placed on the frame shown when you click.

### Step 6: Run Mask Generation (GUI)

1. Click the **"Mask Generation"** button in the right column.
2. SAM-3 propagates the annotated masks across all frames of the video. This runs on the GPU and may take 1-5 minutes depending on video length.
3. When complete, a **segmentation result video** appears in the "Segmentation Result" player on the right. This shows each tracked person highlighted with a colored mask overlay across all frames.

### Step 7: Run 4D Generation (GUI)

1. Click the **"4D Generation"** button.
2. This runs the full pipeline:
   - **Occlusion detection** via Diffusion-VAS amodal segmentation (if enabled in config)
   - **Content completion** for occluded frames
   - **FOV estimation** via MoGe-2
   - **4D human mesh recovery** via SAM-3D-Body for each batch of frames
   - **Mesh rendering** overlaid on the original video
3. Processing time depends on video length, number of humans, and whether occlusion recovery is enabled. See the [Resource & Profiling Summary](assets/doc/resources.md) for estimates.
4. When complete, a **4D result video** appears in the "4D Result" player, showing the recovered 3D human meshes rendered over the original footage.

### Step 8: Retrieve Results

Results are written to the server's local filesystem under `./outputs/<run_id>/`. Each run gets a unique timestamped directory.

#### Output directory structure (on the server)

```
outputs/<YYYYMMDD_HHMMSS_mmm_xxxxxxxx>/
├── images/                          # Extracted video frames (JPG)
├── masks/                           # Per-frame segmentation masks (PNG, DAVIS palette)
├── masks_vis/                       # Mask visualizations with background dimmed
├── completion/                      # Occlusion recovery intermediate results
│   └── <id>/
│       ├── images/                  # Completed RGB frames
│       └── masks/                   # Completed amodal masks
├── rendered_frames/                 # 4D mesh rendered over original frames (all targets combined)
├── rendered_frames_individual/      # Per-target rendered frames
│   ├── 1/                           # Target 1
│   ├── 2/                           # Target 2
│   └── ...
├── mesh_4d_individual/              # Per-target 3D mesh files (.obj)
│   ├── 1/
│   ├── 2/
│   └── ...
├── focal_4d_individual/             # Per-target focal length / camera parameters
│   ├── 1/
│   ├── 2/
│   └── ...
├── video_mask_<timestamp>.mp4       # Segmentation result video
└── 4d_<timestamp>.mp4               # 4D reconstruction result video
```

#### Download results to your local machine

Use rsync over SSH to pull results down:

```bash
# From your local terminal (not inside the Space)
./scripts/cloud/sync_data.sh down
```

Or manually with rsync:

```bash
rsync -avzP hf-sam3:/app/outputs/ ./data/outputs/
```

Results will appear in `./data/outputs/` on your local machine.

### Step 9: Pause the Space

When you're done processing, **pause the Space** to stop billing:

1. Go to the [Space Settings](https://huggingface.co/spaces/troutmoose/sam-body4d/settings).
2. Click **Pause Space**.

Billing stops immediately. The container and all ephemeral data (anything not committed to git) will be lost when the Space restarts.

### Checkpoint Management

Model checkpoints (~20-30 GB total) are stored separately from the code and must be present on the server before running inference.

#### Where checkpoints live

| Location | Path | Contents |
|----------|------|----------|
| **Server (HF Space)** | `/data/checkpoints/` | Active checkpoints loaded by the app at runtime |
| **GCS Bucket** | `gs://<bucket>/sam-body4d/checkpoints/` | Persistent backup — survives Space restarts |
| **Local machine** | `./checkpoints/` | Local copy (if running locally) |

#### Downloading checkpoints to the server

SSH into the Space and run the setup script:

```bash
ssh hf-sam3
cd /app
python3 scripts/setup.py --ckpt-root /data/checkpoints
```

This downloads all required checkpoints from HuggingFace Hub. Gated models (SAM-3, SAM-3D-Body) require prior access approval — see [Installation](#-installation) for details.

#### Syncing checkpoints from GCS (alternative)

If checkpoints have been uploaded to a GCS bucket:

```bash
# On the server via SSH
gcloud auth login
gsutil -m cp -r gs://<bucket>/sam-body4d/checkpoints/* /data/checkpoints/
```

### Where Everything Lives (Summary)

| Asset | Location | Persistence |
|-------|----------|-------------|
| **Source code** | Local: `./` (this repo) | Permanent (git) |
| **Source code (deployed)** | Server: `/app/` | Rebuilt on each `git push hf` |
| **Model checkpoints** | Server: `/data/checkpoints/` | Ephemeral (lost on restart unless on persistent storage) |
| **Model checkpoints (backup)** | GCS: `gs://<bucket>/sam-body4d/checkpoints/` | Permanent |
| **Input videos** | Local: uploaded via Gradio UI in browser | Copied to server temp dir during upload |
| **Example videos** | Server: `/app/assets/examples/` | Deployed with code |
| **Output frames & meshes** | Server: `/app/outputs/<run_id>/` | Ephemeral (download before pausing) |
| **Output videos (masks, 4D)** | Server: `/app/outputs/<run_id>/*.mp4` | Ephemeral (download before pausing) |
| **Downloaded results** | Local: `./data/outputs/` | Permanent |
| **Gradio temp files** | Server: `/app/gradio_tmp/` | Ephemeral |
| **Config** | Server: `/app/configs/body4d.yaml` | Deployed with code (edit `paths.ckpt_root`) |

---

## 🌐 HuggingFace Spaces Deployment

This project is deployed as a private [HuggingFace Docker Space](https://huggingface.co/spaces/troutmoose/sam-body4d) for GPU-accelerated inference.

### Dual-Remote Git Workflow

The codebase is tracked in **two git remotes** — GitHub for development and HuggingFace for deployment:

| Remote | Repository | Purpose |
|--------|-----------|---------|
| `origin` | [GitHub](https://github.com/dr-cummins/sam-body4d) | Source of truth. All development, branches, and pull requests. |
| `hf` | [HuggingFace](https://huggingface.co/spaces/troutmoose/sam-body4d) | Deployment target. Each push triggers a Docker image rebuild. |

### CI/CD Workflow

HuggingFace Spaces uses a **git-push-to-deploy** model. There is no separate CI pipeline — the deployment *is* the push:

```
Local Development          GitHub (origin)         HuggingFace (hf)
┌──────────────┐          ┌──────────────┐         ┌───────────────────┐
│  Edit code   │──push──→ │  Store code  │         │                   │
│  Run tests   │          │  PR reviews  │         │                   │
│  Commit      │          │  History     │         │                   │
│              │──push──→ │              │         │                   │
│              │─────────────push──────────────→   │ Docker rebuild    │
│              │          │              │         │ New VM provisioned│
│              │          │              │         │ App starts        │
└──────────────┘          └──────────────┘         └───────────────────┘
```

**Step-by-step:**

1. **Develop locally** — edit code, test, commit.
2. **Push to GitHub** — `git push origin master` for version control.
3. **Deploy to HF** — `git push hf master` triggers a full Docker rebuild on HuggingFace infrastructure. The Space rebuilds the image, provisions a new GPU VM, and starts the container.
4. **Connect via Dev Mode** — SSH into the running container with VS Code for live debugging, or use the Gradio UI in a browser.
5. **Pause when done** — stop the Space from Settings to halt billing.

### Quick reference

```bash
# Push to GitHub only (no deploy)
git push origin master

# Deploy to HuggingFace Space
git push hf master

# Push to both
git push origin master && git push hf master
```

### Dev Mode (fast iteration)

With [Dev Mode](https://huggingface.co/docs/hub/en/spaces-dev-mode) enabled, you can edit code directly on the running container via VS Code SSH **without triggering a Docker rebuild**. Changes take effect when you click **Refresh** in the Dev Mode modal. To persist changes, commit and push from inside the container.

For full deployment details, see [HUGGINGFACE_MIGRATION_GUIDE.md](HUGGINGFACE_MIGRATION_GUIDE.md).

## 🛠️ Claude Code Configuration

This project uses [Claude Code](https://claude.ai/claude-code) for AI-assisted development workflows. Configuration lives in `.claude/` and is checked into version control.

### File Structure

```
.claude/
├── settings.local.json              # Permission rules (auto-approved CLI commands)
└── skills/
    ├── hf-status/SKILL.md           # /hf-status skill definition
    └── hf-deploy/SKILL.md           # /hf-deploy skill definition
```

### Skills

Skills are slash-command workflows that Claude Code executes with pre-approved tool permissions.

| Skill | Command | Description |
|-------|---------|-------------|
| **hf-status** | `/hf-status` | Queries the HF Space API for runtime stage, hardware, deployed SHA, and streams run/build logs. Presents a formatted status summary. |
| **hf-deploy** | `/hf-deploy` | Runs pre-flight checks on the `hf-space/` repo, pushes to HuggingFace, then polls the Space API until the build completes or fails. Reports deployment result with logs. |

Each skill declares its own `allowed-tools` in frontmatter so all commands it needs run without manual approval prompts.

### Allowed CLI Commands

The following commands are auto-approved in `settings.local.json` and do not require manual confirmation:

**Git operations:**
```
git status, git log, git diff, git branch, git add, git commit, git push,
git fetch, git clone, git checkout, git reset, git rm, git config,
git lfs, git lfs ls-files, git check-attr, git format-patch, git fsck,
git remote add, git -C <path> <cmd>, git xet, git-xet, xargs git rm
```

**Python / package management:**
```
uv run python, uv add, uv pip install, python3, pip3,
.venv/bin/pip install, .venv/bin/python -m pip install
```

**HuggingFace CLI:**
```
huggingface-cli whoami, hf auth whoami, hf auth login,
hf repo create, hf repo delete, .venv/bin/huggingface-cli whoami
```

**Networking / system:**
```
curl, ssh, bash, sleep, brew install
```

**Web fetch (domain-restricted):**
```
WebFetch: github.com, blog.korny.info
```

### Global Rules

Global rules (in `~/.claude/rules/`) enforce consistent behavior across sessions:

| Rule file | Purpose |
|-----------|---------|
| `python/uv-environment.md` | Always use `uv` for local Python environment management |
| `huggingface/authentication.md` | SSH config for `git@hf.co` (git) and `ssh.hf.space` (Dev Mode) |
| `huggingface/spaces-overview.md` | Hardware tiers, environment variables, deployment lifecycle |
| `huggingface/spaces-dev-mode.md` | Dev Mode usage: SSH access, persisting changes, limitations |

### Dual-Repo Deployment Architecture

Claude Code manages two separate git repositories to work around HuggingFace's binary file restrictions:

| Repo | Path | Remote | Purpose |
|------|------|--------|---------|
| **Main repo** | `sam-body4d/` | `origin` (GitHub) | Full source with LFS binaries (assets, models) |
| **HF Space repo** | `hf-space/` | `origin` (HF) | Text/source files only — no binary blobs |

The `/hf-deploy` skill operates on the `hf-space/` repo using `git -C` to avoid `cd` chains that would break permission matching.

## 📝 Citation
If you find this repository useful, please consider giving a star ⭐ and citation.
```
@article{gao2025sambody4d,
  title   = {SAM-Body4D: Training-Free 4D Human Body Mesh Recovery from Videos},
  author  = {Gao, Mingqi and Miao, Yunqi and Han, Jungong},
  journal = {arXiv preprint arXiv:2512.08406},
  year    = {2025},
  url     = {https://arxiv.org/abs/2512.08406}
}
```

## 👏 Acknowledgements

The project is built upon [SAM-3](https://github.com/facebookresearch/sam3), [Diffusion-VAS](https://github.com/Kaihua-Chen/diffusion-vas) and [SAM-3D-Body](https://github.com/facebookresearch/sam-3d-body). We sincerely thank the original authors for their outstanding work and contributions. 
