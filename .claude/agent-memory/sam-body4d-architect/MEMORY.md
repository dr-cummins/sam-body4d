# SAM-Body4D Architect Agent Memory

See [deployment-issues.md](deployment-issues.md) for detailed findings from codebase analysis.

## Key Facts
- Main repo: `/Users/thomascummins/dev/chromatica/projects/sam-body4d`
- HF Space repo: `/Users/thomascummins/dev/chromatica/projects/hf-space`
- Sync script excludes binaries; Dockerfile/start.py/body4d.yaml are manually managed in hf-space
- `demo.launch()` needs `server_name="0.0.0.0"` for Docker/HF Spaces
- sam3 BPE vocab (`bpe_simple_vocab_16e6.txt.gz`) is a binary file missing from hf-space
- Example videos (`.mp4`) are binary and missing from hf-space
- `NVIDIA_VISIBLE_DEVICES=void` in Dev Mode is normal; CUDA still works
- NumPy pinned to 1.26 by sam3; scipy warns about >=1.26.4 but works
- All models load at startup via `init_runtime()` -- ~28-40 GB VRAM total
- Diffusion-VAS uses `enable_model_cpu_offload()` to reduce peak VRAM
- `completion.enable: true` in body4d.yaml triggers loading Diffusion-VAS models

## Import Chain (Critical)
- `sys.path.insert(0, 'models/diffusion_vas')` makes `utils` resolve to `diffusion_vas/utils.py`
- This shadows the project's `utils/__init__.py` package -- root cause of current crash
- Fix: rename import or restructure sys.path order
- sam_3d_body files import `from utils.painter import color_list` -- also depends on project utils
