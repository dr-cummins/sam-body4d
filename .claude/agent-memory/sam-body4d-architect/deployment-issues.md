# Deployment Issues - Detailed Analysis

## Issue 1: sys.path Import Collision — RESOLVED
**Commits**: `bc065b0`, `28b1a52`
**Root cause**: `sys.path.insert(0, 'models/diffusion_vas')` at line 8 puts diffusion_vas dir at front of sys.path. `from utils import draw_point_marker` at line 30 finds `models/diffusion_vas/utils.py` (which only has `set_seed`) instead of `utils/__init__.py`.

**Complication**: Simple `sys.path.append()` breaks diffusion_vas internal imports. `demo.py` does `from models.diffusion_vas.pipeline_diffusion_vas import DiffusionVASPipeline` which resolves through a NESTED `models/diffusion_vas/models/diffusion_vas/pipeline_diffusion_vas.py` directory. This requires diffusion_vas to be at the FRONT of sys.path.

**Fix applied**: Import project utils BEFORE adding diffusion_vas to sys.path:
1. Line 7: `sys.path.append(...)` for sam_3d_body (safe, needs project utils)
2. Line 29: `from utils import ...` — caches project utils in `sys.modules`
3. Line 37: `sys.path.insert(0, ...)` for diffusion_vas (after utils is cached)

**Verification**: Tested locally with `importlib.util.find_spec` — confirmed utils resolves correctly and pipeline_diffusion_vas finds the nested file.

## Issue 2: Missing BPE Vocabulary File — RESOLVED
**Commit**: `bc065b0`
**Fix**: Copied `bpe_simple_vocab_16e6.txt.gz` (1.3MB) to hf-space. Updated `sync_to_hf.sh` to include it via `--include='assets/bpe_simple_vocab_16e6.txt.gz'`.

## Issue 3: Missing Example Videos — RESOLVED
**Commit**: `bc065b0`
**Fix**: Gallery dynamically filters missing thumbnails. `AVAILABLE_EXAMPLES` list built at module load, gallery hidden when empty. Select handler uses index into filtered list.

## Issue 4: Gradio Launch Configuration — RESOLVED
**Commit**: `bc065b0`
**Fix**: `demo.launch(server_name="0.0.0.0", server_port=7860)`

## Issue 5: VRAM Budget — MITIGATED
**Commit**: `bc065b0`
**Fix**: Set `completion.enable: false` in hf-space body4d.yaml. Reduces VRAM from ~28-40 GB to ~15-20 GB. Core pipeline (SAM-3 + SAM-3D-Body + MoGe) loads without diffusion-vas models. Re-enable after confirming stability.

## Issue 6: PYOPENGL_PLATFORM — NOT AN ISSUE
Dockerfile sets osmesa, renderer.py respects env var. Confirmed working.

## Issue 7: NumPy/SciPy Version Mismatch — NOT A BLOCKER
Warning message only. sam3 pins numpy==1.26, scipy wants >=1.26.4.

## Issue 8: startup_duration_timeout — RESOLVED
**Commit**: `bc065b0`
**Fix**: Added `startup_duration_timeout: 1h` to hf-space README YAML frontmatter.
