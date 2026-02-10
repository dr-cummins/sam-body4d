# FBX Export Feature — Rigged & Skinned Animated Meshes

## Overview

Add the ability to export per-person animated 3D meshes as `.fbx` files with full skeleton rigging and skinning weights. This enables importing SAM-Body4D results directly into Blender, Maya, Unreal Engine, Unity, and other 3D tools.

## Current State

The pipeline currently saves per-frame, per-person:
- `.ply` mesh files (vertices + faces) via `save_mesh_results()` in `models/sam_3d_body/notebook/utils.py:214`
- `.json` camera parameters (focal length, camera translation)
- Rendered MP4 video overlay

The underlying SAM-3D-Body model outputs SMPL-format data per frame:
- `pred_vertices` — 3D mesh vertices (6890 vertices for SMPL)
- `pred_cam_t` — camera translation
- `focal_length` — estimated FOV
- `faces` — SMPL mesh face topology (shared across frames)
- SMPL body pose and shape parameters (available from the model internals)

## Requirements

### Must Have
- One `.fbx` file per person per video, containing all frames as animation keyframes
- SMPL skeleton hierarchy (24 joints) with proper bone names
- Skinning weights (SMPL linear blend skinning)
- Per-frame vertex positions or skeletal pose animation
- World-space positioning using estimated camera parameters

### Nice to Have
- SMPL-X support (hands + face, 55 joints) if the model outputs it
- Per-vertex color or UV-mapped texture
- Camera export (focal length per frame)
- Batch export toggle in Gradio UI
- GLB/glTF export as an alternative format

## Architecture

### Data Flow

```
SAM-3D-Body per-frame outputs
  → Collect pose params + vertices across all frames
  → Kalman-smoothed parameters (already in pipeline)
  → Build SMPL skeleton hierarchy
  → Assign skinning weights (from SMPL model)
  → Keyframe joint rotations per frame
  → Export as .fbx
```

### Key Components

1. **SMPL Skeleton Builder** — Construct the 24-joint hierarchy with correct bone names, rest pose, and parent-child relationships. The joint locations come from the SMPL joint regressor matrix (`J_regressor @ vertices`).

2. **Skinning Weight Mapper** — SMPL provides precomputed linear blend skinning (LBS) weights as a `(6890, 24)` matrix. Each vertex has weights for each joint. These ship with the SMPL model files.

3. **Animation Keyframer** — Convert per-frame SMPL pose parameters (axis-angle rotations for 24 joints) into per-frame bone rotations. The Kalman-smoothed parameters from the pipeline should be used.

4. **FBX Writer** — Assemble skeleton + mesh + weights + animation into a valid FBX file.

### FBX Writing Options

| Approach | Pros | Cons |
|----------|------|------|
| **Autodesk FBX SDK** (Python bindings) | Official, full-featured, reliable | Proprietary, binary distribution, license restrictions |
| **Blender bpy** (headless) | Full FBX export, well-tested | Requires Blender install (~150MB), heavy dependency |
| **urchin / fbx-writer** | Lightweight pure Python | Less mature, may have compatibility issues |
| **trimesh + pyassimp** | Already in the project | Limited skeleton/animation support in FBX path |
| **glTF/GLB via trimesh** | Open standard, lightweight | Not FBX (but widely supported) |

**Recommended**: Start with **glTF/GLB** (open, lightweight, trimesh already available) as the primary export, then add FBX via Blender bpy headless if users need it. glTF supports skeletons, skinning, and animation natively.

## SMPL Skeleton Joint Hierarchy

```
pelvis (0)
├── left_hip (1) → left_knee (4) → left_ankle (7) → left_foot (10)
├── right_hip (2) → right_knee (5) → right_ankle (8) → right_foot (11)
├── spine1 (3) → spine2 (6) → spine3 (9)
│   ├── neck (12) → head (15)
│   ├── left_collar (13) → left_shoulder (16) → left_elbow (18) → left_wrist (20)
│   └── right_collar (14) → right_shoulder (17) → right_elbow (19) → right_wrist (21)
├── left_hand (22)  [child of left_wrist]
└── right_hand (23) [child of right_wrist]
```

## Files to Modify / Create

| File | Change |
|------|--------|
| `utils/fbx_export.py` | New — skeleton builder, skinning mapper, animation keyframer, FBX/glTF writer |
| `app.py` | Add export button to Gradio UI, wire up export after 4D generation |
| `models/sam_3d_body/notebook/utils.py` | Possibly extend `save_mesh_results` or add parallel `save_animated_mesh` |
| `Dockerfile` | Add any new dependencies (e.g., pygltflib, or blender if FBX needed) |

## Data Available from Pipeline

From `app.py` `on_4d_generation()` (lines 782-1035), per frame per person:

```python
# From process_image_with_mask() → SAM-3D-Body inference
person_output = {
    "pred_vertices": np.ndarray,    # (6890, 3) mesh vertices
    "pred_cam_t": np.ndarray,       # (3,) camera translation
    "focal_length": float,          # estimated focal length
    # Also available from model internals:
    # "pred_pose": np.ndarray,      # (72,) SMPL pose params (24 joints x 3 axis-angle)
    # "pred_shape": np.ndarray,     # (10,) SMPL shape params (betas)
}

# Shared across all frames/persons:
faces = estimator.faces              # (13776, 3) SMPL face indices
```

## VRAM / Performance Considerations

- FBX/glTF assembly is CPU-only — no additional VRAM needed
- Export time should be negligible compared to inference (~seconds for a 64-frame video)
- SMPL skinning weights are a small static matrix (~600KB)

## Open Questions

1. Should the export happen automatically after 4D generation, or be a separate button?
2. Do we need to expose SMPL pose parameters (axis-angle) or just bake vertex positions per frame?
3. Is glTF/GLB acceptable as the initial format, with FBX as a follow-up?
4. Should the export include the camera (as an animated camera object in the scene)?
5. Where do we source the SMPL skinning weights — from the SAM-3D-Body checkpoint or a separate SMPL model file?
