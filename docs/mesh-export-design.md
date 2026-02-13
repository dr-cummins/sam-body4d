# 3D Mesh Export — End-to-End Design Specification

## 1. Overview

SAM-Body4D recovers temporally consistent 3D human body meshes from monocular video. This document specifies the complete data flow from model inference through animation export, import into DCC tools (Blender, Unreal Engine 5), data cleanup, and final rendering.

**Primary deliverable:** Per-person animated `.glb` files with full MHR skeleton (127 joints), skinning weights, and per-frame joint rotation animation.

**Secondary deliverable:** Baked vertex animation `.glb` (morph targets, no skeleton) as a simpler fallback.

---

## 2. Model Output — What MHR Produces

The MHR (Meta Human Rig) model is a TorchScript body model with 127 skeleton joints and 18,439 mesh vertices. It runs inside `mhr_head.mhr_forward()` and returns:

### Per-frame, per-person output dict

| Field | Shape | Description |
|-------|-------|-------------|
| `pred_vertices` | (18439, 3) | Mesh vertex positions (camera coords) |
| `pred_joint_coords` | (127, 3) | Joint 3D positions (camera coords) |
| `pred_global_rots` | (127, 3, 3) | Global rotation matrix per joint |
| `pred_cam_t` | (3,) | Camera translation vector |
| `focal_length` | scalar | Estimated focal length |
| `global_rot` | (3,) | Global rotation (ZYX Euler angles) |
| `body_pose_params` | (133,) | Body pose Euler angles (mixed 1-DOF and 3-DOF joints) |
| `hand_pose_params` | (108,) | Hand pose in PCA space (54 per hand) |
| `scale_params` | (28,) | Bone scale PCA coefficients |
| `shape_params` | (45,) | Body shape PCA coefficients |
| `expr_params` | (72,) | Face expression params (currently zeroed) |
| `mhr_model_params` | (204,) | Full model parameter vector: [trans×10, global_rot, body_pose, scales] |
| `faces` | (36874, 3) | Triangle indices (shared across frames) |

### Static model data (from `mhr_head` buffers)

| Buffer | Shape | Description |
|--------|-------|-------------|
| `joint_rotation` | (127, 3, 3) | Rest-pose rotation matrix per joint |
| `scale_mean` | (68,) | Mean bone scales |
| `scale_comps` | (28, 68) | Scale PCA components |
| `faces` | (36874, 3) | Mesh face topology |
| `keypoint_mapping` | (308, 18566) | Maps verts+joints to Sapiens 308 keypoints |
| `hand_joint_idxs_left/right` | (27,) | Joint indices for hand params |

### Coordinate system

MHR's native space is **Y-up**. The pipeline applies a coordinate conversion to camera space:

```python
# sam3d_body.py line 2239
verts[..., [1, 2]] *= -1   # Y-up → camera coords (negate Y and Z)
jcoords[..., [1, 2]] *= -1
```

glTF's convention is Y-up, matching MHR native. **Export must undo this flip.**

---

## 3. Pipeline Data Flow — Video to Exported Mesh

```
                                    ┌─────────────────────────────────────────┐
                                    │            PIPELINE STAGES              │
                                    └─────────────────────────────────────────┘

    ┌────────────┐     ┌───────────────┐     ┌──────────────────┐     ┌──────────────┐
    │   Input    │     │   SAM-3       │     │  Diffusion-VAS   │     │   MoGe-2     │
    │   Video    │────▶│   Mask        │────▶│  Occlusion       │────▶│   FOV        │
    │            │     │   Propagation │     │  Recovery        │     │   Estimation │
    └────────────┘     └───────────────┘     └──────────────────┘     └──────┬───────┘
                                                                             │
                              cam_int (camera intrinsics)  ◀─────────────────┘
                                             │
                                             ▼
                       ┌─────────────────────────────────────────────┐
                       │          SAM-3D-Body Estimation             │
                       │   (process_frames → run_inference_batch)    │
                       │                                             │
                       │   Per-frame per-person:                     │
                       │   • DINOv3 backbone → pose token            │
                       │   • MHR Head → body/hand/face params        │
                       │   • MHR Forward → vertices + joints         │
                       │     + global rotations                      │
                       └─────────────────┬───────────────────────────┘
                                         │
                                         │  Raw per-frame estimates
                                         ▼
                       ┌─────────────────────────────────────────────┐
                       │         Temporal Smoothing                  │
                       │   (inside run_inference_batch)              │
                       │                                             │
                       │  1. Body pose & hand params                 │
                       │     kalman_smooth_mhr_params_per_obj_id_    │
                       │     adaptive()                              │
                       │     - Static segments: EMA (α=0.18)         │
                       │     - Dynamic segments: spike suppression   │
                       │     - Occluded frames: linear interpolation │
                       │       between nearest visible frames        │
                       │                                             │
                       │  2. Shape & scale: fixed to frame 0 values  │
                       │                                             │
                       │  3. Global rotation                         │
                       │     ema_smooth_global_rot_per_obj_id_       │
                       │     adaptive()                              │
                       │     - Same strategy as body pose            │
                       └─────────────────┬───────────────────────────┘
                                         │
                                         │  Smoothed parameters
                                         ▼
                       ┌─────────────────────────────────────────────┐
                       │      MHR Forward (Re-run)                   │
                       │                                             │
                       │   Smoothed params → final vertices, joints, │
                       │   global rotations, model params            │
                       │                                             │
                       │   Then: verts[..., [1,2]] *= -1             │
                       │   (convert to camera coords for rendering)  │
                       └─────────────────┬───────────────────────────┘
                                         │
                              ┌──────────┴──────────┐
                              │                     │
                              ▼                     ▼
                 ┌──────────────────┐    ┌─────────────────────┐
                 │  Rendering       │    │  Mesh Export         │
                 │  Pipeline        │    │  (NEW)               │
                 │                  │    │                      │
                 │  • .ply per frame│    │  • Undo Y/Z flip     │
                 │  • .json camera  │    │  • Build GLB         │
                 │  • overlay MP4   │    │  • Skeleton + anim   │
                 └──────────────────┘    │  • Morph targets     │
                                         │  • ZIP for download  │
                                         └─────────────────────┘
```

### 3.1 Data collection hook

The export data collection sits in `on_4d_generation()` after `save_mesh_results()` (app.py line ~1030). For each frame and each person, we capture:

```python
export_data[person_id].vertices.append(person_output["pred_vertices"])      # (18439, 3)
export_data[person_id].joint_coords.append(person_output["pred_joint_coords"])  # (127, 3)
export_data[person_id].global_rots.append(person_output["pred_global_rots"])    # (127, 3, 3)
export_data[person_id].cam_t.append(person_output["pred_cam_t"])            # (3,)
export_data[person_id].focal_length.append(person_output["focal_length"])   # scalar
```

This data has already been through the full smoothing pipeline. No additional smoothing is needed at this stage.

### 3.2 Data volumes

| Video length | Frames @30fps | Vertex data per person | Joint data per person | Total (1 person) |
|-------------|---------------|----------------------|---------------------|-----------------|
| 2 seconds | 60 | 13 MB | 0.9 MB | ~14 MB |
| 5 seconds | 150 | 33 MB | 2.3 MB | ~35 MB |
| 10 seconds | 300 | 66 MB | 4.5 MB | ~70 MB |

GLB files will be roughly these sizes. Compression in ZIP reduces by ~40%.

---

## 4. Export Format — glTF/GLB Structure

### 4.1 Why glTF/GLB

- **Open standard** (Khronos Group, royalty-free)
- **Native support** in Blender (import/export), UE5 (Interchange Framework), Unity, Three.js
- **Binary GLB** is a single self-contained file (no external texture/buffer references)
- **Pure-Python writer** available (`pygltflib`, ~50KB, no native deps)
- Supports skeleton, skinning, morph targets, and animation natively
- Y-up convention matches MHR's native space

FBX is offered as a secondary format via Blender headless conversion (`blender --background --python convert.py`), not as a primary export path.

### 4.2 Export Mode A — Baked Vertex Animation (morph targets)

No skeleton or skinning data required. Works immediately.

**GLB structure:**
```
Scenes: [Scene]
  └── Nodes: [BodyNode]
        └── Mesh: BodyMesh
              ├── Primitive
              │     ├── attributes.POSITION → frame 0 vertices (18439 × VEC3)
              │     ├── indices → triangle indices (36874×3 = 110622 × SCALAR)
              │     └── targets[0..N-1] → morph target deltas per frame
              └── weights: [0.0] × N
Animations: [MeshAnimation]
  └── Channel: weights on BodyNode
        ├── input → timestamps [0/fps, 1/fps, ..., (N-1)/fps]
        └── output → weight keyframes (STEP interpolation)
                      frame 0: [0, 0, 0, ...]
                      frame 1: [1, 0, 0, ...]
                      frame 2: [0, 1, 0, ...]
                      ...
```

**Coordinate conversion:** Undo `verts[..., [1,2]] *= -1` to restore Y-up before writing.

**Pros:** Simple, lossless vertex positions, no skeleton data needed.
**Cons:** Larger files, no skeletal editing in DCC tools, cannot retarget to other characters.

### 4.3 Export Mode B — Skeletal Animation (rigged)

Full skeleton with per-joint rotation animation. Requires MHR skeleton hierarchy and skinning weights.

**GLB structure:**
```
Scenes: [Scene]
  └── Nodes: [RootNode, Joint_0, Joint_1, ..., Joint_126]
        ├── Joint hierarchy via children[] references
        └── Mesh: BodyMesh
              └── Primitive
                    ├── attributes.POSITION → rest-pose vertices
                    ├── attributes.JOINTS_0 → joint indices per vertex (VEC4, UNSIGNED_SHORT)
                    ├── attributes.WEIGHTS_0 → blend weights per vertex (VEC4, FLOAT)
                    ├── attributes.JOINTS_1 → (if > 4 influences)
                    ├── attributes.WEIGHTS_1 → (if > 4 influences)
                    └── indices → triangle indices
Skins: [BodySkin]
  ├── joints → [Joint_0 .. Joint_126]
  └── inverseBindMatrices → 127 × MAT4
Animations: [PoseAnimation]
  └── Channels: 127 rotation channels + 1 root translation channel
        ├── Joint_0/rotation → quaternion keyframes
        ├── Joint_0/translation → root translation keyframes
        ├── Joint_1/rotation → quaternion keyframes
        └── ...
```

**Data requirements for Mode B:**

| Data | Source | Status |
|------|--------|--------|
| 127 joint names | MHR FBX assets (`lod0.fbx` from `assets.zip`) | Needs extraction |
| 127 parent indices | MHR FBX assets (same file) | Needs extraction |
| Rest-pose joint positions | `mhr_head.joint_rotation` buffer + MHR forward at zero pose | Available from model |
| (18439 × 127) skinning weights | MHR FBX assets (`lod0.fbx` vertex groups) | Needs extraction |
| Per-frame global rotations | `pred_global_rots` (127, 3, 3) | Available from pipeline |
| Per-frame local rotations | Computed: `local[j] = global[parent[j]]^T @ global[j]` | Computable |

**Computing local rotations from globals:**
```python
for j in range(127):
    parent = parent_indices[j]
    if parent == -1:  # root
        local_rot[j] = global_rot[j]
    else:
        local_rot[j] = global_rot[parent].T @ global_rot[j]
```

**Validation:** Reconstruct globals from locals via forward kinematics; max Frobenius norm error must be < 1e-5.

**Converting to quaternions for glTF:**
```python
# scipy or roma: rotation matrix → unit quaternion (x, y, z, w)
# glTF uses (x, y, z, w) quaternion ordering
quat = roma.rotmat_to_unitquat(local_rot)  # returns (w, x, y, z)
quat_gltf = quat[..., [1, 2, 3, 0]]       # reorder to (x, y, z, w)
```

---

## 5. MHR Skeleton — Joint Hierarchy

### 5.1 What we know

The MHR skeleton has **127 joints**. The `mhr70.py` metadata file provides names for 70 **keypoints** (body, hands, feet), but these are observation keypoints, not the skeleton joints themselves. The skeleton joints include:

- Body chain: pelvis → spine → chest → neck → head
- Arms: shoulder → elbow → wrist
- Legs: hip → knee → ankle → toes
- Hands: 16 joints per hand (thumb×4, index×4, middle×4, ring×4, pinky×4) — consistent with `hand_dofs_in_order = [3,1,1, 3,1,1, 3,1,1, 3,1,1, 2,3,1,1]`
- Face/jaw: a few joints for jaw rotation

### 5.2 Extracting the hierarchy

The MHR FBX assets (`assets.zip` from GitHub MHR releases) contain `lod0.fbx` with:
- Full 127-joint skeleton tree (names + parent-child)
- Skinning weights (4–8 influences per vertex)
- Rest pose (bind pose)

**Extraction approach:**
1. Download `assets.zip` from `https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip`
2. Open `lod0.fbx` in Blender (headless or UI) or parse with `pyfbx`/`assimp`
3. Walk the armature to extract: joint names, parent indices, rest-pose transforms
4. Extract vertex group weights → (18439, 127) sparse matrix
5. Hard-code into `utils/mhr_skeleton.py` as static data

**Fallback** (if FBX parsing is unreliable): Run the MHR model at zero pose to get rest-pose joint positions, then infer the hierarchy from the known DOF structure in `mhr_utils.py`. The parameter layout (3×trans + 3×global_rot + 130×body_pose + hands) encodes the joint ordering.

### 5.3 Rest-pose computation

Even without the FBX, we can get rest-pose joint positions by running MHR forward with zero parameters:

```python
rest_verts, rest_skel = mhr_model(
    shape_params=torch.zeros(1, 45),
    model_params=torch.zeros(1, 204),
    expr_params=None,
)
rest_joint_coords = rest_skel[:, :, :3] / 100  # centimeters → meters
```

The `joint_rotation` buffer provides the rest-pose rotation for each joint (used to compute inverse bind matrices for glTF skinning).

---

## 6. Data Quality — Known Issues and Cleanup Strategies

### 6.1 Issues already handled by the pipeline

| Issue | How it's handled | Where |
|-------|-----------------|-------|
| **Frame-to-frame jitter** | Kalman/EMA smoothing on body pose + hand params | `kalman_smooth_mhr_params_per_obj_id_adaptive()` |
| **Occlusion artifacts** | Interpolation between nearest visible frames | Same function, occlusion segment handling |
| **Sudden spikes** | Spike detection + suppression in dynamic segments | Same function, `spike_factor=2.2` |
| **Shape/scale drift** | Fixed to first-frame values per person | `sam3d_body.py` lines 2186–2211 |
| **Global rotation jitter** | EMA smoothing with occlusion interpolation | `ema_smooth_global_rot_per_obj_id_adaptive()` |

### 6.2 Residual issues after pipeline smoothing

These issues survive the existing smoothing and will be visible in exported animations:

#### 6.2.1 Foot sliding

**Problem:** Feet translate in world space even during ground contact. This is because MHR estimates body pose relative to the camera, not in a world coordinate frame. There is no explicit ground-plane constraint.

**Impact:** Distracting sliding motion when viewing from a fixed camera angle in DCC tools.

**Cleanup strategies:**

| Strategy | Where | Difficulty |
|----------|-------|-----------|
| **Foot contact detection + IK pinning** | Post-export in Blender/UE5 | Medium — requires detecting stance phases from ankle velocity |
| **Ground plane constraint** | Pre-export Python pass | Medium — detect ankle minima, project to ground, solve pelvis translation |
| **Manual cleanup in DCC** | Blender pose mode / UE5 Control Rig | Low — per-shot manual work |

**Recommended approach — post-export Python pass:**

1. Compute per-frame ankle velocity (finite difference on ankle joint positions)
2. Detect ground contact when velocity < threshold (e.g., < 0.02 m/frame)
3. During contact, pin ankle position to the ground plane (Y = ankle_min)
4. Solve pelvis translation adjustment via simple IK (pelvis–hip–knee–ankle chain)
5. Blend corrections over a few frames at stance-phase boundaries

#### 6.2.2 Temporal discontinuities at batch boundaries

**Problem:** The pipeline processes frames in batches of 64 (`batch_size`). Smoothing happens within each batch. At batch boundaries, parameters may not be continuous.

**Impact:** Visible "pops" every 64 frames in the exported animation.

**Cleanup strategies:**

| Strategy | Where | Difficulty |
|----------|-------|-----------|
| **Cross-batch smoothing pass** | Pre-export Python pass | Low — apply Gaussian/EMA smoothing across full sequence |
| **Overlap batches** | Pipeline modification | Medium — requires running overlapping frame windows |

**Recommended approach:** After collecting all frames, apply a lightweight temporal filter (e.g., Gaussian kernel with σ=2 frames) across the full sequence. This smooths batch boundary discontinuities without significantly altering the motion.

#### 6.2.3 Hand/finger noise

**Problem:** Finger pose estimation is inherently noisy — small occluded digits, ambiguous depth, limited training data. Even after PCA-space smoothing, finger poses can jitter or assume implausible configurations.

**Impact:** Distracting finger motion in close-up views. Less important for full-body shots.

**Cleanup strategies:**

| Strategy | Where | Difficulty |
|----------|-------|-----------|
| **Aggressive finger smoothing** | Pre-export Python pass | Low — apply strong low-pass filter to hand joint rotations |
| **Finger pose clamping** | Pre-export Python pass | Low — enforce anatomical joint limits |
| **Replace with static hand pose** | Pre-export option | Very low — lock fingers in a neutral pose |
| **Manual keyframe editing** | DCC tools | Medium — tedious for long sequences |

**Recommended approach:** Offer a "hand quality" option in the export UI:
- **Full**: Keep pipeline output as-is
- **Smoothed**: Apply extra temporal smoothing (Gaussian σ=3) to hand joints only
- **Static**: Replace all hand joint rotations with a neutral rest pose

#### 6.2.4 Self-intersections / interpenetration

**Problem:** Arms can intersect the torso, hands can clip through the body. The MHR model has no collision detection.

**Impact:** Visible in close-up renders. Usually acceptable at typical viewing distances.

**Cleanup strategies:**

| Strategy | Where | Difficulty |
|----------|-------|-----------|
| **Mesh collision detection + correction** | Pre-export Python pass | High — requires BVH spatial queries, iterative correction |
| **Manual posing in DCC** | Blender/UE5 | Medium — selective keyframe editing |
| **Ignore** | — | — (often acceptable for video rendering) |

**Recommended approach:** Defer to post-import manual correction. Automatic collision resolution is complex and risks introducing new artifacts. Document the issue in the import guides so users know to check for it.

#### 6.2.5 Root translation

**Problem:** MHR doesn't directly estimate world-space translation. The `global_trans` is set to zero in the pipeline (line 2225: `global_trans=pose_output["mhr"]["global_rot"] * 0`). Instead, `pred_cam_t` provides the camera-relative translation.

**Impact:** The exported mesh appears at the origin. For multi-person scenes, persons overlap unless camera translation is applied.

**Cleanup strategies:**

| Strategy | Where | Difficulty |
|----------|-------|-----------|
| **Apply camera translation as root motion** | Export-time | Low — add `pred_cam_t` to root joint translation per frame |
| **Export camera as separate object** | Export-time | Low — add animated camera node to glTF |
| **Both** | Export-time | Low |

**Recommended approach:** Apply `pred_cam_t` as root translation in the exported animation. This places each person at their camera-relative position, which is correct for rendering from the original viewpoint. Also export an animated camera with the estimated focal length for reference.

### 6.3 Pre-export cleanup pass (new module)

A new `utils/mesh_cleanup.py` module will apply optional post-pipeline corrections before GLB export:

```python
def cleanup_animation(
    export_data: PersonExportData,
    smooth_batch_boundaries: bool = True,   # Gaussian smooth across full sequence
    foot_contact_correction: bool = False,  # Ground plane pinning (experimental)
    hand_quality: str = "smoothed",         # "full", "smoothed", or "static"
    apply_root_translation: bool = True,    # Apply cam_t as root motion
) -> PersonExportData:
    """Apply post-pipeline cleanup before export."""
    ...
```

**Correction order matters:**
1. Apply root translation (reinterprets coordinate frame)
2. Smooth batch boundaries (temporal filter on rotations + translations)
3. Hand quality adjustment (temporal filter or replacement on finger joints)
4. Foot contact correction (modifies root + leg chain)

---

## 7. Import into Blender

### 7.1 glTF import

1. File → Import → glTF 2.0 (`.glb`)
2. Settings:
   - Bone Dir: Temperance (default)
   - Merge Vertices: off (MHR mesh is already clean)
3. After import:
   - Armature with 127 bones visible in Outliner
   - Mesh bound to armature via Armature modifier
   - Animation appears in Dope Sheet / Timeline
   - Play with Space bar to preview

### 7.2 Baked vertex animation (morph targets)

1. Import as above — shape keys appear on the mesh
2. Animation drives shape key values over time
3. In Dope Sheet, switch to Shape Key Editor to see keyframes
4. Cannot edit individual bone poses (no skeleton)

### 7.3 Retargeting to other rigs

**Auto-Rig Pro ($50, Blender Market):**
1. Select imported armature → ARP tab → Build Bone List
2. Map MHR bones to ARP's internal names (manual first time, save as preset)
3. Set target armature (e.g., Mixamo, MetaHuman proxy)
4. Click Remap → Bake animation onto target rig

**Free alternatives:**
- **Rokoko Blender plugin** (free): Semi-automatic bone mapping
- **Manual constraints**: Add Copy Rotation constraint per bone from source to target → Bake Action → Remove constraints

### 7.4 Cleanup in Blender

- **Foot sliding**: Add IK constraints on feet, bake to action, remove constraints
- **Jitter**: Graph Editor → select channels → Key → Smooth Keys
- **Finger noise**: Select finger bones → Graph Editor → Smooth or manually key
- **Self-intersection**: Pose mode → manually adjust problematic frames

### 7.5 Rendering in Blender

1. Set up HDRI lighting + ground plane
2. Add camera matching the exported focal length
3. Apply materials to the mesh (default grey or character texture)
4. Render animation: Output Properties → Format: FFmpeg → Render Animation (Ctrl+F12)

---

## 8. Import into Unreal Engine 5

### 8.1 glTF/FBX import

**glTF (via Interchange Framework, UE 5.1+):**
1. Content Browser → Import → select `.glb` file
2. Import dialog: Create New Skeleton, Import Mesh, Import Animation
3. Convert Scene checked (handles Y-up → Z-up)

**FBX (converted from GLB via Blender headless):**
1. Content Browser → Import → select `.fbx` file
2. Skeletal Mesh settings: Create New Skeleton, Import Mesh
3. Animation settings: Import as Animation Sequence

### 8.2 IK Retargeter → MetaHuman workflow

This is the primary UE5 use case: transferring recovered motion onto a MetaHuman character.

1. **Create IK Rig for the MHR skeleton:**
   - Right-click Skeleton → Create → IK Rig
   - Define chains:
     - Spine: pelvis → spine → chest → neck → head
     - LeftArm: left_shoulder → left_elbow → left_wrist
     - RightArm: right_shoulder → right_elbow → right_wrist
     - LeftLeg: left_hip → left_knee → left_ankle
     - RightLeg: right_hip → right_knee → right_ankle
     - LeftHand: per-finger chains (5 chains × 4 joints)
     - RightHand: per-finger chains (5 chains × 4 joints)
   - Set root bone and retarget root

2. **Create IK Retargeter:**
   - Source: MHR IK Rig (created above)
   - Target: `IK_Metahuman` (built-in)
   - Chain mapping: map each source chain to corresponding target chain
   - Adjust retarget pose if T-pose / A-pose don't match

3. **Export retargeted Animation Sequence:**
   - Right-click retargeted asset → Export Anim Sequence
   - Apply to MetaHuman Blueprint in Level Sequencer

### 8.3 Cleanup in UE5

- **Foot sliding**: Use Control Rig → Foot IK node → bake corrected animation
- **Root motion**: Enable Root Motion on Animation Sequence if needed
- **Blending**: Use Animation Montage / Blend Space to smooth transitions

### 8.4 Rendering in UE5

1. Place MetaHuman in level with retargeted animation
2. Set up Cine Camera with matching focal length
3. Level Sequencer → add camera + character tracks
4. Movie Render Queue → output MP4/EXR sequence

---

## 9. SMPL-X Conversion Path (Deferred)

**Status:** Blocked pending SMPL-X model file access (academic registration at smpl-x.is.tue.mpg.de).

**When available:**
1. Use MHR repo's `tools/mhr_smpl_conversion/` for barycentric vertex correspondence
2. Fit SMPL-X pose parameters (55 joints) from converted mesh
3. Export with standard SMPL-X skeleton (well-known hierarchy, broad tool support)
4. Meshcapade `mc-unreal` plugin provides pre-built SMPL-X IK Rig for UE5

**Advantages over native MHR export:** SMPL-X has more tooling support (Meshcapade, SMPLify, HumanML3D). The 55-joint skeleton is simpler and more widely recognized by auto-mapping tools.

---

## 10. Export API

### 10.1 Python API

```python
from utils.mesh_export import PersonExportData, export_baked_vertex_glb
from utils.mesh_export import export_skeletal_glb  # Phase 2
from utils.mesh_cleanup import cleanup_animation

# Accumulate data during pipeline (called per frame)
collect_frame_data(export_data, outputs, id_current, faces, fps)

# Optional cleanup pass
for person_id, data in export_data.items():
    data = cleanup_animation(
        data,
        smooth_batch_boundaries=True,
        hand_quality="smoothed",
        apply_root_translation=True,
    )
    export_data[person_id] = data

# Export GLB files
paths = export_all_persons_glb(export_data, output_dir)

# Or create downloadable ZIP
zip_path = create_export_zip(export_data, output_dir)
```

### 10.2 Gradio UI

After the 4D Result video display:

```
┌─────────────────────────────────────────────┐
│  Export 3D Mesh                              │
│                                              │
│  Format: ○ Baked Vertex (.glb)               │
│          ● Rigged Skeleton (.glb)            │
│          ○ SMPL-X (.glb) [disabled]          │
│                                              │
│  ☑ Apply root translation                    │
│  ☑ Smooth batch boundaries                   │
│  Hand quality: [Smoothed ▾]                  │
│                                              │
│  [Export 3D Mesh]                            │
│                                              │
│  📁 mesh_export.zip (download)               │
└─────────────────────────────────────────────┘
```

### 10.3 CLI (offline_app.py)

```bash
python scripts/offline_app.py \
    --input_video input.mp4 \
    --export-mesh \
    --export-format rigged-gltf \
    --hand-quality smoothed
```

Output: `{output_dir}/export/{person_id}/animated_mesh.glb`

---

## 11. File Layout

### New files

| File | Purpose |
|------|---------|
| `utils/mesh_export.py` | Core export: data collection, GLB writer (baked + skeletal) |
| `utils/mhr_skeleton.py` | MHR 127-joint skeleton definition (names, parents, rest-pose) |
| `utils/mesh_cleanup.py` | Post-pipeline animation cleanup (smoothing, foot contact, hands) |
| `scripts/convert_glb_to_fbx.py` | Blender headless GLB→FBX conversion script |
| `docs/mesh-export-design.md` | This document |
| `docs/blender-import-guide.md` | Blender import + retargeting guide |
| `docs/ue5-import-guide.md` | UE5 import + IK Retargeter guide |

### Modified files

| File | Change |
|------|--------|
| `app.py` | Add data collection hook in `on_4d_generation()`, add export UI controls |
| `scripts/offline_app.py` | Add `--export-mesh` flag |
| `pyproject.toml` | Add `pygltflib` dependency |

---

## 12. Implementation Phases

```
Phase 0: MHR Asset Extraction
  ├── 0.1 Extract joint hierarchy from lod0.fbx
  ├── 0.2 Extract skinning weights from lod0.fbx
  └── 0.3 SMPL-X prep (DEFERRED)

Phase 1: Baked Vertex Export (no skeleton needed)
  ├── 1.1 Data collection hook in app.py
  ├── 1.2 GLB morph target writer
  └── 1.3 FBX conversion script (optional)

Phase 2: Skeletal Export (needs Phase 0)
  ├── 2.1 mhr_skeleton.py module
  ├── 2.2 Local rotation computation
  └── 2.3 GLB skeleton + skinning + animation writer

Phase 3: Data Cleanup Module
  ├── 3.1 Batch boundary smoothing
  ├── 3.2 Hand quality options
  ├── 3.3 Root translation application
  └── 3.4 Foot contact correction (experimental)

Phase 4: UI Integration
  ├── 4.1 Gradio export controls
  └── 4.2 offline_app.py flags

Phase 5: Documentation
  ├── 5.1 Blender import guide
  └── 5.2 UE5 import guide

Phase 6: Validation
  ├── 6.1 Round-trip coordinate tests
  ├── 6.2 GLB → Blender import verification
  └── 6.3 Retargeting to MetaHuman test
```

**Dependency graph:**
```
Phase 1 ──────────────────▶ Phase 4.1 ──▶ Deploy
Phase 0 ──▶ Phase 2 ──────▶ Phase 4.1
Phase 1/2 ─▶ Phase 3 ─────▶ Phase 4.1
Phase 2 ──────────────────▶ Phase 5
Phase 1 ──────────────────▶ Phase 6
```

Phases 1 and 0 can proceed in parallel. Phase 3 (cleanup) can start once either Phase 1 or Phase 2 is complete.

---

## 13. Design Decisions (Resolved)

### 13.1 MHR FBX parsing

**Decision:** Use [Blender](https://www.blender.org/) headless (`blender --background --python extract.py`) for one-time extraction of skeleton hierarchy, joint names, and skinning weights from [`lod0.fbx`](https://github.com/facebookresearch/MHR/releases/tag/v1.0.0) in the MHR assets.

**Rationale:** Blender has the most reliable FBX importer — it handles all FBX versions and correctly parses bone hierarchies, vertex groups (skinning weights), and rest-pose transforms. Alternatives considered:

| Approach | Why not |
|----------|---------|
| [Autodesk FBX SDK](https://aps.autodesk.com/developer/overview/fbx-sdk) (via [PyMomentum](https://github.com/facebookresearch/momentum)) | Proprietary license, platform-specific binaries |
| [PyAssimp](https://pypi.org/project/pyassimp/) | Reverse-engineered FBX parser — can fail on newer formats or complex skinning |
| [pyfbx](https://github.com/nannafudge/pyfbx) / [py-fbx](https://pypi.org/project/py-fbx/) | Immature, may not handle vertex groups correctly |

This is a **one-time local extraction task**, not a runtime dependency. The extracted data is committed as a static file (`utils/mhr_skeleton.py` and/or `utils/mhr_skinning_weights.npz`). Blender is never needed at runtime or in Docker.

The Blender Python API provides everything we need:
- `bpy.ops.import_scene.fbx(filepath=...)` — [import operator](https://docs.blender.org/api/current/bpy.ops.import_scene.html)
- `armature.data.bones` — [Bone](https://docs.blender.org/api/current/bpy.types.Bone.html) hierarchy traversal (`.parent`, `.children`, `.matrix_local`)
- `mesh.vertex_groups` — [VertexGroup](https://docs.blender.org/api/current/bpy.types.VertexGroup.html) skinning weight extraction

### 13.2 Joint order verification

**Decision:** Verify empirically by comparing rest-pose joint positions from the FBX against a zero-pose MHR forward pass.

**Approach:**
1. Extract 127 bone `head_local` positions from `lod0.fbx` via Blender
2. Run MHR forward with zero pose parameters to get `pred_joint_coords` (127, 3) at rest
3. Compare the two sets of 3D positions — if they match index-for-index (within tolerance < 1e-3), the ordering is confirmed
4. If they don't match, compute the permutation mapping by nearest-neighbor matching and hardcode it

**Expected outcome:** The ordering almost certainly matches. Both the FBX asset and the runtime model are produced by the same [Momentum library](https://github.com/facebookresearch/momentum) at Meta. Reordering joints between export and runtime would break their own internal tooling. But we verify to eliminate the risk.

### 13.3 Skinning weight sparsity

**Decision:** 4 influences per vertex (1 set of `JOINTS_0`/`WEIGHTS_0` in glTF).

**Rationale:** The [MHR paper](https://arxiv.org/html/2511.15586v1) specifies:
- LoD 0: up to 8 influences per vertex
- LoD 1–4: 4 influences per vertex

Our runtime model loads at `lod=1` (`mhr_head.py` line 111), so the mesh deformation we're exporting was computed with 4-influence skinning. Using more would add data that doesn't match the actual deformation.

Additionally, 4 influences is the [industry standard](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#meshes) for real-time rendering — UE5 and Unity default to 4. If the FBX contains vertices with more than 4 non-zero weights, we take the top 4 and renormalize to sum to 1.0.

### 13.4 Foot contact detection

**Decision:** Skip for v1. Document the issue and recommend downstream tools.

**Rationale:** Foot sliding (feet translating during ground contact) is a known artifact of monocular 3D reconstruction. The pipeline reconstructs body pose relative to the camera, with no explicit ground-plane constraint. However, foot contact detection is a well-solved problem in downstream DCC tools:

- [Cascadeur](https://cascadeur.com/) — built-in foot sliding cleanup with automatic contact detection (supports [glTF import](https://cascadeur.com/blog/cascadeur-2025-1-new-animation-tools-new-ai-tools) as of 2025.1)
- [Auto-Rig Pro](https://blendermarket.com/products/auto-rig-pro) ($50) — foot pinning via IK constraints in Blender
- UE5 [Control Rig](https://dev.epicgames.com/documentation/en-us/unreal-engine/control-rig-in-unreal-engine) — procedural foot IK with ground detection
- [UnderPressure](https://github.com/InterDigitalInc/UnderPressure) — open-source neural foot contact detection + IK footskate cleanup

For a future v2, an optional pre-export pass with adaptive velocity thresholding (histogram-based bimodal split per foot) could provide a quick automatic improvement.

### 13.5 Camera export

**Decision:** Skip for v1.

**Rationale:** The pipeline estimates focal length once from frame 0 only (`app.py` line 849). The camera is effectively static at the origin with no rotation or movement — each person's position is encoded in their `pred_cam_t` translation, which is already baked into the exported vertex positions.

Adding a [glTF camera node](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-camera) would be trivial (~10 lines) but low-value: the camera doesn't move, and the focal length is already saved in the per-frame `.json` camera files. If users need to match the original perspective for compositing, they can read the focal length from those JSON files.

### 13.6 Multi-person export format

**Decision:** Separate GLB file per person.

**Output structure:**
```
{output_dir}/export/
  1/animated_mesh.glb    ← Person 1
  2/animated_mesh.glb    ← Person 2
  mesh_export.zip        ← All persons bundled for download
```

**Rationale:** Separate files are simpler to implement (already done in `utils/mesh_export.py`), give users flexibility to import/delete individual people, and avoid the complexity of multiple meshes/skins/animations in a single glTF scene graph. Since all meshes share the same camera coordinate space, importing multiple GLBs into the same Blender scene places them in correct relative positions automatically.
