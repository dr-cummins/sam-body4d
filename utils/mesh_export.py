"""
Mesh export utilities for SAM-Body4D.

Exports per-person animated 3D meshes as glTF/GLB files.
Supports two export modes:
  1. Baked vertex animation (morph targets) — no skeleton needed
  2. Skeletal animation (rigged) — requires MHR skeleton data (Phase 2)
"""

import os
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

try:
    import pygltflib
except ImportError:
    pygltflib = None


@dataclass
class PersonExportData:
    """Accumulated per-frame data for a single person."""
    vertices: List[np.ndarray] = field(default_factory=list)       # (V, 3) per frame
    joint_coords: List[np.ndarray] = field(default_factory=list)   # (J, 3) per frame
    global_rots: List[np.ndarray] = field(default_factory=list)    # (J, 3, 3) per frame
    cam_t: List[np.ndarray] = field(default_factory=list)          # (3,) per frame
    focal_length: List[float] = field(default_factory=list)
    faces: Optional[np.ndarray] = None                             # (F, 3), shared
    fps: float = 30.0


def collect_frame_data(
    export_data: Dict[int, PersonExportData],
    outputs,
    id_current,
    faces: np.ndarray,
    fps: float,
):
    """
    Collect per-frame data from a single frame's pipeline output.

    Args:
        export_data: dict mapping person_id → PersonExportData (mutated in place)
        outputs: list of per-person output dicts from the pipeline (or None)
        id_current: list of object IDs for each person
        faces: (F, 3) triangle indices
        fps: video framerate
    """
    if outputs is None:
        return

    for pid, person_output in enumerate(outputs):
        person_id = pid + 1
        if person_id not in export_data:
            export_data[person_id] = PersonExportData(fps=fps)
            export_data[person_id].faces = faces

        data = export_data[person_id]
        data.vertices.append(np.array(person_output["pred_vertices"], dtype=np.float32))
        data.focal_length.append(float(person_output["focal_length"]))
        data.cam_t.append(np.array(person_output["pred_cam_t"], dtype=np.float32))

        if "pred_joint_coords" in person_output and person_output["pred_joint_coords"] is not None:
            data.joint_coords.append(np.array(person_output["pred_joint_coords"], dtype=np.float32))
        if "pred_global_rots" in person_output and person_output["pred_global_rots"] is not None:
            data.global_rots.append(np.array(person_output["pred_global_rots"], dtype=np.float32))


def _vertices_to_gltf_coords(vertices: np.ndarray) -> np.ndarray:
    """
    Convert from SAM-3D-Body camera coordinate system to glTF Y-up.

    The pipeline applies verts[..., [1, 2]] *= -1 (in mhr_head.py line 340)
    to convert from MHR's Y-up to camera coords. We undo this to restore
    Y-up for glTF.
    """
    out = vertices.copy()
    out[:, 1] *= -1
    out[:, 2] *= -1
    return out


def _pack_float32_buffer(arr: np.ndarray) -> bytes:
    """Pack a float32 numpy array into a little-endian binary buffer."""
    return arr.astype(np.float32).tobytes()


def _pack_uint32_buffer(arr: np.ndarray) -> bytes:
    """Pack a uint32 numpy array into a little-endian binary buffer."""
    return arr.astype(np.uint32).tobytes()


def _compute_bounds(arr: np.ndarray):
    """Compute min/max for a (N, 3) array, return as lists."""
    mins = arr.min(axis=0).tolist()
    maxs = arr.max(axis=0).tolist()
    return mins, maxs


def export_baked_vertex_glb(
    data: PersonExportData,
    output_path: str,
) -> str:
    """
    Export a baked vertex animation as a GLB file using morph targets.

    Frame 0 is the base mesh. Each subsequent frame is a morph target
    containing vertex position deltas. An animation drives morph weights
    so that exactly one target is active per frame.

    Args:
        data: PersonExportData with accumulated frames
        output_path: path to write the .glb file

    Returns:
        The output path written.
    """
    if pygltflib is None:
        raise ImportError("pygltflib is required for GLB export. Install with: pip install pygltflib")

    n_frames = len(data.vertices)
    if n_frames == 0:
        raise ValueError("No frames to export")

    faces = data.faces
    n_verts = data.vertices[0].shape[0]
    n_faces = faces.shape[0]
    fps = data.fps

    # Convert all vertices to glTF Y-up coordinate system
    all_verts = [_vertices_to_gltf_coords(v) for v in data.vertices]

    base_verts = all_verts[0]

    # Compute morph target deltas (each target is displacement from frame 0)
    morph_deltas = []
    for i in range(1, n_frames):
        delta = all_verts[i] - base_verts
        morph_deltas.append(delta.astype(np.float32))

    # Build binary buffer
    buffer_data = bytearray()

    # --- Section 1: Base mesh position data ---
    base_pos_offset = len(buffer_data)
    base_pos_bytes = _pack_float32_buffer(base_verts)
    buffer_data.extend(base_pos_bytes)
    base_pos_length = len(base_pos_bytes)

    # --- Section 2: Index data (faces) ---
    # Pad to 4-byte alignment
    while len(buffer_data) % 4 != 0:
        buffer_data.append(0)
    index_offset = len(buffer_data)
    index_bytes = _pack_uint32_buffer(faces.flatten())
    buffer_data.extend(index_bytes)
    index_length = len(index_bytes)

    # --- Section 3: Morph target deltas ---
    morph_offsets = []
    morph_lengths = []
    for delta in morph_deltas:
        while len(buffer_data) % 4 != 0:
            buffer_data.append(0)
        offset = len(buffer_data)
        delta_bytes = _pack_float32_buffer(delta)
        buffer_data.extend(delta_bytes)
        morph_offsets.append(offset)
        morph_lengths.append(len(delta_bytes))

    # --- Section 4: Animation data ---
    # Time keyframes: one per frame
    times = np.array([i / fps for i in range(n_frames)], dtype=np.float32)
    while len(buffer_data) % 4 != 0:
        buffer_data.append(0)
    time_offset = len(buffer_data)
    time_bytes = _pack_float32_buffer(times)
    buffer_data.extend(time_bytes)
    time_length = len(time_bytes)

    # Morph weight keyframes: at each time step, exactly one morph target has weight 1.0
    # Weight layout: n_frames rows × (n_frames-1) morph targets
    n_targets = max(n_frames - 1, 0)
    if n_targets > 0:
        weights = np.zeros((n_frames, n_targets), dtype=np.float32)
        # Frame 0: all weights 0 (base mesh shown)
        # Frame i (i>=1): weight[i-1] = 1.0
        for i in range(1, n_frames):
            weights[i, i - 1] = 1.0

        while len(buffer_data) % 4 != 0:
            buffer_data.append(0)
        weights_offset = len(buffer_data)
        weights_bytes = _pack_float32_buffer(weights.flatten())
        buffer_data.extend(weights_bytes)
        weights_length = len(weights_bytes)

    # Build glTF structure
    gltf = pygltflib.GLTF2(
        asset=pygltflib.Asset(version="2.0", generator="SAM-Body4D"),
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(mesh=0, name="Body")],
        meshes=[],
        accessors=[],
        bufferViews=[],
        buffers=[pygltflib.Buffer(byteLength=len(buffer_data))],
        animations=[],
    )

    # --- Buffer Views ---
    # 0: base positions
    gltf.bufferViews.append(pygltflib.BufferView(
        buffer=0,
        byteOffset=base_pos_offset,
        byteLength=base_pos_length,
        target=pygltflib.ARRAY_BUFFER,
    ))
    # 1: indices
    gltf.bufferViews.append(pygltflib.BufferView(
        buffer=0,
        byteOffset=index_offset,
        byteLength=index_length,
        target=pygltflib.ELEMENT_ARRAY_BUFFER,
    ))
    # 2+: morph target buffer views
    for i, (off, length) in enumerate(zip(morph_offsets, morph_lengths)):
        gltf.bufferViews.append(pygltflib.BufferView(
            buffer=0,
            byteOffset=off,
            byteLength=length,
            target=pygltflib.ARRAY_BUFFER,
        ))

    # Animation buffer views (after morph targets)
    anim_time_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(pygltflib.BufferView(
        buffer=0,
        byteOffset=time_offset,
        byteLength=time_length,
    ))

    if n_targets > 0:
        anim_weights_bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(pygltflib.BufferView(
            buffer=0,
            byteOffset=weights_offset,
            byteLength=weights_length,
        ))

    # --- Accessors ---
    # 0: base positions
    base_min, base_max = _compute_bounds(base_verts)
    gltf.accessors.append(pygltflib.Accessor(
        bufferView=0,
        byteOffset=0,
        componentType=pygltflib.FLOAT,
        count=n_verts,
        type="VEC3",
        max=base_max,
        min=base_min,
    ))
    # 1: indices
    gltf.accessors.append(pygltflib.Accessor(
        bufferView=1,
        byteOffset=0,
        componentType=pygltflib.UNSIGNED_INT,
        count=n_faces * 3,
        type="SCALAR",
        max=[int(faces.max())],
        min=[int(faces.min())],
    ))
    # 2+: morph target position deltas
    morph_accessor_start = 2
    for i, delta in enumerate(morph_deltas):
        d_min, d_max = _compute_bounds(delta)
        gltf.accessors.append(pygltflib.Accessor(
            bufferView=2 + i,
            byteOffset=0,
            componentType=pygltflib.FLOAT,
            count=n_verts,
            type="VEC3",
            max=d_max,
            min=d_min,
        ))

    # Animation time accessor
    anim_time_acc_idx = len(gltf.accessors)
    gltf.accessors.append(pygltflib.Accessor(
        bufferView=anim_time_bv_idx,
        byteOffset=0,
        componentType=pygltflib.FLOAT,
        count=n_frames,
        type="SCALAR",
        max=[float(times[-1])],
        min=[float(times[0])],
    ))

    # Animation weights accessor
    if n_targets > 0:
        anim_weights_acc_idx = len(gltf.accessors)
        gltf.accessors.append(pygltflib.Accessor(
            bufferView=anim_weights_bv_idx,
            byteOffset=0,
            componentType=pygltflib.FLOAT,
            count=n_frames * n_targets,
            type="SCALAR",
            max=[1.0],
            min=[0.0],
        ))

    # --- Mesh with morph targets ---
    morph_targets = []
    for i in range(n_targets):
        morph_targets.append(pygltflib.Attributes(POSITION=morph_accessor_start + i))

    primitive = pygltflib.Primitive(
        attributes=pygltflib.Attributes(POSITION=0),
        indices=1,
        targets=morph_targets if morph_targets else None,
    )
    gltf.meshes.append(pygltflib.Mesh(
        primitives=[primitive],
        name="BodyMesh",
        weights=[0.0] * n_targets if n_targets > 0 else None,
    ))

    # --- Animation ---
    if n_targets > 0:
        sampler = pygltflib.AnimationSampler(
            input=anim_time_acc_idx,
            output=anim_weights_acc_idx,
            interpolation="STEP",
        )
        channel = pygltflib.AnimationChannel(
            sampler=0,
            target=pygltflib.AnimationChannelTarget(
                node=0,
                path="weights",
            ),
        )
        gltf.animations.append(pygltflib.Animation(
            name="MeshAnimation",
            samplers=[sampler],
            channels=[channel],
        ))

    # Set binary blob
    gltf.set_binary_blob(bytes(buffer_data))

    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gltf.save(output_path)

    return output_path


def export_all_persons_glb(
    export_data: Dict[int, PersonExportData],
    output_dir: str,
) -> List[str]:
    """
    Export GLB files for all persons in the export data.

    Args:
        export_data: dict mapping person_id → PersonExportData
        output_dir: directory to write files into

    Returns:
        List of written file paths.
    """
    paths = []
    export_dir = os.path.join(output_dir, "export")
    os.makedirs(export_dir, exist_ok=True)

    for person_id, data in sorted(export_data.items()):
        person_dir = os.path.join(export_dir, str(person_id))
        os.makedirs(person_dir, exist_ok=True)
        out_path = os.path.join(person_dir, "animated_mesh.glb")
        export_baked_vertex_glb(data, out_path)
        paths.append(out_path)

    return paths


def create_export_zip(
    export_data: Dict[int, PersonExportData],
    output_dir: str,
) -> str:
    """
    Export all persons and package into a ZIP file for download.

    Args:
        export_data: dict mapping person_id → PersonExportData
        output_dir: base output directory

    Returns:
        Path to the created ZIP file.
    """
    import zipfile

    glb_paths = export_all_persons_glb(export_data, output_dir)

    zip_path = os.path.join(output_dir, "mesh_export.zip")
    export_dir = os.path.join(output_dir, "export")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in glb_paths:
            arcname = os.path.relpath(path, export_dir)
            zf.write(path, arcname)

    return zip_path
