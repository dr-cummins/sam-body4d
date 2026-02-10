#!/usr/bin/env python3
"""Validate import resolution for the HF Space deployment.

Tests that sys.path ordering in app.py resolves modules correctly:
1. Project utils found before diffusion_vas/utils.py shadows it
2. diffusion_vas pipeline found via nested models/ directory
3. sam_3d_body kalman utils resolve to project utils

Run before deploying to catch import collisions early.
No heavy dependencies needed (torch, cv2, etc. not imported).
"""
import sys
import os
import importlib.util

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hf_space = os.environ.get(
        "HF_REPO", os.path.join(os.path.dirname(project_root), "hf-space")
    )

    # Test against whichever repo is specified
    target = hf_space if os.path.isdir(hf_space) else project_root
    print(f"Validating imports in: {target}")

    errors = []

    # Simulate app.py sys.path setup
    sys.path.insert(0, target)
    sys.path.append(os.path.join(target, "models", "sam_3d_body"))

    # 1. utils must resolve to project utils BEFORE diffusion_vas is added
    spec = importlib.util.find_spec("utils")
    if spec is None:
        errors.append("utils package not found")
    elif not spec.submodule_search_locations:
        errors.append(f"utils is a module, not a package: {spec.origin}")
    else:
        utils_path = str(spec.submodule_search_locations)
        if "diffusion_vas" in utils_path:
            errors.append(f"utils resolves to diffusion_vas: {utils_path}")
        else:
            print(f"  OK  utils -> {utils_path}")

    # 2. Add diffusion_vas to front (simulates app.py line 37)
    sys.path.insert(0, os.path.join(target, "models", "diffusion_vas"))

    # 3. pipeline_diffusion_vas must be findable via nested models/
    pipe_spec = importlib.util.find_spec(
        "models.diffusion_vas.pipeline_diffusion_vas"
    )
    if pipe_spec is None:
        errors.append("models.diffusion_vas.pipeline_diffusion_vas NOT FOUND")
    else:
        expected = os.path.join(
            "models", "diffusion_vas", "models", "diffusion_vas",
            "pipeline_diffusion_vas.py"
        )
        if expected not in pipe_spec.origin:
            errors.append(
                f"pipeline_diffusion_vas at wrong path: {pipe_spec.origin}"
            )
        else:
            print(f"  OK  pipeline_diffusion_vas -> {pipe_spec.origin}")

    # 4. Verify utils would be shadowed WITHOUT caching (proves ordering matters)
    if "utils" in sys.modules:
        del sys.modules["utils"]
    shadow_spec = importlib.util.find_spec("utils")
    shadow_path = str(shadow_spec.origin or shadow_spec.submodule_search_locations)
    if "diffusion_vas/utils.py" in shadow_path:
        print(f"  OK  utils shadowing confirmed (ordering protection works)")
    else:
        print(f"  WARN  utils NOT shadowed by diffusion_vas — unexpected")

    # 5. Check BPE vocab exists
    bpe_path = os.path.join(
        target, "models", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz"
    )
    if os.path.exists(bpe_path):
        size_mb = os.path.getsize(bpe_path) / 1024 / 1024
        print(f"  OK  BPE vocab present ({size_mb:.1f} MB)")
    else:
        errors.append(f"BPE vocab missing: {bpe_path}")

    # 6. Check Gradio launch config
    app_path = os.path.join(target, "app.py")
    if os.path.exists(app_path):
        with open(app_path) as f:
            content = f.read()
        if 'server_name="0.0.0.0"' in content:
            print(f"  OK  Gradio server_name configured")
        else:
            errors.append("Gradio launch missing server_name='0.0.0.0'")

    print()
    if errors:
        print(f"FAILED ({len(errors)} errors):")
        for e in errors:
            print(f"  - {e}")
        return 1
    else:
        print("ALL CHECKS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
