"""
Quick validation that the correct export code is deployed on the HF Space.
Run after every Dev Mode restart to catch stale code before wasting a pipeline run.

Usage:
    python3 scripts/validate_export_code.py
"""

import subprocess
import sys

HOST = "hf-sam3"
CHECKS = [
    # (description, grep pattern, should_match: True=must exist, False=must NOT exist)
    ("allowed_paths in launch()", "allowed_paths=", True),
    ("export_file widget", 'export_file = gr.File', True),
    ("plain zip_path return (not gr.update/gr.File)", "return video_out, zip_path", True),
    ("collect_frame_data import", "from utils.mesh_export import", True),
    ("collect_frame_data call in loop", "collect_frame_data(", True),
    ("create_export_zip call", "create_export_zip(export_data", True),
    ("export_data dict init", "export_data = {}", True),
]


def run_ssh(cmd):
    result = subprocess.run(
        ["ssh", HOST, cmd],
        capture_output=True, text=True, timeout=10,
    )
    return result.stdout, result.stderr, result.returncode


def main():
    # First check SSH connectivity
    out, err, rc = run_ssh("cat /app/app.py | wc -l")
    if rc != 0:
        print(f"FAIL: Cannot SSH to {HOST}: {err.strip()}")
        sys.exit(1)
    print(f"Connected to {HOST}, app.py has {out.strip()} lines\n")

    failures = 0
    for desc, pattern, should_match in CHECKS:
        escaped = pattern.replace('"', '\\"').replace("'", "'\\''")
        out, err, rc = run_ssh(f'grep -c "{escaped}" /app/app.py')
        count = int(out.strip()) if out.strip().isdigit() else 0

        if should_match and count > 0:
            print(f"  PASS  {desc}")
        elif not should_match and count == 0:
            print(f"  PASS  {desc}")
        elif should_match and count == 0:
            print(f"  FAIL  {desc} — NOT FOUND")
            failures += 1
        else:
            print(f"  FAIL  {desc} — found but should NOT exist")
            failures += 1

    # Also check the on_4d_generation_ui function doesn't use gr.update or gr.File wrapper
    out, _, _ = run_ssh('grep -A3 "def on_4d_generation_ui" /app/app.py')
    if "gr.update" in out or "gr.File(" in out:
        print(f"  FAIL  on_4d_generation_ui still wraps return with gr.update/gr.File")
        failures += 1
    else:
        print(f"  PASS  on_4d_generation_ui returns plain values")

    print()
    if failures:
        print(f"FAILED: {failures} check(s). Code on Space is stale or wrong.")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED.")
        sys.exit(0)


if __name__ == "__main__":
    main()
