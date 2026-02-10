#!/usr/bin/env python3
"""Fetch HuggingFace Space logs (build or run) and print them.

Usage:
    python3 scripts/hf_space_logs.py build
    python3 scripts/hf_space_logs.py run
    python3 scripts/hf_space_logs.py build --lines 50
    python3 scripts/hf_space_logs.py run --timeout 15

Reads HF_TOKEN from .env.local in the repo root.
Connects to the SSE log stream, collects for --timeout seconds, then prints.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

SPACE_ID = "troutmoose/sam-body4d"
BASE_URL = f"https://huggingface.co/api/spaces/{SPACE_ID}/logs"


def read_token():
    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env.local")
    with open(env_file) as f:
        for line in f:
            if line.startswith("HF_TOKEN="):
                return line.strip().split("=", 1)[1]
    print("ERROR: HF_TOKEN not found in .env.local", file=sys.stderr)
    sys.exit(1)


def fetch_logs(log_type, token, timeout=10, max_lines=100):
    url = f"{BASE_URL}/{log_type}"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")

    lines = []
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            while time.time() - start < timeout:
                line = resp.readline()
                if not line:
                    break
                line = line.decode("utf-8", errors="replace").strip()
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str:
                        try:
                            entry = json.loads(data_str)
                            text = entry.get("data", "")
                            if text:
                                lines.append(text)
                        except json.JSONDecodeError:
                            lines.append(data_str)
    except Exception:
        pass  # timeout or connection closed — expected for SSE

    # Return last N lines
    return lines[-max_lines:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("type", choices=["build", "run"])
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--lines", type=int, default=100)
    args = parser.parse_args()

    token = read_token()
    lines = fetch_logs(args.type, token, timeout=args.timeout, max_lines=args.lines)

    if not lines:
        print(f"No {args.type} logs available.")
    else:
        print(f"--- {args.type.upper()} LOGS (last {len(lines)} lines) ---")
        for line in lines:
            print(line)


if __name__ == "__main__":
    main()
