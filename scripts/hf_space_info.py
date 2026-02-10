#!/usr/bin/env python3
"""Query HuggingFace Space API and print a clean status summary.

Usage:
    python3 scripts/hf_space_info.py [--poll N] [--interval S]

Options:
    --poll N       Poll up to N times waiting for a target stage (default: 1, no polling)
    --interval S   Seconds between polls (default: 20)

Reads HF_TOKEN from .env.local in the repo root.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

SPACE_ID = "troutmoose/sam-body4d"
API_URL = f"https://huggingface.co/api/spaces/{SPACE_ID}"


def read_token():
    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env.local")
    with open(env_file) as f:
        for line in f:
            if line.startswith("HF_TOKEN="):
                return line.strip().split("=", 1)[1]
    print("ERROR: HF_TOKEN not found in .env.local", file=sys.stderr)
    sys.exit(1)


def fetch_space_info(token):
    req = urllib.request.Request(API_URL)
    req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def print_summary(data):
    runtime = data.get("runtime", {})
    hardware = runtime.get("hardware", {})
    storage = runtime.get("storage", {})

    print(f"Stage:      {runtime.get('stage', 'UNKNOWN')}")
    print(f"SHA:        {runtime.get('sha', 'N/A')}")
    print(f"Repo SHA:   {data.get('sha', 'N/A')}")
    print(f"Hardware:   {hardware.get('current', 'N/A')}")
    print(f"Storage:    {storage.get('current', 'N/A') if storage else 'none'}")
    print(f"Dev Mode:   {runtime.get('devMode', False)}")
    print(f"URL:        {data.get('host', 'N/A')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll", type=int, default=1)
    parser.add_argument("--interval", type=int, default=20)
    args = parser.parse_args()

    token = read_token()

    for i in range(args.poll):
        if i > 0:
            print(f"\n--- Poll {i + 1}/{args.poll} (waiting {args.interval}s) ---")
            time.sleep(args.interval)

        data = fetch_space_info(token)
        print_summary(data)

        stage = data.get("runtime", {}).get("stage", "")
        if stage in ("RUNNING", "APP_STARTING", "RUNNING_APP_STARTING"):
            # Check if runtime SHA matches repo SHA
            runtime_sha = data.get("runtime", {}).get("sha", "")
            repo_sha = data.get("sha", "")
            if runtime_sha == repo_sha:
                print(f"\nDeployed SHA matches repo SHA: {repo_sha[:8]}")
                break
            elif args.poll > 1:
                print(f"\nRuntime SHA ({runtime_sha[:8]}) != Repo SHA ({repo_sha[:8]}) — waiting for rebuild...")
                continue
            else:
                break
        elif "ERROR" in stage:
            print(f"\nSpace is in error state: {stage}")
            break
        elif "BUILDING" in stage:
            if args.poll == 1:
                print("\nSpace is building...")
                break
            print("\nStill building...")
            continue
        else:
            if args.poll == 1:
                break
            continue

    # Final stage
    stage = data.get("runtime", {}).get("stage", "")
    if args.poll > 1 and "BUILDING" in stage:
        print(f"\nStill building after {args.poll} polls. Check build logs.")


if __name__ == "__main__":
    main()
