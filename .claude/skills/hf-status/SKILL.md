---
name: hf-status
description: Check the current state of the HuggingFace Space troutmoose/sam-body4d. Use when asked about space status, logs, or deployment state.
allowed-tools: Bash(python3 *)
---

# HF Space Status Check

Check the current state of the HuggingFace Space `troutmoose/sam-body4d`.

**IMPORTANT: One command per Bash call. Never pipe, chain, or redirect commands.**

## Steps

1. Run the space info script:
   ```bash
   python3 /Users/thomascummins/dev/chromatica/projects/sam-body4d/scripts/hf_space_info.py
   ```

2. Fetch run logs and build logs **in parallel** (two separate Bash calls):

   **Run logs:**
   ```bash
   python3 /Users/thomascummins/dev/chromatica/projects/sam-body4d/scripts/hf_space_logs.py run --timeout 10 --lines 30
   ```

   **Build logs:**
   ```bash
   python3 /Users/thomascummins/dev/chromatica/projects/sam-body4d/scripts/hf_space_logs.py build --timeout 10 --lines 30
   ```

3. Present a clean summary:
   - **Stage**: (BUILDING, RUNNING, ERROR, STOPPED, etc.)
   - **Hardware**: current GPU/CPU
   - **Storage**: persistent storage tier
   - **Dev Mode**: enabled/disabled
   - **Deployed SHA vs Repo SHA**: whether the running container matches the latest push
   - **URL**: https://troutmoose-sam-body4d.hf.space
   - **Run logs**: last few meaningful lines (skip empty/boilerplate)
   - **Build status**: whether the last build succeeded, plus any errors
   - Any warnings (e.g. SHA mismatch means a rebuild hasn't triggered yet)
