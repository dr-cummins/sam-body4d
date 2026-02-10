---
name: hf-deploy
description: Push the hf-space repo to HuggingFace and verify the deployment succeeds.
allowed-tools: Bash(python3 *), Bash(git -C *), Bash(sleep *)
---

# HF Space Deploy

Push the current state of the `hf-space/` repo to HuggingFace and verify the deployment succeeds.

**IMPORTANT: One command per Bash call. Never pipe, chain, or redirect commands.**

## Pre-flight

1. Run these two git commands **in parallel** (two separate Bash calls):
   ```bash
   git -C /Users/thomascummins/dev/chromatica/projects/hf-space status
   ```
   ```bash
   git -C /Users/thomascummins/dev/chromatica/projects/hf-space log --oneline -5
   ```

2. Check what will be pushed:
   ```bash
   git -C /Users/thomascummins/dev/chromatica/projects/hf-space log origin/main..HEAD --oneline
   ```
   If there's nothing to push, inform the user and stop. If there are uncommitted changes, warn the user and stop.

## Deploy

3. Push to the HF remote:
   ```bash
   git -C /Users/thomascummins/dev/chromatica/projects/hf-space push origin main
   ```

## Monitor

4. Wait 5 seconds for the build to trigger:
   ```bash
   sleep 5
   ```

5. Poll the space status using the helper script with `--poll` to auto-retry:
   ```bash
   python3 /Users/thomascummins/dev/chromatica/projects/sam-body4d/scripts/hf_space_info.py --poll 10 --interval 20
   ```
   This polls up to 10 times (max ~3 minutes), printing status each time. It stops early if the deployed SHA matches the repo SHA or if an error occurs.

6. Once done polling, fetch both log streams **in parallel** (two separate Bash calls):

   **Build logs:**
   ```bash
   python3 /Users/thomascummins/dev/chromatica/projects/sam-body4d/scripts/hf_space_logs.py build --timeout 10 --lines 50
   ```

   **Run logs:**
   ```bash
   python3 /Users/thomascummins/dev/chromatica/projects/sam-body4d/scripts/hf_space_logs.py run --timeout 10 --lines 30
   ```

## Report

7. Present a deployment summary:
   - **Pushed commits**: list of commits that were pushed
   - **Build result**: SUCCESS or FAILURE
   - **Runtime stage**: RUNNING, BUILDING, ERROR, etc.
   - **Deployed SHA**: confirm it matches what was pushed (or note if still building)
   - **Run logs**: last few meaningful lines from the container
   - **URL**: https://troutmoose-sam-body4d.hf.space
   - Any errors or warnings
