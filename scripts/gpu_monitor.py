"""
Live GPU monitoring script for SAM-Body4D inference profiling.

Run via SSH on the HuggingFace Space while triggering jobs from the web UI.
No app restart needed — runs as a separate process.

Usage:
    python3 scripts/gpu_monitor.py              # default 0.5s interval
    python3 scripts/gpu_monitor.py --interval 1  # 1s interval
    python3 scripts/gpu_monitor.py --csv         # CSV output for analysis
"""

import argparse
import csv
import io
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class GpuSample:
    timestamp: float
    gpu_util: int       # SM utilization %
    mem_used_mb: int    # framebuffer memory used (MB)
    mem_total_mb: int   # framebuffer memory total (MB)
    mem_util: int       # memory controller utilization %
    temperature: int    # GPU temp (C)
    power_w: float      # power draw (W)


@dataclass
class PhaseSummary:
    name: str
    start_time: float
    end_time: float
    samples: list = field(default_factory=list)

    @property
    def duration(self):
        return self.end_time - self.start_time

    @property
    def avg_gpu_util(self):
        if not self.samples:
            return 0
        return sum(s.gpu_util for s in self.samples) / len(self.samples)

    @property
    def peak_mem_mb(self):
        if not self.samples:
            return 0
        return max(s.mem_used_mb for s in self.samples)

    @property
    def avg_mem_mb(self):
        if not self.samples:
            return 0
        return sum(s.mem_used_mb for s in self.samples) / len(self.samples)

    @property
    def peak_gpu_util(self):
        if not self.samples:
            return 0
        return max(s.gpu_util for s in self.samples)

    @property
    def avg_power_w(self):
        if not self.samples:
            return 0
        return sum(s.power_w for s in self.samples) / len(self.samples)


def query_gpu():
    """Query nvidia-smi for current GPU stats. Returns GpuSample or None."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,utilization.memory,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        return GpuSample(
            timestamp=time.time(),
            gpu_util=int(parts[0]),
            mem_used_mb=int(parts[1]),
            mem_total_mb=int(parts[2]),
            mem_util=int(parts[3]),
            temperature=int(parts[4]),
            power_w=float(parts[5]),
        )
    except Exception as e:
        print(f"[error] nvidia-smi query failed: {e}", file=sys.stderr)
        return None


IDLE_THRESHOLD = 5  # GPU util % below which we consider "idle"
ACTIVE_THRESHOLD = 15  # GPU util % above which we consider "active"


def detect_phases(samples):
    """Detect active GPU phases from a list of samples."""
    if not samples:
        return []

    phases = []
    in_phase = False
    phase_start = 0
    phase_samples = []
    idle_count = 0
    phase_idx = 1

    for s in samples:
        if not in_phase:
            if s.gpu_util >= ACTIVE_THRESHOLD:
                in_phase = True
                phase_start = s.timestamp
                phase_samples = [s]
                idle_count = 0
        else:
            phase_samples.append(s)
            if s.gpu_util < IDLE_THRESHOLD:
                idle_count += 1
                # End phase after 3 consecutive idle samples
                if idle_count >= 3:
                    # Trim trailing idle samples
                    active_samples = phase_samples[: len(phase_samples) - idle_count]
                    if active_samples:
                        phases.append(PhaseSummary(
                            name=f"Phase {phase_idx}",
                            start_time=phase_start,
                            end_time=active_samples[-1].timestamp,
                            samples=active_samples,
                        ))
                        phase_idx += 1
                    in_phase = False
                    phase_samples = []
            else:
                idle_count = 0

    # Close any open phase
    if in_phase and phase_samples:
        phases.append(PhaseSummary(
            name=f"Phase {phase_idx}",
            start_time=phase_start,
            end_time=phase_samples[-1].timestamp,
            samples=phase_samples,
        ))

    return phases


def fmt_duration(sec):
    if sec < 60:
        return f"{sec:.1f}s"
    m = int(sec // 60)
    s = sec % 60
    return f"{m}m {s:.1f}s"


def fmt_mem(mb):
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb} MB"


def print_summary(all_samples, t_start):
    """Print a summary of the monitoring session."""
    total_time = time.time() - t_start
    phases = detect_phases(all_samples)

    print("\n")
    print("=" * 70)
    print("  GPU MONITORING SUMMARY")
    print("=" * 70)
    print(f"  Total monitoring time: {fmt_duration(total_time)}")
    print(f"  Samples collected:     {len(all_samples)}")

    if all_samples:
        print(f"  Overall peak VRAM:     {fmt_mem(max(s.mem_used_mb for s in all_samples))}"
              f" / {fmt_mem(all_samples[0].mem_total_mb)}")
        print(f"  Baseline VRAM (idle):  {fmt_mem(all_samples[0].mem_used_mb)}")

    if not phases:
        print("\n  No active GPU phases detected.")
        print("  (Trigger a job from the web UI while monitoring.)")
    else:
        print(f"\n  Detected {len(phases)} active phase(s):")
        print("-" * 70)
        print(f"  {'Phase':<12} {'Duration':>10} {'GPU Avg':>10} {'GPU Peak':>10} "
              f"{'VRAM Peak':>12} {'VRAM Avg':>12} {'Power Avg':>10}")
        print("-" * 70)
        for p in phases:
            print(f"  {p.name:<12} {fmt_duration(p.duration):>10} "
                  f"{p.avg_gpu_util:>9.0f}% {p.peak_gpu_util:>9d}% "
                  f"{fmt_mem(p.peak_mem_mb):>12} {fmt_mem(int(p.avg_mem_mb)):>12} "
                  f"{p.avg_power_w:>8.0f} W")
        print("-" * 70)

        # VRAM delta for each phase
        if all_samples:
            baseline = all_samples[0].mem_used_mb
            print(f"\n  VRAM above baseline per phase:")
            for p in phases:
                delta = p.peak_mem_mb - baseline
                print(f"    {p.name}: +{fmt_mem(delta)}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Live GPU monitoring for SAM-Body4D")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Polling interval in seconds (default: 0.5)")
    parser.add_argument("--csv", action="store_true",
                        help="Output CSV format instead of live display")
    args = parser.parse_args()

    # Test nvidia-smi availability
    sample = query_gpu()
    if sample is None:
        print("ERROR: nvidia-smi not available. Run this on the GPU machine.", file=sys.stderr)
        sys.exit(1)

    all_samples = []
    t_start = time.time()
    running = True

    def handle_sigint(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_sigint)

    if args.csv:
        writer = csv.writer(sys.stdout)
        writer.writerow(["elapsed_s", "gpu_util_%", "mem_used_mb", "mem_total_mb",
                         "mem_util_%", "temp_c", "power_w"])
    else:
        print(f"GPU: {fmt_mem(sample.mem_total_mb)} total VRAM")
        print(f"Polling every {args.interval}s. Press Ctrl+C to stop and see summary.\n")
        print(f"{'Time':>8}  {'GPU':>5}  {'VRAM Used':>12}  {'VRAM %':>7}  {'Temp':>5}  {'Power':>7}  {'Status'}")
        print("-" * 70)

    while running:
        s = query_gpu()
        if s is None:
            time.sleep(args.interval)
            continue

        all_samples.append(s)
        elapsed = s.timestamp - t_start

        if args.csv:
            writer = csv.writer(sys.stdout)
            writer.writerow([f"{elapsed:.1f}", s.gpu_util, s.mem_used_mb,
                             s.mem_total_mb, s.mem_util, s.temperature, s.power_w])
            sys.stdout.flush()
        else:
            mem_pct = (s.mem_used_mb / s.mem_total_mb * 100) if s.mem_total_mb > 0 else 0
            status = "ACTIVE" if s.gpu_util >= ACTIVE_THRESHOLD else "idle"
            bar = "#" * (s.gpu_util // 5) + "." * (20 - s.gpu_util // 5)
            print(f"{elapsed:>7.1f}s  {s.gpu_util:>4d}%  {fmt_mem(s.mem_used_mb):>12}  "
                  f"{mem_pct:>5.1f}%  {s.temperature:>4d}C  {s.power_w:>5.0f} W  "
                  f"[{bar}] {status}")

        time.sleep(args.interval)

    if not args.csv:
        print_summary(all_samples, t_start)


if __name__ == "__main__":
    main()
