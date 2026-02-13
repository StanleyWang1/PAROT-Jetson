#!/usr/bin/env python3
"""
realtime_plot.py

Create scrolling "real-time style" plot videos from CoinFT CSV.

Input CSV columns expected:
  t_rel, Fx, Fy, Fz, Mx, My, Mz
(also works if only t_wall exists; t_rel is preferred)

Outputs:
  <prefix>_force.mp4
  <prefix>_moment.mp4

Usage:
  python3 realtime_plot.py coinft_ft_YYYYMMDD_HHMMSS.csv

Optional args:
  --window 5.0        # seconds shown in the moving window
  --fps 30            # output video FPS
  --speed 1.0         # playback speed (2.0 = twice as fast, 0.5 = half speed)
  --prefix outname    # output file prefix
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


def _get_time_axis(df: pd.DataFrame) -> np.ndarray:
    if "t_rel" in df.columns:
        t = df["t_rel"].astype(float).to_numpy()
    elif "t_sec" in df.columns:
        t = df["t_sec"].astype(float).to_numpy()
    elif "t_wall" in df.columns:
        # fall back to wall clock; normalize to start at 0
        t = df["t_wall"].astype(float).to_numpy()
        t = t - t[0]
    else:
        raise ValueError("CSV must contain one of: t_rel, t_sec, or t_wall")
    return t


def make_scrolling_video(
    t: np.ndarray,
    y_series: dict,
    title: str,
    ylabel: str,
    out_path: Path,
    window_s: float,
    fps: int,
    speed: float,
):
    """
    Render a scrolling plot video.

    t: time array in seconds (monotonic)
    y_series: dict name -> array (same length as t)
    window_s: width of visible time window
    fps: output frames per second
    speed: playback speed factor (1.0 = realtime)
    """
    # Basic prep
    t0, t1 = float(t[0]), float(t[-1])
    duration = max(0.0, t1 - t0)
    out_duration = duration / max(1e-9, speed)
    n_frames = max(1, int(np.ceil(out_duration * fps)))

    # Figure
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True)

    # Pre-create lines
    lines = {}
    for name in y_series.keys():
        (ln,) = ax.plot([], [], label=name)
        lines[name] = ln
    ax.legend(loc="upper right")

    # Fixed y-limits based on full data (nice and stable)
    all_y = np.concatenate([np.asarray(y, dtype=float) for y in y_series.values()])
    y_min = float(np.nanmin(all_y))
    y_max = float(np.nanmax(all_y))
    if np.isfinite(y_min) and np.isfinite(y_max):
        pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
        ax.set_ylim(y_min - pad, y_max + pad)

    writer = FFMpegWriter(fps=fps, metadata={"title": title})

    print(f"Writing: {out_path}  ({n_frames} frames @ {fps} fps, speed={speed}x)")
    with writer.saving(fig, str(out_path), dpi=150):
        for k in range(n_frames):
            # Current playback time in the source signal
            t_play = t0 + (k / fps) * speed

            # Visible window
            left = t_play - window_s
            right = t_play

            # Clamp for early part so we still show something
            if left < t0:
                left = t0
            if right < t0:
                right = t0

            # Index range in t for window
            i0 = int(np.searchsorted(t, left, side="left"))
            i1 = int(np.searchsorted(t, right, side="right"))
            i0 = max(0, min(i0, len(t)))
            i1 = max(0, min(i1, len(t)))

            # Update line data
            tw = t[i0:i1]
            for name, y in y_series.items():
                yw = np.asarray(y, dtype=float)[i0:i1]
                lines[name].set_data(tw, yw)

            # Update x-limits to scroll
            ax.set_xlim(left, left + window_s)

            writer.grab_frame()

    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to coinft_ft_*.csv")
    ap.add_argument("--window", type=float, default=5.0, help="Seconds shown in scrolling window")
    ap.add_argument("--fps", type=int, default=30, help="Output video FPS")
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed (2.0 = faster, 0.5 = slower)")
    ap.add_argument("--prefix", type=str, default=None, help="Output prefix (default: csv stem)")
    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    df = pd.read_csv(csv_path)

    t = _get_time_axis(df)

    # Require these columns
    required = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    prefix = args.prefix if args.prefix else csv_path.stem
    out_force = csv_path.parent / f"{prefix}_force.mp4"
    out_moment = csv_path.parent / f"{prefix}_moment.mp4"

    make_scrolling_video(
        t=t,
        y_series={
            "Fx": df["Fx"].astype(float).to_numpy(),
            "Fy": df["Fy"].astype(float).to_numpy(),
            "Fz": df["Fz"].astype(float).to_numpy(),
        },
        title="CoinFT Force (Scrolling)",
        ylabel="Force",
        out_path=out_force,
        window_s=args.window,
        fps=args.fps,
        speed=args.speed,
    )

    make_scrolling_video(
        t=t,
        y_series={
            "Mx": df["Mx"].astype(float).to_numpy(),
            "My": df["My"].astype(float).to_numpy(),
            "Mz": df["Mz"].astype(float).to_numpy(),
        },
        title="CoinFT Moment (Scrolling)",
        ylabel="Moment",
        out_path=out_moment,
        window_s=args.window,
        fps=args.fps,
        speed=args.speed,
    )

    print("Done.")
    print("Saved:", out_force)
    print("Saved:", out_moment)


if __name__ == "__main__":
    main()
