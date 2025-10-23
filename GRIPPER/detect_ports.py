#!/usr/bin/env python3
"""
detect_port.py — find the serial port for a ROBOTIS U2D2 on Linux (Ubuntu 24.04)

Usage:
  python3 detect_port.py          # prints the best-match device path only
  python3 detect_port.py -v       # verbose (shows scoring & metadata)
  python3 detect_port.py --all    # list all serial ports with metadata
  python3 detect_port.py --json   # machine-readable JSON of candidates

Requires:
  pip install pyserial
"""

import sys
import json
import argparse
from typing import List, Tuple
try:
    from serial.tools import list_ports
except Exception as e:
    print("ERROR: pyserial is required. Install with: python3 -m pip install pyserial", file=sys.stderr)
    raise

# Common VID:PID for CP210x bridges (U2D2 uses Silicon Labs CP210x)
CANDIDATE_IDS = {
    (0x10C4, 0xEA60),  # Silicon Labs CP210x USB to UART Bridge
    # Add alternates just in case a different bridge is used
    (0x0403, 0x6001),  # FTDI FT232 (unlikely for U2D2, but harmless)
}

USB_PREFIXES = ("/dev/ttyUSB", "/dev/ttyACM")  # typical USB serial device nodes on Linux

def score_port(p: "list_ports.ListPortInfo") -> int:
    """Heuristic score for how likely this port is a U2D2."""
    score = 0
    dev = (p.device or "").lower()
    desc = (p.description or "").lower()
    manu = (p.manufacturer or "").lower()
    prod = (getattr(p, "product", "") or "").lower()
    hwid = (p.hwid or "").lower()
    vid = getattr(p, "vid", None)
    pid = getattr(p, "pid", None)

    # Must look like a USB serial
    if dev.startswith(USB_PREFIXES):
        score += 10

    # Prefer Silicon Labs CP210x
    if "silicon labs" in manu or "cp210" in desc or "cp210" in prod or "cp210" in hwid:
        score += 40

    # Bonus for exact VID:PID matches
    if vid is not None and pid is not None and (vid, pid) in CANDIDATE_IDS:
        score += 60

    # Direct hints
    if "u2d2" in desc or "u2d2" in prod or "robotis" in manu or "robotis" in desc:
        score += 100

    # Slight bump if generic USB-Serial strings show up
    if "usb-to-serial" in desc or "usb serial" in desc:
        score += 5

    # Penalize built-in UARTs (not USB)
    if dev.startswith("/dev/ttyAMA") or dev.startswith("/dev/ttyS"):
        score -= 50

    return score

def gather_candidates() -> List[Tuple[int, dict]]:
    ports = list(list_ports.comports())
    candidates = []
    for p in ports:
        info = {
            "device": p.device,
            "description": p.description,
            "manufacturer": getattr(p, "manufacturer", None),
            "product": getattr(p, "product", None),
            "serial_number": getattr(p, "serial_number", None),
            "vid": getattr(p, "vid", None),
            "pid": getattr(p, "pid", None),
            "hwid": getattr(p, "hwid", None),
        }
        candidates.append((score_port(p), info))
    # Sort best-first
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates

def main():
    ap = argparse.ArgumentParser(description="Detect the serial port for ROBOTIS U2D2 on Ubuntu.")
    ap.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    ap.add_argument("--all", action="store_true", help="list all serial ports with scores")
    ap.add_argument("--json", action="store_true", help="print JSON of candidates")
    args = ap.parse_args()

    candidates = gather_candidates()

    if args.all or args.json or args.verbose:
        # Show full candidate table
        out = []
        for score, info in candidates:
            row = dict(score=score, **info)
            out.append(row)
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            if not out:
                print("No serial ports found.")
            else:
                for row in out:
                    print(f"[{row['score']:>3}] {row['device']}: {row['description']}")
                    print(f"      manufacturer={row['manufacturer']} product={row['product']} "
                          f"vid={row['vid']} pid={row['pid']} sn={row['serial_number']}")
        # Also print best pick at the end for convenience (if any)
        if candidates:
            best = candidates[0][1]["device"]
            if args.verbose:
                print(f"\nBest match: {best}")
        return 0

    # Quiet mode: print best device path only (for scripts)
    if not candidates:
        print("", end="")
        return 1  # not found

    best_score, best_info = candidates[0]
    device = best_info["device"]

    # If the best candidate looks very weak (e.g., negative score), fail cautiously
    if best_score < 0:
        print("", end="")
        return 2

    print(device, end="")  # no newline if you prefer; change to print(device) if desired
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
