#!/usr/bin/env python3
"""
Camera + CoinFT Visualizer (1 sensor)

- Camera preview: EXACT code block you provided (UNCHANGED)
- CoinFT serial read + ONNX inference + live plot
"""

import time
import struct
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
import serial
import cv2


# =========================
# CoinFT Configuration
# =========================
NUM_COINFTS  = 1

# UART Settings
PORT_NAME    = "/dev/ttyTHS1"
BAUD_RATE    = 1000000
READ_TIMEOUT = 0.1

# Paths (script is in coinFT/, configs in coinFT/hardware_configs/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(SCRIPT_DIR, "hardware_configs")

MODEL_FILE = "CFT24_MLP.onnx"
NORM_FILE  = "CFT24_norm.json"

# Tare / Filtering
INITIAL_SAMPLES = 500
IGNORED_SAMPLES = 10
WINDOW_SIZE     = 10

# Plotting
PLOT_HISTORY    = 5.0
PLOT_INTERVAL   = 40

# Packet constants (STX/ETX framing)
COINFT_CH     = 12
STX           = b"\x02"
ETX           = b"\x03"
PACKET_SIZE   = 26  # 1 STX + 1 header + 24 data + 1 ETX


# =========================
# CoinFT Helpers
# =========================
def load_norms(norm_filename):
    path = os.path.join(CONFIG_DIR, norm_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find norm file: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    return {
        "mu_x": np.array(data["mu_x"], dtype=np.float32),
        "sd_x": np.array(data["sd_x"], dtype=np.float32),
        "mu_y": np.array(data["mu_y"], dtype=np.float32),
        "sd_y": np.array(data["sd_y"], dtype=np.float32),
    }


def start_stream(ser):
    """Handshake to start streaming."""
    print("Resetting sensor...")
    ser.write(b"i")
    time.sleep(0.2)
    ser.reset_input_buffer()
    ser.write(b"s")
    time.sleep(0.05)


def read_packet(ser):
    """Read one CoinFT packet. Returns np.array shape (12,) or None."""
    # 1) Wait for STX
    while True:
        b = ser.read(1)
        if not b:
            return None  # timeout
        if b == STX:
            break

    # 2) Read remaining bytes: header + data + ETX (25 bytes)
    packet_data = ser.read(PACKET_SIZE - 1)
    if len(packet_data) != PACKET_SIZE - 1:
        return None

    # 3) Verify ETX
    if packet_data[-1:] != ETX:
        return None

    # 4) Extract 24 channel bytes (skip 1-byte header)
    channel_data = packet_data[1:25]

    # 5) Big-endian uint16s (matches coinft_interface.py)
    vals = struct.unpack(">" + "H" * COINFT_CH, channel_data)
    return np.array(vals, dtype=np.float64)


def tare_sensor(ser):
    print(f"Taring... ({INITIAL_SAMPLES} samples)")
    buf = []

    # Collect until we actually have enough valid packets
    while len(buf) < INITIAL_SAMPLES:
        pkt = read_packet(ser)
        if pkt is not None:
            buf.append(pkt)

    arr = np.array(buf)
    if len(arr) <= IGNORED_SAMPLES:
        raise RuntimeError("Not enough data for tare. Check connection.")

    offset = np.mean(arr[IGNORED_SAMPLES:], axis=0)
    print("Tare complete.")
    return offset


# =========================
# Main
# =========================
def main():
    # --- Load ONNX + norms
    model_path = os.path.join(CONFIG_DIR, MODEL_FILE)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    norms = load_norms(NORM_FILE)
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    print("ONNX providers:", session.get_providers())
    print("Using configs from:", CONFIG_DIR)

    # --- Setup Serial
    print(f"Opening CoinFT serial: {PORT_NAME} @ {BAUD_RATE}")
    ser = serial.Serial(PORT_NAME, BAUD_RATE, timeout=READ_TIMEOUT)
    start_stream(ser)
    offset = tare_sensor(ser)

    # --- Setup Plotting
    plt.ion()
    fig, axes = plt.subplots(2, 1, figsize=(7, 6))

    # Force plot
    ax_f = axes[0]
    lfx, = ax_f.plot([], [], label="Fx")
    lfy, = ax_f.plot([], [], label="Fy")
    lfz, = ax_f.plot([], [], label="Fz")
    ax_f.set_title("CoinFT - Force")
    ax_f.grid(True)
    ax_f.legend(loc="upper right")

    # Moment plot
    ax_m = axes[1]
    lmx, = ax_m.plot([], [], label="Mx")
    lmy, = ax_m.plot([], [], label="My")
    lmz, = ax_m.plot([], [], label="Mz")
    ax_m.set_title("CoinFT - Moment")
    ax_m.grid(True)
    ax_m.legend(loc="upper right")

    t_hist = []
    f_hist = [[], [], []]
    m_hist = [[], [], []]

    ma_queue = []

    print("Starting camera + CoinFT... (press q to quit)")

    # =========================
    # Camera code block (UNCHANGED)
    # =========================
    cap = cv2.VideoCapture(0)  # try 0 first

    if not cap.isOpened():
        print("Camera not opened")
        ser.write(b"i")
        ser.close()
        exit()

    start_time = time.time()
    packet_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame failed")
                break
            frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("camera", frame_small)

            # ---- CoinFT update (non-blocking due to serial timeout)
            pkt = read_packet(ser)
            if pkt is not None:
                now = time.time() - start_time

                # 1) Offset
                raw_zeroed = pkt - offset

                # 2) Normalize
                raw_norm = (raw_zeroed.astype(np.float32) - norms["mu_x"]) / norms["sd_x"]

                # 3) Inference
                pred_norm = session.run(None, {input_name: raw_norm.reshape(1, 12)})[0].flatten()

                # 4) Denormalize -> [Fx,Fy,Fz,Mx,My,Mz]
                ft_val = pred_norm * norms["sd_y"] + norms["mu_y"]

                # 5) Moving average
                ma_queue.append(ft_val)
                if len(ma_queue) > WINDOW_SIZE:
                    ma_queue.pop(0)
                ft_avg = np.mean(ma_queue, axis=0)

                # 6) Store
                t_hist.append(now)
                for j in range(3):
                    f_hist[j].append(float(ft_avg[j]))
                    m_hist[j].append(float(ft_avg[j + 3]))

                # Trim history
                while t_hist and (t_hist[-1] - t_hist[0] > PLOT_HISTORY):
                    t_hist.pop(0)
                    for j in range(3):
                        f_hist[j].pop(0)
                        m_hist[j].pop(0)

                # Plot update throttled
                packet_count += 1
                if packet_count % PLOT_INTERVAL == 0 and t_hist:
                    lfx.set_data(t_hist, f_hist[0])
                    lfy.set_data(t_hist, f_hist[1])
                    lfz.set_data(t_hist, f_hist[2])
                    lmx.set_data(t_hist, m_hist[0])
                    lmy.set_data(t_hist, m_hist[1])
                    lmz.set_data(t_hist, m_hist[2])

                    # autoscale Y only
                    ax_f.relim(); ax_f.autoscale_view(scalex=False, scaley=True)
                    ax_m.relim(); ax_m.autoscale_view(scalex=False, scaley=True)

                    plt.pause(0.001)

            # --- Quit (UNCHANGED)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            ser.write(b"i")
        except Exception:
            pass
        ser.close()
        print("Closed.")


if __name__ == "__main__":
    main()
