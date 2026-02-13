#!/usr/bin/env python3
"""
Camera (unchanged) + CoinFT FT-only logger -> CSV + save camera video.

- CoinFT logs: t_wall, t_rel, Fx,Fy,Fz,Mx,My,Mz
- CSV is saved NEXT TO THIS SCRIPT (not your terminal cwd)
- Video is saved NEXT TO THIS SCRIPT
- Ctrl+C cleanly stops and saves (flushes periodically too)
"""

import time
import struct
import json
import os
import csv
import threading

import numpy as np
import serial
import onnxruntime as ort
import cv2


# =========================
# CoinFT Config (1 sensor)
# =========================
PORT_NAME    = "/dev/ttyTHS1"
BAUD_RATE    = 1000000
READ_TIMEOUT = 0.1

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(SCRIPT_DIR, "..", "hardware_configs")

MODEL_PATH = os.path.join(CONFIG_DIR, "CFT24_MLP.onnx")
NORM_PATH  = os.path.join(CONFIG_DIR, "CFT24_norm.json")

INITIAL_SAMPLES = 500
IGNORED_SAMPLES = 10

COINFT_CH   = 12
STX         = b"\x02"
ETX         = b"\x03"
PACKET_SIZE = 26

# Save CSV next to this script (reliable location)
OUT_CSV = os.path.join(SCRIPT_DIR, f"coinft_ft_{time.strftime('%Y%m%d_%H%M%S')}.csv")

# Save video next to this script
OUT_VIDEO = os.path.join(SCRIPT_DIR, f"camera_{time.strftime('%Y%m%d_%H%M%S')}.mp4")

# Ensure data is written regularly (so Ctrl+C never loses everything)
FLUSH_EVERY = 50


# =========================
# CoinFT Helpers
# =========================
def start_stream(ser):
    ser.write(b"i")
    time.sleep(0.2)
    ser.reset_input_buffer()
    ser.write(b"s")
    time.sleep(0.05)


def read_packet(ser):
    while True:
        b = ser.read(1)
        if not b:
            return None
        if b == STX:
            break

    packet_data = ser.read(PACKET_SIZE - 1)
    if len(packet_data) != PACKET_SIZE - 1:
        return None
    if packet_data[-1:] != ETX:
        return None

    channel_data = packet_data[1:25]
    vals = struct.unpack(">" + "H" * COINFT_CH, channel_data)
    return np.array(vals, dtype=np.float64)


def tare_sensor(ser):
    print("[CoinFT] Taring...")
    buf = []
    while len(buf) < INITIAL_SAMPLES:
        pkt = read_packet(ser)
        if pkt is not None:
            buf.append(pkt)

    arr = np.array(buf)
    if len(arr) <= IGNORED_SAMPLES:
        raise RuntimeError("[CoinFT] Not enough data for tare. Check connection.")

    offset = np.mean(arr[IGNORED_SAMPLES:], axis=0)
    print("[CoinFT] Tare complete.")
    return offset


def coinft_logger(stop_event):
    ser = None
    try:
        # Load norms
        with open(NORM_PATH, "r") as f:
            data = json.load(f)

        mu_x = np.array(data["mu_x"], dtype=np.float32)
        sd_x = np.array(data["sd_x"], dtype=np.float32)
        mu_y = np.array(data["mu_y"], dtype=np.float32)
        sd_y = np.array(data["sd_y"], dtype=np.float32)

        # Load model
        session = ort.InferenceSession(MODEL_PATH)
        input_name = session.get_inputs()[0].name
        print("[CoinFT] ONNX providers:", session.get_providers())

        # Serial
        print(f"[CoinFT] Opening {PORT_NAME} @ {BAUD_RATE}")
        ser = serial.Serial(PORT_NAME, BAUD_RATE, timeout=READ_TIMEOUT)
        start_stream(ser)
        offset = tare_sensor(ser)

        t0 = time.time()
        print("[CoinFT] Saving CSV to:", OUT_CSV)

        with open(OUT_CSV, "w", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["t_wall", "t_rel", "Fx", "Fy", "Fz", "Mx", "My", "Mz"])
            fcsv.flush()

            rows = 0
            while not stop_event.is_set():
                pkt = read_packet(ser)
                if pkt is None:
                    continue

                t_wall = time.time()
                t_rel  = t_wall - t0

                raw_zeroed = pkt - offset
                raw_norm = (raw_zeroed.astype(np.float32) - mu_x) / sd_x
                pred_norm = session.run(None, {input_name: raw_norm.reshape(1, 12)})[0].flatten()
                ft = pred_norm * sd_y + mu_y  # Fx..Mz

                writer.writerow([f"{t_wall:.6f}", f"{t_rel:.6f}"] +
                                [f"{v:.6f}" for v in ft])

                rows += 1
                if rows % FLUSH_EVERY == 0:
                    fcsv.flush()

    except Exception as e:
        print("[CoinFT] Error:", e)

    finally:
        if ser is not None:
            try:
                ser.write(b"i")
            except Exception:
                pass
            ser.close()
        print("[CoinFT] Logger stopped.")


# =========================
# Main (camera code UNCHANGED, plus VideoWriter)
# =========================
def main():
    stop_event = threading.Event()

    # IMPORTANT: not daemon -> allows clean join on Ctrl+C
    logger_thread = threading.Thread(target=coinft_logger, args=(stop_event,))
    logger_thread.start()

    video_writer = None

    try:
        # =========================
        # CAMERA CODE (UNCHANGED)
        # =========================
        import cv2
        import time

        # Open camera
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Camera not opened")
            stop_event.set()
            logger_thread.join()
            return

        # 🔧 Reduce camera resolution at source (VERY important)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # 🔧 Reduce internal buffering (helps latency)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("camera", 800, 450)

        # ---- VideoWriter (added) ----
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(OUT_VIDEO, fourcc, 30.0, (640, 360))
        if video_writer.isOpened():
            print("Saving video to:", OUT_VIDEO)
        else:
            print("WARNING: VideoWriter not opened; video will not be saved.")
            video_writer = None
        # -----------------------------

        print("Press q to quit")

        prev_time = time.time()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame failed")
                break

            # OPTIONAL: remove resize entirely since we set camera to 640x360
            # frame = cv2.resize(frame, (640, 360))

            cv2.imshow("camera", frame)

            # ---- Write video frame (added) ----
            if video_writer is not None:
                if frame.shape[1] != 640 or frame.shape[0] != 360:
                    frame_to_write = cv2.resize(frame, (640, 360))
                else:
                    frame_to_write = frame
                video_writer.write(frame_to_write)
            # ----------------------------------

            # FPS counter
            frame_count += 1
            if frame_count % 30 == 0:
                now = time.time()
                fps = 30 / (now - prev_time)
                print(f"FPS: {fps:.1f}")
                prev_time = now

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        # =========================

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Stopping...")

    finally:
        # Stop CoinFT logger
        stop_event.set()
        logger_thread.join()

        # Close video writer
        if video_writer is not None:
            video_writer.release()
            print("Saved video:", OUT_VIDEO)

        print("Saved CSV:", OUT_CSV)


if __name__ == "__main__":
    main()
