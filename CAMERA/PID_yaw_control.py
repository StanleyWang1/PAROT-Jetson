#!/usr/bin/env python3
import time
import math
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import depthai as dai
import torch
from mobile_sam import sam_model_registry, SamPredictor

# =========================================================
# CONFIG
# =========================================================
# Models
MODEL_PATH = sorted(Path("MODELS/yolo/runs/pose").glob("train*/weights/best.pt"))[-1]
IMG_SIZE = 384
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# How often to run SAM (every N frames). N=1 means every frame.
SAM_EVERY = 1

# Detection / association
CONF_THRES = 0.80
CONF_THRES_LOW = 0.60
EMA_ALPHA = 0.40
ASSOC_MAX_DIST = 80
ASSOC_MIN_IOU = 0.05
COAST_MAX_FRAMES = 2  # tolerate 1-2 missed frames

# Control (yaw-only, slow & stable)
V_FWD = 0.10                      # m/s forward crawl
V_CREEP = 0.05                    # m/s when close/misaligned
YAW_RATE_MAX = math.radians(15)   # rad/s cap (~15 deg/s)
YAW_SLEW_MAX = math.radians(60)   # rad/s^2 limit on change per second

# 1-D PD on normalized horizontal error e in [-1,1]
Kp, Ki, Kd = 0.5, 0.0, 0.5
ERR_EMA_ALPHA = 0.35   # error low-pass
ERR_DEADBAND   = 0.02  # ignore tiny errors (<2% of half-width)

# Stop / slow thresholds based on apparent width proxy (pixels) — tune for your camera
SLOW_PX = 180
STOP_PX = 220

# Target robustness
TX_EMA_ALPHA   = 0.45  # EMA for target_x smoothing
TX_JUMP_FRAC   = 0.15  # ignore jumps > 15% of image width
TX_COAST_FRAMES = 2    # coast this many frames on missing target

# UI
WINDOW_NAME = "Tree Centering (Yaw-only IBVS)"
DRAW_STEP_PX = 10

# Commands (implement send_velocity_yawrate() for your autopilot and set True)
SEND_COMMANDS = False

# =========================================================
# LOAD MODELS
# =========================================================
print(f"[info] Loading YOLO: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))

sam_ckpt = None
for p in [
    Path("MODELS") / "mobile_sam" / "weights" / "mobile_sam.pt",
    Path.home() / "treepoint" / "MobileSAM" / "weights" / "mobile_sam.pt",
    Path("weights") / "mobile_sam.pt",
    Path("MobileSAM") / "weights" / "mobile_sam.pt",
]:
    if p.expanduser().exists():
        sam_ckpt = str(p.expanduser())
        break
if not sam_ckpt:
    raise FileNotFoundError("MobileSAM checkpoint not found.")
print(f"[info] Loading MobileSAM: {sam_ckpt}")
predictor = SamPredictor(sam_model_registry["vit_t"](checkpoint=sam_ckpt))
predictor.model.to(DEVICE)
predictor.model.eval()

# =========================================================
# UTILS / STATE
# =========================================================
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)

def ema(prev, curr, alpha):
    return alpha * np.array(curr, dtype=float) + (1 - alpha) * np.array(prev, dtype=float)

class StableState:
    def __init__(self):
        self.has = False
        self.box = None
        self.kpt = None
        self.conf = 0.0
        self.miss = 0

STATE = StableState()

def select_best(results):
    if not results or not getattr(results[0], "boxes", None) or len(results[0].boxes) == 0:
        return None, None, None
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    kpts  = getattr(results[0], "keypoints", None)
    idx = int(np.argmax(confs))
    box = boxes[idx]
    kp = None
    if kpts is not None and getattr(kpts, "xy", None) is not None:
        kxy = kpts.xy.cpu().numpy()
        if kxy.shape[0] > idx and kxy.shape[1] > 0:
            kp = kxy[idx][0]
    if kp is None:
        kp = np.array([(box[0]+box[2])/2.0, (box[1]+box[3])/2.0], dtype=np.float32)
    return box, kp, float(confs[idx])

def should_accept(curr_box, curr_kp, curr_conf, state: StableState):
    if curr_box is None or curr_kp is None or curr_conf is None:
        return False
    if curr_conf >= CONF_THRES:
        return True
    if curr_conf >= CONF_THRES_LOW and state.has:
        dist = np.linalg.norm(np.array(curr_kp) - state.kpt)
        iou  = iou_xyxy(curr_box, state.box)
        if dist <= ASSOC_MAX_DIST or iou >= ASSOC_MIN_IOU:
            return True
    return False

def update_state(curr_box, curr_kp, curr_conf, state: StableState):
    if curr_box is not None and curr_kp is not None:
        if state.has:
            state.box = ema(state.box, curr_box, EMA_ALPHA)
            state.kpt = ema(state.kpt, curr_kp, EMA_ALPHA)
        else:
            state.box = np.array(curr_box, dtype=float)
            state.kpt = np.array(curr_kp, dtype=float)
        state.conf = curr_conf
        state.has = True
        state.miss = 0
    else:
        if state.has and state.miss < COAST_MAX_FRAMES:
            state.miss += 1
        else:
            state.has = False
            state.box = None
            state.kpt = None
            state.conf = 0.0
            state.miss = 0

# =========================================================
# IMAGE OPS (mask → centerline)
# =========================================================
def clean_mask(mask, k=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m > 0
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label)

def row_edges(mask_bool, min_width=5):
    H, _ = mask_bool.shape
    ys, xL, xR = [], [], []
    for y in range(H):
        xs = np.flatnonzero(mask_bool[y])
        if xs.size >= min_width:
            ys.append(y); xL.append(int(xs[0])); xR.append(int(xs[-1]))
    return np.array(ys, int), np.array(xL, int), np.array(xR, int)

def fit_centerline(ys, xC):
    # x = a*y + b
    A = np.vstack([ys.astype(np.float32), np.ones_like(ys, np.float32)]).T
    a, b = np.linalg.lstsq(A, xC.astype(np.float32), rcond=None)[0]
    return float(a), float(b)

def measure_centerline_only(mask, min_width_px=5):
    mb = clean_mask(mask)
    ys, xL, xR = row_edges(mb, min_width=min_width_px)
    if ys.size == 0:
        return None
    xC = 0.5 * (xL + xR)
    a, b = fit_centerline(ys, xC)
    return ys, xL, xR, xC, a, b, mb

def median_width_px(ys, xL, xR):
    if ys is None or len(ys) == 0:
        return None
    w = np.asarray(xR) - np.asarray(xL)
    return float(np.median(w)) if w.size > 0 else None

def inflate_box(box_xyxy, scale, W, H):
    """Inflate a [x1,y1,x2,y2] box by `scale` (e.g., 0.10 adds 10% margin), clipped to image."""
    x1, y1, x2, y2 = map(float, box_xyxy)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1) * (1.0 + scale), (y2 - y1) * (1.0 + scale)
    nx1, ny1 = max(0.0, cx - w/2.0), max(0.0, cy - h/2.0)
    nx2, ny2 = min(float(W - 1), cx + w/2.0), min(float(H - 1), cy + h/2.0)
    return np.array([nx1, ny1, nx2, ny2], dtype=np.float32)

# =========================================================
# TARGET FILTERS & CONTROLLER
# =========================================================
class TargetXFilter:
    def __init__(self, alpha=TX_EMA_ALPHA, jump_frac=TX_JUMP_FRAC, coast_max=TX_COAST_FRAMES):
        self.alpha = float(alpha)
        self.jump_frac = float(jump_frac)
        self.coast_max = int(coast_max)
        self.has = False
        self.x = None
        self.coast = 0

    def update(self, meas_x, W):
        if meas_x is None:
            if self.has and self.coast < self.coast_max:
                self.coast += 1
                return self.x
            self.has = False
            self.x = None
            self.coast = 0
            return None

        meas_x = float(meas_x)
        if not self.has:
            self.x = meas_x
            self.has = True
            self.coast = 0
            return self.x

        # outlier rejection
        jump_thresh = self.jump_frac * float(W)
        if abs(meas_x - self.x) > jump_thresh:
            self.coast = min(self.coast + 1, self.coast_max)
            return self.x

        # EMA
        self.x = self.alpha * meas_x + (1.0 - self.alpha) * self.x
        self.coast = 0
        return self.x

class RateLimiter:
    def __init__(self, max_delta_per_s):
        self.max_step = float(max_delta_per_s)
        self.prev = 0.0
        self.init = True
    def step(self, u_des, dt):
        if self.init:
            self.prev = float(u_des); self.init = False
            return self.prev
        max_step = self.max_step * max(dt, 1e-3)
        du = float(u_des) - self.prev
        if   du >  max_step: self.prev += max_step
        elif du < -max_step: self.prev -= max_step
        else:                self.prev  = float(u_des)
        return self.prev

class PID1D:
    def __init__(self, Kp, Ki, Kd, out_limit):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.out_limit = float(out_limit)
        self.i = 0.0
        self.prev = 0.0
        self.first = True
    def reset(self):
        self.i = 0.0; self.prev = 0.0; self.first = True
    def update(self, err, dt):
        if self.first:
            de = 0.0; self.first = False
        else:
            de = (err - self.prev) / max(dt, 1e-3)
        self.prev = err
        self.i += err * dt
        if self.Ki > 0:
            i_max = self.out_limit / self.Ki
            self.i = max(-i_max, min(i_max, self.i))
        u = self.Kp*err + self.Ki*self.i + self.Kd*de
        return max(-self.out_limit, min(self.out_limit, u))

pid_yaw   = PID1D(Kp, Ki, Kd, out_limit=YAW_RATE_MAX)
yaw_slew  = RateLimiter(max_delta_per_s=YAW_SLEW_MAX)
_err_filt = 0.0
TXF       = TargetXFilter()

def yaw_only_step(target_x, frame_width, dt, width_px=None):
    """Compute vx, vy, vz, yaw_rate (rad/s) from target_x (px)."""
    global _err_filt
    W = float(frame_width)
    x_mid = 0.5 * W
    err_px = float(target_x - x_mid)

    # normalize to [-1,1] via half-width
    err_norm = err_px / (0.5 * W)
    _err_filt = ERR_EMA_ALPHA * err_norm + (1 - ERR_EMA_ALPHA) * _err_filt
    e = 0.0 if abs(_err_filt) < ERR_DEADBAND else _err_filt

    # PD -> desired yaw rate
    yaw_rate_des = pid_yaw.update(e, dt)

    # forward speed schedule
    vx = V_FWD
    if abs(e) > 0.5: vx *= 0.5
    if abs(e) > 0.8: vx *= 0.3

    if width_px is not None:
        if width_px >= STOP_PX:
            vx = 0.0
        elif width_px >= SLOW_PX:
            s = (width_px - SLOW_PX) / max(1.0, (STOP_PX - SLOW_PX))
            s = max(0.0, min(1.0, s))
            vx = (1 - s) * vx + s * V_CREEP

    vy = 0.0
    vz = 0.0
    return vx, vy, vz, yaw_rate_des, err_px

def decay_to_zero(val, dt, tau=0.35):
    a = math.exp(-dt / max(tau, 1e-3))
    return val * a

# =========================================================
# DRAWING
# =========================================================
def draw_state(im, state: StableState):
    if not state.has:
        cv2.putText(im, "NO TREE FOUND", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
        return
    x1, y1, x2, y2 = map(int, np.round(state.box))
    cx, cy = map(int, np.round(state.kpt))
    color = (0, 255, 0) if state.miss == 0 else (0, 180, 255)
    cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
    cv2.circle(im, (cx, cy), 6, (0, 0, 255), -1)
    lbl = f"Conf: {state.conf:.2f}"
    if state.miss > 0:
        lbl += f"  (coast {state.miss}/{COAST_MAX_FRAMES})"
    cv2.putText(im, lbl, (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

def draw_centerline_only(overlay, a, b):
    """Draw ONLY the red dashed centerline; no shading or edge ticks."""
    H, W = overlay.shape[:2]
    for y in range(0, H, DRAW_STEP_PX):
        xi = int(a * y + b)
        if 0 <= xi < W:
            cv2.line(overlay, (xi-2, y), (xi+2, y), (0, 0, 255), 2)

def draw_yaw_command_arrow(im, yaw_rate_cmd, yaw_rate_max=YAW_RATE_MAX):
    """
    Draw a big arrow showing the COMMANDED yaw (left/right) and magnitude.
    Arrow length scales with |yaw_rate_cmd| / yaw_rate_max.
    """
    H, W = im.shape[:2]
    center = (W // 2, H - 60)
    base_len = 160  # px for full-scale yaw
    scale = float(min(1.0, max(0.0, abs(yaw_rate_cmd) / max(1e-6, yaw_rate_max))))
    length = int(base_len * scale)
    direction = -1 if yaw_rate_cmd < 0 else 1  # left if negative

    start = (center[0] - direction * length, center[1])
    end   = (center[0] + direction * length, center[1])

    # baseline
    cv2.line(im, (center[0] - base_len, center[1]), (center[0] + base_len, center[1]),
             (90, 90, 90), 2, cv2.LINE_AA)
    # arrow
    cv2.arrowedLine(im, start, end, (0, 255, 255), 6, cv2.LINE_AA, tipLength=0.15)

    # label
    txt = f"Yaw cmd: {math.degrees(yaw_rate_cmd):+.1f} deg/s"
    cv2.putText(im, txt, (center[0] - 140, center[1] - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    # center crosshair and vertical midline for reference
    x_mid = W // 2; y_mid = H // 2
    cv2.drawMarker(im, (x_mid, y_mid), (255, 255, 255), cv2.MARKER_CROSS, 24, 1)
    cv2.line(im, (x_mid, 0), (x_mid, H - 1), (200, 200, 200), 1, cv2.LINE_AA)

# =========================================================
# COMMAND OUT (stub)
# =========================================================
def send_velocity_yawrate(vx, vy, vz, yaw_rate):
    """
    Implement for your autopilot (PX4 Offboard ROS Twist, MAVSDK, etc.).
    Keep sending at ~20–50 Hz if you enable SEND_COMMANDS.
    """
    pass

# =========================================================
# MAIN
# =========================================================
def main():
    # RGB-only OAK-D pipeline (640x480 preview)
    RGB_RES = (640, 480)
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setFps(15)
    cam.setInterleaved(False)
    cam.setPreviewSize(*RGB_RES)
    cam.setPreviewKeepAspectRatio(True)
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam.preview.link(xout_rgb.input)

    with dai.Device(pipeline) as dev:
        q_rgb = dev.getOutputQueue("rgb", maxSize=4, blocking=False)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, RGB_RES[0], RGB_RES[1])

        latest_rgb = None
        last_t = time.time()
        fps = 0.0
        yaw_cmd_prev = 0.0

        # SAM scheduling + cache
        frame_id = 0
        last_centerline = None  # (a, b, width_px)

        while True:
            # get newest frame
            while True:
                pkt = q_rgb.tryGet()
                if pkt is None:
                    break
                latest_rgb = pkt.getCvFrame()

            if latest_rgb is None:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            now = time.time()
            dt = max(1e-3, now - last_t)
            last_t = now

            rgb_bgr = latest_rgb
            rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            overlay = rgb_bgr.copy()
            H, W = overlay.shape[:2]

            # YOLO detection
            yolo_res = model.predict(source=rgb_bgr, imgsz=IMG_SIZE, conf=CONF_THRES,
                                     verbose=False, device=DEVICE)
            box, kp, conf = select_best(yolo_res)
            if box is not None and kp is not None and should_accept(box, kp, conf, STATE):
                update_state(box, kp, conf, STATE)
            else:
                update_state(None, None, None, STATE)

            # SAM scheduling (run every SAM_EVERY frames)
            raw_target_x = None
            width_px = None
            do_sam = (frame_id % max(1, int(SAM_EVERY)) == 0)
            frame_id += 1

            if STATE.has and do_sam:
                predictor.set_image(rgb_rgb)

                # Positive point at stabilized detection center
                pts = np.asarray([STATE.kpt], dtype=np.float32)
                lbl = np.asarray([1], dtype=np.int32)  # 1 = positive

                # Stabilized YOLO box (slightly inflated for margin)
                inf_box = inflate_box(STATE.box, scale=0.10, W=W, H=H)  # +10% margin
                box_xyxy = inf_box[None, :]  # shape (1,4), float32

                # SAM with point + box prompts
                masks, scores, _ = predictor.predict(
                    point_coords=pts,
                    point_labels=lbl,
                    box=box_xyxy,
                    multimask_output=False
                )
                mask = masks[int(np.argmax(scores))].astype(np.uint8)

                res = measure_centerline_only(mask, min_width_px=5)
                if res is not None:
                    ys, xL, xR, xC, a, b, mb = res
                    # ONLY draw red centerline (no shading or edge ticks)
                    draw_centerline_only(overlay, a, b)
                    # cache
                    last_centerline = (a, b, median_width_px(ys, xL, xR))
                    raw_target_x = float(a * (H * 0.5) + b)
                    width_px = last_centerline[2]
                else:
                    # fall back: stabilized detection center + bbox width
                    raw_target_x = float(STATE.kpt[0])
                    x1, y1, x2, y2 = STATE.box
                    width_px = float(max(0.0, x2 - x1))

            elif STATE.has and last_centerline is not None:
                # Reuse last centerline this frame; draw only the red line
                a, b, w_med = last_centerline
                draw_centerline_only(overlay, a, b)
                raw_target_x = float(a * (H * 0.5) + b)
                # proxy width by last width or current bbox width if available
                if w_med is None:
                    x1, y1, x2, y2 = STATE.box
                    width_px = float(max(0.0, x2 - x1))
                else:
                    width_px = w_med

            elif STATE.has:
                # No previous centerline yet; use bbox center as coarse target
                raw_target_x = float(STATE.kpt[0])
                x1, y1, x2, y2 = STATE.box
                width_px = float(max(0.0, x2 - x1))

            # Filter target & compute commands
            stable_target_x = TXF.update(raw_target_x, W)

            if stable_target_x is not None:
                vx, vy, vz, yaw_rate_des, err_px = yaw_only_step(stable_target_x, W, dt, width_px)
                # Slew-limit yaw
                yaw_rate_cmd = yaw_slew.step(yaw_rate_des, dt)
                yaw_cmd_prev = yaw_rate_cmd

                if SEND_COMMANDS:
                    send_velocity_yawrate(vx, vy, vz, yaw_rate_cmd)

                # HUD
                hud = f"vx={vx:.2f} m/s  yaw={math.degrees(yaw_rate_cmd):+.1f} deg/s"
                if width_px is not None: hud += f"  w_px={width_px:.0f}"
                cv2.putText(overlay, hud, (10, H - 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                draw_state(overlay, STATE)
                draw_yaw_command_arrow(overlay, yaw_rate_cmd)

            else:
                # Lost target beyond coast window: gracefully decay last yaw & forward
                yaw_cmd_prev = decay_to_zero(yaw_cmd_prev, dt, tau=0.35)
                vx_hold = decay_to_zero(V_FWD, dt, tau=0.35)
                if SEND_COMMANDS:
                    send_velocity_yawrate(vx_hold, 0.0, 0.0, yaw_cmd_prev)
                draw_state(overlay, STATE)
                draw_yaw_command_arrow(overlay, yaw_cmd_prev)

            # FPS
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
            cv2.putText(overlay, f"FPS: {fps:.1f}", (10, H - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow(WINDOW_NAME, overlay)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                if SEND_COMMANDS:
                    send_velocity_yawrate(0.0, 0.0, 0.0, 0.0)
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
