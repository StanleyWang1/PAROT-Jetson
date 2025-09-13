#!/usr/bin/env python3
import time
import cv2
import numpy as np
import depthai as dai
import torch
from mobile_sam import sam_model_registry, SamPredictor

# ====== USER SETTINGS ======
RGB_RESOLUTION = (640, 360)  # (width, height) for preview
RGB_FPS = 30
LIVE_UI_SCALE = 1.6          # display scale
MODEL_TYPE = "vit_t"
SAM_CHECKPOINT = "./MODELS/mobile_sam/weights/mobile_sam.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SAM rate limit (5 Hz)
SAM_PERIOD = 1/30  # seconds

# ====== MobileSAM (load once) ======
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)
sam.eval()
predictor = SamPredictor(sam)

# ====== Overlay styling ======
def make_overlay(image_bgr, mask_bool, point_xy=None, score=None, mode="auto"):
    """
    Darken background strongly and tint masked region light cyan.
    """
    vis = image_bgr.copy()

    # Background: strong darkening
    bg = ~mask_bool
    vis[bg] = (vis[bg].astype(np.float32) * 0.85).astype(np.uint8)

    # Masked region: light cyan tint (BGR cyan ≈ (255,255,0))
    cyan = np.array([255, 255, 0], dtype=np.uint8)
    fg = mask_bool
    if np.any(fg):
        alpha = 0.25
        blend = (1.0 - alpha) * vis[fg].astype(np.float32) + alpha * cyan.astype(np.float32)
        vis[fg] = np.clip(blend, 0, 255).astype(np.uint8)

    # Mark point + HUD
    if point_xy is not None:
        cv2.circle(vis, point_xy, 6, (0,255,255), 2, cv2.LINE_AA)
    label = f"mode={mode}"
    if score is not None:
        label += f"  score={score:.3f}"
    cv2.putText(vis, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
    return vis

# ====== DepthAI pipeline (RGB only) ======
pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.ColorCamera)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setFps(RGB_FPS)
cam.setInterleaved(False)
cam.setPreviewSize(*RGB_RESOLUTION)
cam.setPreviewKeepAspectRatio(True)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam.preview.link(xout_rgb.input)

# ====== Realtime loop (5 Hz SAM, center-point) ======
with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue("rgb", maxSize=2, blocking=False)

    win = "MobileSAM Live (center point, 5 Hz) — r: manual point, q: quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    manual_point = None   # one-shot override on key 'r'
    last_mask = None
    last_point = None
    last_score = None
    last_mode = "auto"
    last_sam_time = 0.0

    while True:
        pkt_rgb = q_rgb.tryGet()
        if pkt_rgb is None:
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
            continue

        rgb_bgr = pkt_rgb.getCvFrame()
        H, W = rgb_bgr.shape[:2]

        # Keys
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            break
        if k == ord('r'):
            # one-shot manual point on current frozen frame
            frozen = rgb_bgr.copy()
            clicked = {'pt': None}
            pick_win = "Pick point (click; q=cancel)"
            cv2.namedWindow(pick_win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(pick_win, int(RGB_RESOLUTION[0]*LIVE_UI_SCALE), int(RGB_RESOLUTION[1]*LIVE_UI_SCALE))
            def on_mouse(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    clicked['pt'] = (x, y)
            cv2.setMouseCallback(pick_win, on_mouse)

            while True:
                view = frozen.copy()
                if clicked['pt'] is not None:
                    cv2.circle(view, clicked['pt'], 6, (0,255,255), 2, cv2.LINE_AA)
                cv2.imshow(pick_win, view)
                k2 = cv2.waitKey(20) & 0xFF
                if k2 in (ord('q'), 27):
                    clicked['pt'] = None
                    cv2.destroyWindow(pick_win)
                    break
                if clicked['pt'] is not None:
                    manual_point = clicked['pt']
                    cv2.destroyWindow(pick_win)
                    break

        # Run SAM at most every SAM_PERIOD seconds
        now = time.time()
        if (now - last_sam_time) >= SAM_PERIOD:
            # Point choice: manual (one-shot) else image center
            if manual_point is not None:
                pt_xy = manual_point
                last_mode = "manual"
            else:
                pt_xy = (W // 2, H // 2)
                last_mode = "auto"

            # Run MobileSAM with that point
            rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            predictor.set_image(rgb_rgb)  # heavy step
            point_coords = np.array([[pt_xy[0], pt_xy[1]]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)  # foreground click

            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )
            best_idx = int(np.argmax(scores))
            last_mask = masks[best_idx]
            last_point = pt_xy
            last_score = float(scores[best_idx])
            last_sam_time = now

            manual_point = None  # one-shot override consumed

        # Draw using the latest available mask
        if last_mask is not None:
            overlay = make_overlay(rgb_bgr, last_mask.astype(bool), last_point, last_score, mode=last_mode)
        else:
            overlay = rgb_bgr.copy()
            cv2.putText(overlay, "No mask yet… (press 'r' to pick)", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

        disp = cv2.resize(
            overlay,
            (int(RGB_RESOLUTION[0]*LIVE_UI_SCALE), int(RGB_RESOLUTION[1]*LIVE_UI_SCALE)),
            interpolation=cv2.INTER_NEAREST
        )
        cv2.imshow(win, disp)

    cv2.destroyAllWindows()
