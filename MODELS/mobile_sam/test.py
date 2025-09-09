#!/usr/bin/env python3
import torch
import cv2
import numpy as np
from mobile_sam import sam_model_registry, SamPredictor

# --- Load model ---
model_type = "vit_t"
sam_checkpoint = "./MODELS/mobile_sam/weights/mobile_sam.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
sam.eval()

predictor = SamPredictor(sam)

# --- Load image ---
img_path = "./DATA/RGB/sw_snapshot_20250811_132040_xyz.png"
image_bgr = cv2.imread(img_path)
if image_bgr is None:
    raise FileNotFoundError(f"Could not read image: {img_path}")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

predictor.set_image(image_rgb)

# --- UI state ---
click = {"pt": None}
win = "MobileSAM: left-click a point (r=repick, a/d=cycle, q=quit)"

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        click["pt"] = (x, y)

cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, min(1280, image_bgr.shape[1]), min(800, image_bgr.shape[0]))
cv2.setMouseCallback(win, on_mouse)

h, w, _ = image_rgb.shape
curr_idx = 0
masks = None
scores = None

while True:
    disp = image_bgr.copy()
    if click["pt"] is None:
        # Waiting for user click
        cv2.putText(disp, "Left-click a point on the object", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow(win, disp)
        k = cv2.waitKey(20) & 0xFF
        if k in (ord('q'), 27):  # q or ESC
            break
        if click["pt"] is not None:
            # Run SAM once we have a point
            pt = np.array([[click["pt"][0], click["pt"][1]]], dtype=np.float32)  # (1,2)
            lbl = np.array([1], dtype=np.int32)  # 1 = foreground
            masks, scores, _ = predictor.predict(
                point_coords=pt,
                point_labels=lbl,
                multimask_output=True
            )
            curr_idx = int(np.argmax(scores))  # start at best
    else:
        # We have predictions; show current selection
        best_mask = masks[curr_idx]
        overlay = image_bgr.copy()
        overlay[~best_mask] = (overlay[~best_mask] * 0.3).astype(np.uint8)
        # draw the click
        cv2.circle(overlay, click["pt"], 6, (0,255,255), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"mask {curr_idx+1}/{len(masks)}  score={scores[curr_idx]:.3f}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

        cv2.imshow(win, overlay)
        k = cv2.waitKey(0) & 0xFF

        if k in (ord('q'), 27):  # quit
            break
        elif k == ord('r'):      # repick point
            click["pt"] = None
            masks = None
            scores = None
            curr_idx = 0
        elif k == ord('a'):      # previous mask
            curr_idx = (curr_idx - 1) % len(masks)
        elif k == ord('d'):      # next mask
            curr_idx = (curr_idx + 1) % len(masks)
        else:
            # any other key: accept current and exit (or you could save here)
            break

cv2.destroyAllWindows()
