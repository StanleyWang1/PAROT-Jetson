#!/usr/bin/env python3
import os, glob
import cv2
import numpy as np
import torch
from mobile_sam import sam_model_registry, SamPredictor

IN_DIR_RGB  = "./DATA/RGB/"
IN_DIR_NPZ  = "./DATA/RGB_DEPTH/"

# -------------------- MobileSAM (load once) --------------------
model_type = "vit_t"
sam_checkpoint = "./MODELS/mobile_sam/weights/mobile_sam.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
sam.eval()
predictor = SamPredictor(sam)

# -------------------- Depth → point helpers --------------------
def find_peak_depth(depth_mm, rough_min=500, rough_max=3000, bins=200):
    m = (depth_mm >= rough_min) & (depth_mm <= rough_max) & (depth_mm > 0)
    vals = depth_mm[m]
    if vals.size == 0:
        return None
    hist, edges = np.histogram(vals, bins=bins, range=(rough_min, rough_max))
    if hist.max() == 0:
        return None
    k = int(np.argmax(hist))
    return 0.5 * (edges[k] + edges[k+1])

def mask_band(depth_mm, center_mm, tol_mm=200):
    lo, hi = max(1, center_mm - tol_mm), center_mm + tol_mm
    m = ((depth_mm >= lo) & (depth_mm <= hi)).astype(np.uint8) * 255
    m = cv2.medianBlur(m, 5)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=1)
    return m

def largest_vertical_component(mask, min_area=1500, min_aspect=1.4):
    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best, best_area = None, 0
    for i in range(1, num):
        x,y,w,h,area = stats[i]
        if area < min_area: 
            continue
        aspect = h / max(1, w)
        if aspect >= min_aspect and area > best_area:
            best, best_area = (x,y,w,h,i), area
    if not best:
        return None
    x,y,w,h,i = best
    comp = (lab == i).astype(np.uint8) * 255
    return comp

def centerline_from_mask(comp_mask, min_width_px=6, smooth_win=11, margin_px=20):
    H, W = comp_mask.shape
    ys, xs = [], []
    y0 = max(0, margin_px)
    y1 = max(y0+1, H - margin_px)
    for y in range(y0, y1):
        row = np.where(comp_mask[y] > 0)[0]
        if row.size >= min_width_px:
            xs.append(row.mean())
            ys.append(y)
    if len(xs) < 5:
        return None
    xs = np.asarray(xs, np.float32); ys = np.asarray(ys, np.float32)
    k = max(3, int(smooth_win) | 1); pad = k//2
    ap = np.pad(xs, (pad,pad), mode='edge')
    kern = np.ones(k, np.float32)/k
    xs_s = np.convolve(ap, kern, mode='valid')
    return np.stack([xs_s, ys], axis=1)

def pick_point_from_depth(depth_mm, rough_min=500, rough_max=3000, tol_mm=200):
    """Return (x,y) point inside trunk-like blob, or None if not found."""
    peak = find_peak_depth(depth_mm, rough_min, rough_max)
    if peak is None:
        return None
    band = mask_band(depth_mm, peak, tol_mm)
    comp = largest_vertical_component(band)
    if comp is None:
        return None
    cl = centerline_from_mask(comp)
    if cl is not None and len(cl) > 0:
        mid = len(cl)//2
        return int(round(cl[mid,0])), int(round(cl[mid,1]))
    # Fallback: component centroid
    ys, xs = np.where(comp > 0)
    if xs.size == 0:
        return None
    return int(round(xs.mean())), int(round(ys.mean()))

def load_matching_depth(base_name):
    """Try to find ./DATA/RGB_DEPTH/<base_name>.npz with a few prefix fallbacks."""
    candidates = [
        os.path.join(IN_DIR_NPZ, f"{base_name}.npz"),
    ]
    # handle sw_ prefix differences gracefully
    if base_name.startswith("sw_"):
        candidates.append(os.path.join(IN_DIR_NPZ, f"{base_name[3:]}.npz"))
    else:
        candidates.append(os.path.join(IN_DIR_NPZ, f"sw_{base_name}.npz"))

    for p in candidates:
        if os.path.exists(p):
            return np.load(p)
    return None

def depth_from_npz(npz):
    if "depth_mm" in npz:
        return npz["depth_mm"]
    if "depth" in npz:
        d = npz["depth"]
        if d.dtype != np.uint16 or d.max() < 100:
            d = (d.astype(np.float32) * 1000).astype(np.uint16)
        return d
    if "xyz" in npz:
        z_m = np.nan_to_num(npz["xyz"][...,2], nan=0.0)
        return (z_m * 1000).astype(np.uint16)
    return None

# -------------------- Collect RGB images --------------------
exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
files = []
for e in exts:
    files.extend(glob.glob(os.path.join(IN_DIR_RGB, e)))
files = sorted(files)
if not files:
    print(f"[WARN] No images found in {IN_DIR_RGB}")
    raise SystemExit

win = "MobileSAM (auto-point; r=manual repick, SPACE=next, q=quit)"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)

for idx, img_path in enumerate(files, start=1):
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        print(f"[ERROR] Could not read: {img_path}")
        continue
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    H, W = image_bgr.shape[:2]
    cv2.resizeWindow(win, min(1280, W), min(800, H))

    # --- auto-pick point from depth ---
    base = os.path.splitext(os.path.basename(img_path))[0]
    npz = load_matching_depth(base)
    auto_pt = None
    if npz is not None:
        depth_mm = depth_from_npz(npz)
        if depth_mm is not None and depth_mm.shape[:2] == image_bgr.shape[:2]:
            auto_pt = pick_point_from_depth(depth_mm,
                                            rough_min=500, rough_max=3000, tol_mm=200)

    click = {"pt": auto_pt}  # start with auto point if available
    masks = None
    scores = None

    # mouse handler for manual repick
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click["pt"] = (x, y)

    cv2.setMouseCallback(win, on_mouse)

    while True:
        # (re)compute prediction if we have a point and no masks yet
        if click["pt"] is not None and masks is None:
            pt = np.array([[click["pt"][0], click["pt"][1]]], dtype=np.float32)
            lbl = np.array([1], dtype=np.int32)
            masks, scores, _ = predictor.predict(
                point_coords=pt,
                point_labels=lbl,
                multimask_output=True
            )

        # prepare display
        if masks is not None:
            best_idx = int(np.argmax(scores))
            best_mask = masks[best_idx]
            overlay = image_bgr.copy()
            overlay[~best_mask] = (overlay[~best_mask] * 0.3).astype(np.uint8)
            if click["pt"] is not None:
                cv2.circle(overlay, click["pt"], 6, (0,255,255), 2, cv2.LINE_AA)
            msg = f"{idx}/{len(files)}  score={scores[best_idx]:.3f}  (r=repick, SPACE=next, q=quit)"
            cv2.putText(overlay, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
            cv2.imshow(win, overlay)
        else:
            disp = image_bgr.copy()
            hint = "Auto-point not found; click a point" if auto_pt is None else "Review auto-point (SPACE=next, r=repick)"
            cv2.putText(disp, f"{idx}/{len(files)}  {hint}  (q=quit)",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
            if click["pt"] is not None:
                cv2.circle(disp, click["pt"], 6, (0,255,255), 2, cv2.LINE_AA)
            cv2.imshow(win, disp)

        k = cv2.waitKey(20 if masks is None else 0) & 0xFF
        if k in (ord('q'), 27):  # q or ESC
            cv2.destroyAllWindows()
            raise SystemExit
        elif k == ord('r'):
            # manual repick on same image
            masks = None
            scores = None
            click["pt"] = None
        elif k == 32 and masks is not None:  # SPACE -> next image
            break
        elif k == 255:  # no key / continue loop for 20ms waits
            continue
        else:
            # Any other key ignored; loop continues
            continue

cv2.destroyAllWindows()
print("Done.")
