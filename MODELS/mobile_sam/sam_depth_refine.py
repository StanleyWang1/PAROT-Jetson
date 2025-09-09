#!/usr/bin/env python3
import os
import glob
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from mobile_sam import sam_model_registry, SamPredictor

# ---------------- Config ----------------
NPZ_DIR   = "./DATA/RGB_DEPTH"           # folder of NPZ files
OUT_DIR   = "PROCESSING/sam_point_from_depth_centroid/"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_TYPE = "vit_t"
SAM_CKPT   = "./MODELS/mobile_sam/weights/mobile_sam.pt"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Depth range (mm)
DEPTH_MIN_MM = 1000   # 1 m
DEPTH_MAX_MM = 3000   # 3 m

# Noise removal / size filter (pixels)
MIN_AREA_PX  = 3000   # remove blobs smaller than this
MEDIAN_K     = 5      # median blur kernel
MORPH_K      = 5      # morphology kernel size

# Overlay look
BG_DARK      = 0.20   # multiply background brightness
MASK_TINT    = 0.30   # cyan tint strength on mask


# --------------- Helpers ---------------
def load_rgbd_npz(path):
    d = np.load(path)
    if "rgb" not in d:
        raise ValueError("NPZ missing 'rgb'")
    rgb = d["rgb"]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Unexpected rgb shape: {rgb.shape}")
    rgb_rgb = rgb
    rgb_bgr = cv2.cvtColor(rgb_rgb, cv2.COLOR_RGB2BGR)

    if "depth_mm" in d:
        depth_mm = d["depth_mm"]
    elif "depth" in d:
        depth_mm = d["depth"]
        # convert meters->mm if needed
        if depth_mm.dtype != np.uint16 or depth_mm.max() < 100:
            depth_mm = (depth_mm.astype(np.float32) * 1000).astype(np.uint16)
    elif "xyz" in d:
        z_m = np.nan_to_num(d["xyz"][..., 2], nan=0.0)
        depth_mm = (z_m * 1000).astype(np.uint16)
    else:
        raise ValueError("NPZ missing depth ('depth_mm'/'depth'/'xyz')")
    return rgb_rgb, rgb_bgr, depth_mm


def depth_mask_1_3m(depth_mm):
    m = (depth_mm >= DEPTH_MIN_MM) & (depth_mm <= DEPTH_MAX_MM) & (depth_mm > 0)
    mask = (m.astype(np.uint8) * 255)
    if MEDIAN_K > 1:
        mask = cv2.medianBlur(mask, MEDIAN_K)
    if MORPH_K > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask


def keep_large_components(mask_u8, min_area=MIN_AREA_PX):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    out = np.zeros_like(mask_u8)
    if num <= 1:
        return out
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            out[labels == i] = 255
    return out


def centroid_of_largest(mask_u8):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    best_i, best_area = None, 0
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > best_area:
            best_area = area
            best_i = i
    if best_i is None or best_area == 0:
        return None
    ys, xs = np.where(labels == best_i)
    if xs.size == 0:
        return None
    cx, cy = int(round(xs.mean())), int(round(ys.mean()))
    return (cx, cy)


def overlay_mask_cyan(image_bgr, mask_bool, point_xy=None, score=None):
    vis = image_bgr.copy()
    bg = ~mask_bool
    vis[bg] = (vis[bg].astype(np.float32) * BG_DARK).astype(np.uint8)
    cyan = np.array([255, 255, 0], dtype=np.uint8)  # BGR cyan
    fg = mask_bool
    if np.any(fg):
        blend = (1.0 - MASK_TINT) * vis[fg].astype(np.float32) + MASK_TINT * cyan.astype(np.float32)
        vis[fg] = np.clip(blend, 0, 255).astype(np.uint8)
    if point_xy is not None:
        cv2.circle(vis, point_xy, 6, (0, 255, 255), 2, cv2.LINE_AA)
    if score is not None:
        cv2.putText(vis, f"score={score:.3f}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
    return vis


# ------------- Load MobileSAM once -------------
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CKPT)
sam.to(device=DEVICE)
sam.eval()
predictor = SamPredictor(sam)

# ------------- Iterate NPZ files -------------
npz_files = sorted(glob.glob(os.path.join(NPZ_DIR, "*.npz")))
if not npz_files:
    raise FileNotFoundError(f"No NPZ files found in {NPZ_DIR}")

for npz_path in npz_files:
    try:
        base = os.path.splitext(os.path.basename(npz_path))[0]
        print(f"Processing {base}")

        # Load
        rgb_rgb, rgb_bgr, depth_mm = load_rgbd_npz(npz_path)
        H, W = depth_mm.shape

        # Depth → mask → filter → centroid
        mask_u8 = depth_mask_1_3m(depth_mm)
        mask_large = keep_large_components(mask_u8, MIN_AREA_PX)
        centroid = centroid_of_largest(mask_large)

        if centroid is None:
            print(f"  [WARN] No large component found (>{MIN_AREA_PX} px). Skipping SAM.")
            # Still save a 2-panel (RGB + depth mask) for debugging
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(rgb_rgb); axs[0].set_title("Original RGB"); axs[0].axis("off")
            dm_vis = cv2.cvtColor(cv2.merge([mask_large]*3), cv2.COLOR_BGR2RGB)
            axs[1].imshow(dm_vis);  axs[1].set_title("Depth mask (filtered)"); axs[1].axis("off")
            plt.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, f"{base}_rgb_depthmask_noSAM.png"), dpi=150)
            plt.close(fig)
            continue

        # Run SAM with centroid point
        predictor.set_image(rgb_rgb)  # SAM expects RGB
        point_coords = np.array([[centroid[0], centroid[1]]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)  # foreground
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )
        best_idx = int(np.argmax(scores))
        sam_mask = masks[best_idx].astype(bool)
        score = float(scores[best_idx])

        # Build visuals
        # Panel 1: raw RGB
        p1 = rgb_rgb

        # Panel 2: filtered depth mask with centroid dot
        mask_vis_bgr = cv2.merge([mask_large, mask_large, mask_large])  # gray mask
        cv2.circle(mask_vis_bgr, centroid, 6, (0, 255, 255), 2, cv2.LINE_AA)
        p2 = cv2.cvtColor(mask_vis_bgr, cv2.COLOR_BGR2RGB)

        # Panel 3: SAM overlay with input point
        overlay_bgr = overlay_mask_cyan(rgb_bgr, sam_mask, centroid, score)
        p3 = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        # Save combined plot (no BW mask file)
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(p1); axs[0].set_title("Original RGB"); axs[0].axis("off")
        axs[1].imshow(p2); axs[1].set_title("Depth mask (1–3 m, filtered + centroid)"); axs[1].axis("off")
        axs[2].imshow(p3); axs[2].set_title("SAM mask (cyan) with input point"); axs[2].axis("off")
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"{base}_rgb_depthmask_centroid_sam.png"), dpi=150)
        plt.close(fig)

    except Exception as e:
        print(f"[ERROR] {npz_path}: {e}")

print(f"Done. Figures saved to {OUT_DIR}")
