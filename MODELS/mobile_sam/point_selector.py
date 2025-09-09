import numpy as np
import cv2

def find_peak_depth(depth_mm, rough_min=500, rough_max=3000, bins=200):
    m = (depth_mm >= rough_min) & (depth_mm <= rough_max) & (depth_mm > 0)
    vals = depth_mm[m]
    if vals.size == 0:
        return None
    hist, edges = np.histogram(vals, bins=bins, range=(rough_min, rough_max))
    if hist.max() == 0:
        return None
    k = np.argmax(hist)
    return 0.5 * (edges[k] + edges[k+1])  # mm

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
        return None, None, None
    x,y,w,h,i = best
    comp = (lab == i).astype(np.uint8) * 255
    return comp, (x,y,w,h), i

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
    # Edge-padded moving average to st
