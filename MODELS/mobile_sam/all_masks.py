#!/usr/bin/env python3
import torch
import cv2
import numpy as np
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

# --- Load model ---
model_type = "vit_t"
sam_checkpoint = "./MODELS/mobile_sam/weights/mobile_sam.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
sam.eval()

# Use automatic mask generator instead of predictor
mask_generator = SamAutomaticMaskGenerator(sam)

# --- Load image ---
img_path = "./DATA/RGB/sw_snapshot_20250811_132228_xyz.png"
image_bgr = cv2.imread(img_path)
if image_bgr is None:
    raise FileNotFoundError(f"Could not read image: {img_path}")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# --- Generate masks ---
print("Running MobileSAM AutomaticMaskGenerator...")
masks = mask_generator.generate(image_rgb)
print(f"Generated {len(masks)} masks.")

# --- Create colored overlay ---
overlay = np.zeros_like(image_bgr, dtype=np.uint8)

for mask in masks:
    color = np.random.randint(0, 255, size=3, dtype=np.uint8)
    overlay[mask["segmentation"]] = color

# Blend with original image for visualization
alpha = 0.6
vis = cv2.addWeighted(image_bgr, 1 - alpha, overlay, alpha, 0)

# --- Display ---
cv2.namedWindow("MobileSAM All Masks", cv2.WINDOW_NORMAL)
cv2.imshow("MobileSAM All Masks", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- (Optional) Save the mask overlay ---
# cv2.imwrite("./DATA/RGB/mobile_sam_all_masks.png", vis)
