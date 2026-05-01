"""
ELDA-Net Visualization Utilities
==================================
Thesis Section 5.1 / Figure 5.2.

Overlay rendering:
  - Solid green  : high-confidence lane detections (C >= conf_threshold)
  - Dashed blue  : ALEM-active region bounding box (C < conf_threshold)
  - Orange dashed: polynomial extrapolation curves
"""

import cv2
import numpy as np


def overlay_lanes(frame: np.ndarray, mask: np.ndarray,
                  confidence: float = 1.0,
                  conf_threshold: float = 0.70,
                  poly_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Render lane overlay onto frame (thesis Figure 5.2).

    Args:
        frame:          original BGR image [H, W, 3]
        mask:           binary lane mask   [H, W]  float or uint8
        confidence:     scalar C from model confidence head
        conf_threshold: C_min = 0.70 (ALEM threshold)
        poly_mask:      optional polynomial extrapolation mask [H, W]

    Returns:
        BGR overlay image
    """
    overlay = frame.copy()
    mask_bool = mask > 0.5

    if confidence >= conf_threshold:
        # High confidence: solid green overlay
        overlay[mask_bool] = [0, 255, 0]
    else:
        # ALEM active: semi-transparent green for detected region
        green_layer = overlay.copy()
        green_layer[mask_bool] = [0, 200, 0]
        overlay = cv2.addWeighted(overlay, 0.6, green_layer, 0.4, 0)

        # Draw ALEM dashed blue bounding box
        ys, xs = np.where(mask_bool)
        if len(ys) > 0:
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            _draw_dashed_rect(overlay, (x1, y1), (x2, y2), color=(255, 100, 0), thickness=2)
            # Label
            label = f"ALEM Active, C={confidence:.2f} < {conf_threshold:.2f}"
            cv2.putText(overlay, label, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1, cv2.LINE_AA)

        # Orange dashed polynomial extrapolation curves
        if poly_mask is not None:
            poly_bool = poly_mask > 0.5
            poly_layer = overlay.copy()
            poly_layer[poly_bool] = [0, 140, 255]   # orange in BGR
            overlay = cv2.addWeighted(overlay, 0.7, poly_layer, 0.3, 0)

    # Confidence HUD
    color = (0, 255, 0) if confidence >= conf_threshold else (0, 100, 255)
    cv2.putText(overlay, f"C={confidence:.3f}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    return overlay


def _draw_dashed_rect(img, pt1, pt2, color, thickness=1, dash_length=10):
    """Draw a dashed rectangle on img in-place."""
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, dash_length * 2):
        cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
    for y in range(y1, y2, dash_length * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)
