"""
ELDA-Net Hybrid Classical Preprocessing Pipeline
==================================================
Thesis Chapter 3, Stages 2-5 (Sections 3.3 - 3.5).

Pipeline stages:
  Stage 2: Grayscale conversion + Gaussian noise reduction
  Stage 3: ROI masking
  Stage 4: Canny edge detection + HSV color filtering
  Stage 5: Probabilistic Hough Transform lane detection
  Final  : Normalize and convert to model input tensor format
"""

import cv2
import numpy as np


# ── Stage 2: Classical Preprocessing ─────────────────────────────────────────

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Luminance-weighted RGB -> grayscale (thesis Section 3.3.1)."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_denoise(gray: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """5x5 Gaussian smoothing, sigma=1.0 (thesis Section 3.3.2)."""
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)


def hsv_color_filter(image: np.ndarray,
                     white_lower=(0, 0, 200), white_upper=(180, 30, 255),
                     yellow_lower=(15, 80, 80), yellow_upper=(40, 255, 255)) -> np.ndarray:
    """
    HSV-space white + yellow lane marking filter (thesis Section 3.3).
    Returns binary mask with lane-colored pixels set to 255.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white_mask  = cv2.inRange(hsv, np.array(white_lower),  np.array(white_upper))
    yellow_mask = cv2.inRange(hsv, np.array(yellow_lower), np.array(yellow_upper))
    return cv2.bitwise_or(white_mask, yellow_mask)


# ── Stage 3: ROI Selection ────────────────────────────────────────────────────

def build_roi_mask(image_shape, vertices=None):
    """
    Trapezoidal ROI mask (thesis Section 3.4).
    Default vertices for 256x512 (H x W):
      bottom-left (0,256), bottom-right (512,256),
      top-right (340,140), top-left (172,140)
    Vertices in (x, y) = (col, row) order as required by cv2.fillPoly.
    """
    h, w = image_shape[:2]
    if vertices is None:
        vertices = np.array([[
            (0,   h),        # bottom-left
            (w,   h),        # bottom-right
            (int(w * 0.664), int(h * 0.547)),  # top-right (~340,140 for 256x512)
            (int(w * 0.336), int(h * 0.547)),  # top-left  (~172,140 for 256x512)
        ]], dtype=np.int32)
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, vertices, 255)
    return mask


def apply_roi(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Bitwise AND to apply ROI mask (thesis Eq. I_ROI = I & M)."""
    return cv2.bitwise_and(image, image, mask=mask)


# ── Stage 4: Canny Edge Detection ────────────────────────────────────────────

def canny_edges(gray: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Canny edge detector (thesis Section 3.4)."""
    return cv2.Canny(gray, low, high)


# ── Stage 5: Hough Transform Lane Detection ──────────────────────────────────

def hough_lane_lines(edges: np.ndarray,
                     rho: float = 1,
                     theta_deg: float = 1.0,
                     threshold: int = 20,
                     min_line_length: int = 40,
                     max_line_gap: int = 20):
    """
    Probabilistic Hough Transform (thesis Section 3.5).
    Returns list of line segments [[x1,y1,x2,y2], ...] or empty list.
    """
    lines = cv2.HoughLinesP(
        edges,
        rho=rho,
        theta=np.deg2rad(theta_deg),
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return []
    return lines.reshape(-1, 4).tolist()


# ── Full preprocessing for model input ───────────────────────────────────────

def preprocess_image(image: np.ndarray, size,
                     mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225)) -> np.ndarray:
    """
    Resize + ImageNet normalize + HWC->CHW (thesis Section 3.2).
    size: (height, width) tuple
    Returns float32 array of shape (3, H, W) ready for torch.from_numpy().
    """
    h, w = size[0], size[1]
    img = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
    img = img[:, :, ::-1].astype(np.float32) / 255.0   # BGR->RGB, normalize to [0,1]
    img = (img - np.array(mean)) / np.array(std)
    return np.transpose(img, (2, 0, 1)).astype(np.float32)


def classical_preprocess_full(image: np.ndarray, cfg: dict) -> dict:
    """
    Run the complete classical preprocessing pipeline and return all intermediates.
    cfg: dict from config.yaml['preprocessing']
    """
    proc = cfg
    size = tuple(cfg.get('input_size', [256, 512]))  # (H, W)
    h, w = size

    # Resize first
    frame = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)

    # Grayscale + denoise
    gray    = to_grayscale(frame)
    denoised = gaussian_denoise(gray,
                                kernel_size=proc.get('gaussian_kernel', 5),
                                sigma=proc.get('gaussian_sigma', 1.0))

    # ROI
    roi_verts = proc.get('roi_vertices', None)
    if roi_verts is not None:
        roi_verts = np.array([roi_verts], dtype=np.int32)
    roi_mask = build_roi_mask(frame.shape, vertices=roi_verts)

    # HSV color filter
    color_mask = hsv_color_filter(
        frame,
        white_lower=proc.get('hsv_white_lower',  [0, 0, 200]),
        white_upper=proc.get('hsv_white_upper',  [180, 30, 255]),
        yellow_lower=proc.get('hsv_yellow_lower', [15, 80, 80]),
        yellow_upper=proc.get('hsv_yellow_upper', [40, 255, 255]),
    )

    # Canny on ROI-masked denoised gray
    masked_gray = apply_roi(denoised, roi_mask)
    edges = canny_edges(masked_gray,
                        low=proc.get('canny_low', 50),
                        high=proc.get('canny_high', 150))

    # Hough lines
    lines = hough_lane_lines(
        edges,
        rho=proc.get('hough_rho', 1),
        theta_deg=proc.get('hough_theta_deg', 1.0),
        threshold=proc.get('hough_threshold', 20),
        min_line_length=proc.get('hough_min_line_length', 40),
        max_line_gap=proc.get('hough_max_line_gap', 20),
    )

    return {
        'frame': frame,
        'gray': gray,
        'denoised': denoised,
        'roi_mask': roi_mask,
        'color_mask': color_mask,
        'edges': edges,
        'hough_lines': lines,
    }
