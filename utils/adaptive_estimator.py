"""
ELDA-Net Adaptive Lane Estimation Mechanism (ALEM)
====================================================
Thesis Chapter 4, Sections 4.1 - 4.6.

ALEM is a confidence-gated mechanism (C_min = 0.70) that activates when
the U-Net segmentation confidence falls below threshold. It integrates:

  Component 1: Exponential temporal smoothing (alpha=0.3, 10-frame buffer)
  Component 2: Second-order polynomial curve extrapolation (x = ay^2 + by + c)
  Component 3: Kalman filter predictive tracking

When confidence >= C_min, direct segmentation output is used.
When confidence <  C_min, the three components are fused adaptively
based on per-component confidence weights (thesis Section 4.6).
"""

import cv2
import numpy as np


class TemporalSmoother:
    """
    Component 1: Exponential Moving Average over lane mask predictions.
    Equation 6 in thesis: L_bar_t = alpha * L_t + (1 - alpha) * L_bar_{t-1}
    alpha = 0.3, buffer = 10 frames (thesis Section 1.4 / Section 4.2).
    """

    def __init__(self, alpha: float = 0.3, buffer_size: int = 10):
        self.alpha = alpha
        self.buffer_size = buffer_size
        self.buffer = []       # ring buffer of recent masks
        self.smoothed = None   # current EMA estimate

    def update(self, mask: np.ndarray) -> np.ndarray:
        """Add new mask and return EMA smoothed estimate."""
        if self.smoothed is None:
            self.smoothed = mask.astype(np.float32)
        else:
            self.smoothed = (self.alpha * mask.astype(np.float32)
                             + (1 - self.alpha) * self.smoothed)
        self.buffer.append(mask.copy())
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        return self.smoothed

    def estimate(self) -> np.ndarray | None:
        """Return current smoothed estimate (None if no history)."""
        return self.smoothed

    def reset(self):
        self.buffer.clear()
        self.smoothed = None


class PolynomialExtrapolator:
    """
    Component 2: Second-order polynomial curve extrapolation.
    Fits x = ay^2 + by + c to detected lane points (thesis Section 4.3 / Eq. 11).
    Requires >= 15 points spanning >= 30% of lane height (thesis Section 4.4).

    Confidence for extrapolated region decays exponentially (thesis Eq. 15):
      C_poly(y) = C_visible * exp(-lambda_decay * |y - y_vis_boundary| / H)
    lambda_decay = 2.5
    """

    def __init__(self, min_points: int = 15, min_span: float = 0.30,
                 lambda_decay: float = 2.5):
        self.min_points = min_points
        self.min_span = min_span
        self.lambda_decay = lambda_decay
        self.coeffs = None    # (a, b, c)
        self.y_vis_boundary = None
        self.c_visible = 0.0

    def fit(self, mask: np.ndarray, confidence: float) -> bool:
        """
        Extract lane pixels from mask and fit polynomial.
        Returns True if fit was successful.
        """
        ys, xs = np.where(mask > 0.5)
        if len(ys) < self.min_points:
            return False
        h = mask.shape[0]
        span = (ys.max() - ys.min()) / h
        if span < self.min_span:
            return False
        # Fit x = a*y^2 + b*y + c using least squares (thesis Eq. 11)
        A = np.column_stack([ys**2, ys, np.ones_like(ys)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, xs, rcond=None)
        except np.linalg.LinAlgError:
            return False
        self.coeffs = coeffs
        self.y_vis_boundary = float(ys.min())
        self.c_visible = float(confidence)
        return True

    def extrapolate(self, image_shape) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate extrapolated mask and per-pixel confidence map.
        Returns (extrapolated_mask, confidence_map).
        """
        if self.coeffs is None:
            h, w = image_shape[:2]
            return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

        h, w = image_shape[:2]
        a, b, c = self.coeffs
        mask = np.zeros((h, w), dtype=np.float32)
        conf_map = np.zeros((h, w), dtype=np.float32)

        ys = np.arange(h)
        xs = (a * ys**2 + b * ys + c).astype(int)
        valid = (xs >= 0) & (xs < w)

        for y_idx, x_idx in zip(ys[valid], xs[valid]):
            mask[y_idx, x_idx] = 1.0
            dist = max(0.0, self.y_vis_boundary - y_idx) / h
            c_val = self.c_visible * np.exp(-self.lambda_decay * dist)
            conf_map[y_idx, x_idx] = float(c_val)

        return mask, conf_map

    def reset(self):
        self.coeffs = None
        self.y_vis_boundary = None
        self.c_visible = 0.0


class KalmanLaneTracker:
    """
    Component 3: Kalman filter predictive tracking (thesis Section 4.5).

    State vector: [a, b, c, da, db, dc] where (a,b,c) are polynomial coefficients
    and (da,db,dc) their velocities.

    Transition: constant velocity model F = [[I, I], [0, I]]
    Observation: H = [I, 0] (observe coefficients directly)
    """

    def __init__(self, process_noise: float = 1e-4, measurement_noise: float = 1e-2):
        self.dim_x = 6   # state: [a, b, c, da, db, dc]
        self.dim_z = 3   # observation: [a, b, c]
        self.initialized = False

        # State transition F (constant velocity)
        self.F = np.eye(self.dim_x)
        self.F[0, 3] = 1.0
        self.F[1, 4] = 1.0
        self.F[2, 5] = 1.0

        # Observation matrix H
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Noise covariances
        self.Q = np.eye(self.dim_x) * process_noise
        self.R = np.eye(self.dim_z) * measurement_noise

        # State and covariance
        self.x = np.zeros((self.dim_x, 1))
        self.P = np.eye(self.dim_x)

    def _predict(self):
        """Kalman prediction step."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def _update(self, z: np.ndarray):
        """Kalman update step given measurement z [3,]."""
        z = z.reshape(self.dim_z, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P

    def update(self, coeffs: np.ndarray | None) -> np.ndarray:
        """
        Update with new polynomial coefficients (or None during occlusion).
        Returns predicted/updated [a, b, c] coefficients.
        """
        if not self.initialized:
            if coeffs is not None:
                self.x[:3] = coeffs.reshape(3, 1)
                self.initialized = True
            return self.x[:3].ravel()

        self._predict()
        if coeffs is not None:
            self._update(coeffs)
        return self.x[:3].ravel()

    def reset(self):
        self.initialized = False
        self.x = np.zeros((self.dim_x, 1))
        self.P = np.eye(self.dim_x)


class AdaptiveLaneEstimator:
    """
    Full ALEM system (thesis Chapter 4).
    Integrates all three components with confidence-gated activation.

    Usage:
        estimator = AdaptiveLaneEstimator(cfg['alem'])
        refined_mask = estimator.refine(pred_mask, confidence_score, poly_coeffs)
    """

    def __init__(self, cfg: dict | None = None):
        if cfg is None:
            cfg = {}
        self.conf_threshold  = cfg.get('confidence_threshold', 0.70)
        self.morph_kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        alpha       = cfg.get('smoothing_alpha', 0.3)
        buf_size    = cfg.get('temporal_buffer_size', 10)
        min_pts     = cfg.get('min_poly_points', 15)
        min_span    = cfg.get('min_poly_span', 0.30)
        lam_decay   = cfg.get('lambda_decay', 2.5)
        proc_noise  = cfg.get('kalman', {}).get('process_noise_cov', 1e-4)
        meas_noise  = cfg.get('kalman', {}).get('measurement_noise_cov', 1e-2)

        self.smoother   = TemporalSmoother(alpha=alpha, buffer_size=buf_size)
        self.poly_ext   = PolynomialExtrapolator(min_points=min_pts,
                                                  min_span=min_span,
                                                  lambda_decay=lam_decay)
        self.kalman     = KalmanLaneTracker(process_noise=proc_noise,
                                             measurement_noise=meas_noise)

    def _morphological_close(self, mask: np.ndarray) -> np.ndarray:
        """5x5 elliptical closing to fill gaps (thesis Section 3.7)."""
        return cv2.morphologyEx(mask.astype(np.uint8),
                                cv2.MORPH_CLOSE, self.morph_kernel).astype(np.float32)

    def refine(self, pred_mask: np.ndarray,
               confidence: float = 1.0,
               poly_coeffs: np.ndarray | None = None) -> np.ndarray:
        """
        Main ALEM entry point.

        Args:
            pred_mask:   binary prediction from U-Net [H, W] float/uint8
            confidence:  scalar C from confidence head (0-1)
            poly_coeffs: [a, b, c] from polynomial head (optional)

        Returns:
            refined_mask: [H, W] float32
        """
        mask = pred_mask.astype(np.float32)

        # Always update temporal smoother with latest prediction
        smoothed = self.smoother.update(mask)

        if confidence >= self.conf_threshold:
            # High confidence: direct detection path
            # Update Kalman and poly with this good observation
            if poly_coeffs is not None:
                self.kalman.update(poly_coeffs)
            self.poly_ext.fit(mask, confidence)
            result = self._morphological_close(mask)
        else:
            # Low confidence: ALEM adaptive estimation path
            # -- Component 1: temporal smoothing estimate
            smooth_est = smoothed if smoothed is not None else mask

            # -- Component 2: polynomial extrapolation
            if poly_coeffs is not None:
                self.poly_ext.fit(mask, confidence)
            poly_mask, poly_conf = self.poly_ext.extrapolate(mask.shape)

            # -- Component 3: Kalman prediction (no measurement during occlusion)
            kalman_coeffs = self.kalman.update(None)
            a, b, c = kalman_coeffs
            h, w = mask.shape
            kalman_mask = np.zeros((h, w), dtype=np.float32)
            ys = np.arange(h)
            xs = (a * ys**2 + b * ys + c).astype(int)
            valid = (xs >= 0) & (xs < w)
            for y_idx, x_idx in zip(ys[valid], xs[valid]):
                kalman_mask[y_idx, x_idx] = 1.0

            # Adaptive fusion: weight by per-component confidence
            w_smooth = 0.4
            w_poly   = float(np.clip(confidence, 0, 1)) * 0.35
            w_kalman = 0.25
            total = w_smooth + w_poly + w_kalman
            fused = (w_smooth * smooth_est
                     + w_poly   * poly_mask
                     + w_kalman * kalman_mask) / total

            result = self._morphological_close((fused > 0.5).astype(np.float32))

        return result

    def reset(self):
        self.smoother.reset()
        self.poly_ext.reset()
        self.kalman.reset()
