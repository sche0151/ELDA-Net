"""
ELDA-Net Evaluation Metrics
=============================
Thesis Section 5.2.3.

Metrics computed at IoU threshold = 0.5 (thesis Section 1.2.4):
  - F1 score  = 2 * TP / (2*TP + FP + FN)
  - IoU score = TP / (TP + FP + FN)
  - Precision = TP / (TP + FP)
  - Recall    = TP / (TP + FN)

Epsilon-smoothed to avoid division by zero.
"""

import numpy as np


def compute_metrics(pred: np.ndarray, label: np.ndarray, eps: float = 1e-6) -> tuple:
    """
    Compute F1 and IoU between binary prediction and ground truth.

    Args:
        pred:  binary numpy array (bool or 0/1)
        label: binary numpy array (bool or 0/1)
        eps:   smoothing constant to avoid division by zero

    Returns:
        (f1, iou, precision, recall)
    """
    pred  = pred.astype(bool)
    label = label.astype(bool)

    tp = np.logical_and(pred, label).sum()
    fp = np.logical_and(pred, ~label).sum()
    fn = np.logical_and(~pred, label).sum()

    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * tp / (2 * tp + fp + fn + eps)
    iou       = tp / (tp + fp + fn + eps)

    return float(f1), float(iou), float(precision), float(recall)


def compute_metrics_batch(preds: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute averaged metrics over a batch.

    Args:
        preds:  [N, 1, H, W] binary numpy array
        labels: [N, 1, H, W] binary numpy array

    Returns:
        dict with mean f1, iou, precision, recall
    """
    results = [compute_metrics(p.ravel(), l.ravel()) for p, l in zip(preds, labels)]
    f1s, ious, precs, recs = zip(*results)
    return {
        'f1':        float(np.mean(f1s)),
        'iou':       float(np.mean(ious)),
        'precision': float(np.mean(precs)),
        'recall':    float(np.mean(recs)),
    }
