"""
ELDA-Net Multi-Task Loss
=========================
Thesis Section 3.6.2, Equations 8-11.

L_total = L_seg + lambda_conf * L_conf + lambda_poly * L_poly
L_seg   = lambda_bce * L_BCE + lambda_dice * L_Dice
L_conf  = BCE on confidence head
L_poly  = MSE on polynomial coefficient regression

lambda_bce  = 0.5, lambda_dice = 0.5 (Eq. 5)
lambda_conf = 0.3 (Section 3.6.2)
lambda_poly = 0.2 (Section 3.6.2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Dice loss addressing class imbalance in lane segmentation (thesis Eq. 4).
    pred_logits: raw logits [B, 1, H, W]
    target:      binary ground truth [B, 1, H, W]
    """
    pred = torch.sigmoid(pred_logits)
    intersection = (pred * target).sum(dim=(2, 3))
    union        = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


class ELDANetLoss(nn.Module):
    """
    Combined multi-task loss for ELDA-Net (thesis Equations 8-11).

    Args:
        lambda_bce:  weight for BCE component of seg loss (default 0.5)
        lambda_dice: weight for Dice component of seg loss (default 0.5)
        lambda_conf: weight for confidence loss (default 0.3)
        lambda_poly: weight for polynomial regression loss (default 0.2)
    """

    def __init__(self, lambda_bce=0.5, lambda_dice=0.5,
                 lambda_conf=0.3, lambda_poly=0.2):
        super().__init__()
        self.lambda_bce  = lambda_bce
        self.lambda_dice = lambda_dice
        self.lambda_conf = lambda_conf
        self.lambda_poly = lambda_poly
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, seg_logits, conf_pred, poly_pred,
                seg_target, conf_target=None, poly_target=None):
        """
        Args:
            seg_logits:  [B, 1, H, W] raw logits
            conf_pred:   [B, 1] confidence scores
            poly_pred:   [B, 3] polynomial coefficients
            seg_target:  [B, 1, H, W] binary masks
            conf_target: [B, 1] ground truth confidence (optional)
            poly_target: [B, 3] ground truth poly coefficients (optional)

        Returns:
            total_loss, dict of component losses
        """
        # Segmentation loss (Eq. 9)
        l_bce  = self.bce(seg_logits, seg_target)
        l_dice = dice_loss(seg_logits, seg_target)
        l_seg  = self.lambda_bce * l_bce + self.lambda_dice * l_dice

        # Confidence loss (Eq. 10) — use mean lane pixel probability as proxy gt
        if conf_target is None:
            with torch.no_grad():
                conf_target = seg_target.mean(dim=(2, 3)).view(-1, 1)
        l_conf = F.binary_cross_entropy(conf_pred, conf_target.clamp(0, 1))

        # Polynomial regression loss (Eq. 11) — only if targets available
        if poly_target is not None:
            l_poly = F.mse_loss(poly_pred, poly_target)
        else:
            l_poly = torch.tensor(0.0, device=seg_logits.device)

        total = l_seg + self.lambda_conf * l_conf + self.lambda_poly * l_poly

        return total, {
            'total': total.item(),
            'seg':   l_seg.item(),
            'bce':   l_bce.item(),
            'dice':  l_dice.item(),
            'conf':  l_conf.item(),
            'poly':  l_poly.item(),
        }
