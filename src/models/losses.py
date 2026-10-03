"""
Loss Functions for Extreme Meteorological Bust Imbalance.
Addresses severe 5–10% base-rate sparsity using Focal Loss and Positive-Weighted BCE.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class FocalLoss(nn.Module):
        """
        Focal Loss for addressing severe class imbalance in extreme weather events:
          FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
        """
        def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            p = torch.sigmoid(inputs)
            ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
            p_t = p * targets + (1 - p) * (1 - targets)
            loss = ce_loss * ((1 - p_t) ** self.gamma)

            if self.alpha >= 0:
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
                loss = alpha_t * loss

            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            return loss

    class WeightedBCELoss(nn.Module):
        """Binary Cross Entropy with positive weight multiplier."""
        def __init__(self, pos_weight: float = 8.0):
            super().__init__()
            self.pos_weight = torch.tensor([pos_weight])
            self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            if self.pos_weight.device != inputs.device:
                self.pos_weight = self.pos_weight.to(inputs.device)
                self.bce.pos_weight = self.pos_weight
            return self.bce(inputs, targets)

else:
    # Pure NumPy fallback implementations
    class FocalLoss:
        def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
            self.alpha = alpha
            self.gamma = gamma

        def __call__(self, inputs: np.ndarray, targets: np.ndarray) -> float:
            p = 1.0 / (1.0 + np.exp(-inputs))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            p_t = p * targets + (1 - p) * (1 - targets)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = -alpha_t * ((1 - p_t) ** self.gamma) * np.log(p_t)
            return float(np.mean(loss))

    class WeightedBCELoss:
        def __init__(self, pos_weight: float = 8.0):
            self.pos_weight = pos_weight

        def __call__(self, inputs: np.ndarray, targets: np.ndarray) -> float:
            p = 1.0 / (1.0 + np.exp(-inputs))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            loss = -(self.pos_weight * targets * np.log(p) + (1 - targets) * np.log(1 - p))
            return float(np.mean(loss))
