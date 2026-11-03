"""J-CRT training objective.

One shared softmax network with K output channels (channel 0 = background). Every foreground class c is treated
one-vs-rest on p_c = softmax(z)_c. Each training batch is [D_cal; D_pred]:

    lambda_c(theta) = windowed differentiable CRC threshold on the c-positive D_cal slices   (get_lambda_theta)
    L_FP,c          = sigmoid surrogate of the voxel FPR of class c on D_pred at lambda_c   (fpr_loss)
    L(theta)        = mean_c L_FP,c + w_anchor * [CE(label smoothing) + soft Dice](whole batch)

Ablation (detach_lambda=True): lambda_c(theta) is still recomputed from the current model every step, but is detached,
so no gradient flows from L_FP,c through the threshold into the D_cal scores (only through the D_pred scores).

get_lambda_theta / fpr_loss follow the reference CRT implementation of Yeh et al.
(github.com/chrisyeh96/conformal-risk-training, run_polyp.py).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

DEFAULT_CONFIG = {
    "lr": 1e-4,
    "T": 0.01,                 # temperature of the sigmoid FP surrogate
    "alpha": 0.10,             # target FNR used inside training (same for every class)
    "num_cal_per_class": 12,   # k: c-positive slices per class in D_cal
    "num_pred": 16,            # m: D_pred slices (any slice)
    "steps_per_epoch": 250,
    "epochs": 50,
    "w_anchor": 1.0,
    "label_smoothing": 0.02,
    "seed": 0,
    "detach_lambda": False,    # ablation: True stops the gradient through lambda_c(theta)
}


def get_lambda_theta(preds_cal: Tensor, masks_cal: Tensor, alpha: float) -> Tensor:
    """Differentiable windowed calibration threshold lambda(theta).

    preds_cal: [n, 1, H, W] class probabilities on D_cal (NOT detached), masks_cal: [n, 1, H, W] bool.
    Positive voxels are weighted so that every slice sums to 1 (slice-level FNR). Only the indices are
    computed without gradient; lambda is the mean of the scores in a 0.5 % window around the CRC quantile
    with the (n+1) finite-sample correction, so it stays connected to theta.
    """
    num_cal = len(preds_cal)
    flat_preds = preds_cal[masks_cal]
    with torch.no_grad():
        weights = masks_cal / masks_cal.sum(dim=(-2, -1), keepdim=True)
        flat_weights = weights[masks_cal]
        sort_inds = torch.argsort(flat_preds, descending=True)
        cum_weights = flat_weights[sort_inds].cumsum(dim=0)
        nz = torch.nonzero(cum_weights > (1 - alpha) * (num_cal + 1))
        if len(nz) == 0:
            raise RuntimeError(f"no valid calibration index for alpha={alpha} with {num_cal} slices; "
                               f"need num_cal_per_class > 1/alpha - 1")
        lam_ind = int(nz[0].item())
        half = int(np.ceil(len(flat_preds) / 400))
        lam_inds = range(max(0, lam_ind - half), min(len(flat_preds), lam_ind + half))
    return flat_preds[sort_inds][lam_inds].mean()


def fpr_loss(preds: Tensor, masks: Tensor, lam: Tensor, T: float) -> Tensor:
    """Sigmoid surrogate of the slice-normalised voxel FPR, differentiable in preds and lam.
    preds: [n, 1, H, W] probabilities, masks: [n, 1, H, W] bool positives."""
    n = len(preds)
    with torch.no_grad():
        neg = ~masks
        w = (neg / neg.sum(dim=(-2, -1), keepdim=True))[neg]
    return (torch.sigmoid((preds[neg] - lam) / T) * w).sum() / n


def anchor_loss(logits: Tensor, target: Tensor, label_smoothing: float, eps: float = 1e-6) -> Tensor:
    """CE(label smoothing) + soft Dice averaged over all K classes. Label smoothing keeps the softmax away from
    saturation, otherwise background probabilities collapse towards 0 and the FP surrogate has no gradient."""
    ce = F.cross_entropy(logits, target, label_smoothing=label_smoothing)
    probs = torch.softmax(logits, dim=1)
    onehot = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    inter = (probs * onehot).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + onehot.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return ce + (1 - dice.mean())


class ClasswiseBatchSampler:
    """Index sampler for [D_cal; D_pred].

    D_cal: for every foreground class (in order) num_cal_per_class slices containing that class, without duplicates
    within the batch. D_pred: num_pred slices drawn from all slices (independently of D_cal).

    slice_classes: per slice, the set of foreground labels present in its mask.
    """

    def __init__(self, slice_classes: list[set[int]], classes: list[int], num_cal_per_class: int, num_pred: int,
                 seed: int):
        self.pos = {k: np.array([i for i, s in enumerate(slice_classes) if k in s]) for k in classes}
        for k, idx in self.pos.items():
            if len(idx) < num_cal_per_class:
                raise ValueError(f"class {k}: only {len(idx)} positive slices < num_cal_per_class")
        self.n_all = len(slice_classes)
        self.k, self.num_pred = num_cal_per_class, num_pred
        self.rng = np.random.default_rng(seed)

    def next(self) -> tuple[np.ndarray, np.ndarray]:
        cal: list[int] = []
        for k in self.pos:
            cand = np.setdiff1d(self.pos[k], cal, assume_unique=True)
            cal.extend(self.rng.choice(cand, self.k, replace=False).tolist())
        pred = self.rng.choice(self.n_all, self.num_pred, replace=False)
        return np.array(cal), pred


def jcrt_loss(model, images: Tensor, masks: Tensor, n_cal: int, classes: list[int], alphas: dict[int, float],
              T: float, w_anchor: float, label_smoothing: float, detach_lambda: bool = False) -> dict:
    """images/masks = [D_cal; D_pred] (first n_cal rows are D_cal); masks hold integer labels.
    One forward pass; returns the total loss and its terms (not yet backpropagated)."""
    logits = model(images)
    prob = torch.softmax(logits.float(), dim=1)
    lam, fp = {}, {}
    for k in classes:
        pos_cal = masks[:n_cal] == k
        units = pos_cal.flatten(1).any(1)
        lam[k] = get_lambda_theta(prob[:n_cal, k][units].unsqueeze(1), pos_cal[units].unsqueeze(1), alphas[k])
        if detach_lambda:
            lam[k] = lam[k].detach()
        fp[k] = fpr_loss(prob[n_cal:, k:k + 1], (masks[n_cal:] == k).unsqueeze(1), lam=lam[k], T=T)
    fp_mean = torch.stack(list(fp.values())).mean()
    anchor = anchor_loss(logits.float(), masks, label_smoothing)
    return {"total": fp_mean + w_anchor * anchor, "fp_mean": fp_mean, "anchor": anchor, "lam": lam, "fp": fp}
