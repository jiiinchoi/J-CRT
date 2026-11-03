"""Class-wise patient-level conformal risk control (CRC) and evaluation.

For every foreground class c, independently:
    units      = patients with >= 1 ground-truth voxel of class c
    L_c,i(lam) = |{v in Y_c,i : p_c(v) < lam}| / |Y_c,i|            (patient voxel FNR)
    lam_hat_c  = max{lam : (N_c * R_hat_c(lam) + 1) / (N_c + 1) < alpha_c}
Guarantee (exchangeable patients): E[L_c | c present] <= alpha_c for each class separately.
Prediction: Y_hat_c = 1[p_c >= lam_hat_c], one binary mask per class.
"""

from __future__ import annotations

import numpy as np
import torch


class InfeasibleAlpha(ValueError):
    pass


@torch.no_grad()
def collect_predictions(model, dataset, device, batch_size: int = 64) -> dict[str, dict]:
    """patient_id -> {"prob": float16 [S, K-1, H, W] (foreground classes 1..K-1), "gt": uint8 [S, H, W]}.
    `dataset` must provide patient_slices() -> {patient_id: [slice indices]} and get_batch(indices)."""
    model.eval()
    out = {}
    for pid, idxs in dataset.patient_slices().items():
        idxs = np.asarray(idxs)
        prob, gt = [], []
        for s in range(0, len(idxs), batch_size):
            img, msk = dataset.get_batch(idxs[s:s + batch_size])
            p = torch.softmax(model(img.to(device)).float(), dim=1)
            prob.append(p[:, 1:].half().cpu())
            gt.append(msk.to(torch.uint8))
        out[pid] = {"prob": torch.cat(prob).numpy(), "gt": torch.cat(gt).numpy()}
    return out


def positive_scores(preds: dict, k: int) -> list[np.ndarray]:
    """Sorted p_k on ground-truth-k voxels, one array per patient that contains class k."""
    out = []
    for pid in sorted(preds):
        d = preds[pid]
        s = d["prob"][:, k - 1][d["gt"] == k].astype(np.float32)
        if s.size:
            out.append(np.sort(s))
    return out


def patient_fnr_risk(scores: list[np.ndarray], lam: float) -> float:
    return float(np.mean([np.searchsorted(s, lam, side="left") / s.size for s in scores]))


def crc_lambda(scores: list[np.ndarray], alpha: float) -> float:
    """Largest candidate threshold (a positive score) satisfying the finite-sample corrected CRC rule.
    The empirical risk is non-decreasing in lam, so the feasible set is found by binary search."""
    N = len(scores)
    if alpha <= 1.0 / (N + 1):
        raise InfeasibleAlpha(f"alpha={alpha} <= 1/(N+1)={1 / (N + 1):.4f} for N={N}")
    cands = np.unique(np.concatenate(scores))
    ok = lambda lam: (N * patient_fnr_risk(scores, lam) + 1) / (N + 1) < alpha
    lo, hi = 0, len(cands) - 1
    assert ok(cands[lo])
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ok(cands[mid]):
            lo = mid
        else:
            hi = mid - 1
    return float(cands[lo])


def calibrate(preds: dict, classes: list[int], alphas: dict[int, float]) -> dict[int, float | None]:
    """{class: lambda} from calibration-set predictions; None if alpha is infeasible for that class."""
    lams = {}
    for k in classes:
        try:
            lams[k] = crc_lambda(positive_scores(preds, k), alphas[k])
        except InfeasibleAlpha:
            lams[k] = None
    return lams


def _dice(p: np.ndarray, g: np.ndarray) -> float | None:
    gs = g.sum()
    return float(2 * (p & g).sum() / (p.sum() + gs)) if gs else None


def evaluate(preds: dict, lams: dict[int, float | None]) -> dict:
    """Per class at Y_hat_c = p_c >= lam_c: patient-mean voxel FNR over patients containing c (the controlled risk),
    patient-mean voxel FPR over all patients (negatives = voxels with label != c), patient-mean Dice, pooled
    predicted/ground-truth volume ratio."""
    out = {}
    for k, lam in lams.items():
        if lam is None:
            out[k] = None
            continue
        fnr, fpr, dice, pv, gv = [], [], [], 0, 0
        for pid in sorted(preds):
            d = preds[pid]
            p = d["prob"][:, k - 1].astype(np.float32) >= lam
            g = d["gt"] == k
            if g.any():
                fnr.append(float((~p & g).sum() / g.sum()))
                dice.append(_dice(p, g))
            fpr.append(float((p & ~g).sum() / (~g).sum()))
            pv += int(p.sum()); gv += int(g.sum())
        out[k] = {"lambda": lam, "fnr_patient_mean": float(np.mean(fnr)), "n_patients_with_class": len(fnr),
                  "fpr_patient_mean": float(np.mean(fpr)), "dice_patient_mean": float(np.mean(dice)),
                  "vol_ratio": pv / gv}
    return out
