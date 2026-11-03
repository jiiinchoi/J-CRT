"""J-CRT fine-tuning from a trained softmax baseline.

Fixed budget (50 epochs x 250 steps, no early stopping). After every epoch each class is calibrated with
patient-level CRC at alpha on the validation split; the checkpoint with the lowest mean (over classes)
patient-mean voxel FPR is kept as best.pt.

    python train_jcrt.py --train data/train --val data/val --in-channels 4 --num-classes 4 \
        --base-ckpt runs/baseline/best.pt --out runs/jcrt

Ablation without the threshold gradient (lambda_c(theta) detached): add --detach-lambda.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from jcrt import DEFAULT_CONFIG, ClasswiseBatchSampler, SliceDataset, UNet2D, calibrate, collect_predictions, evaluate, \
    jcrt_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--val", required=True)
    ap.add_argument("--in-channels", type=int, required=True); ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--base-ckpt", required=True); ap.add_argument("--out", required=True)
    for k, v in DEFAULT_CONFIG.items():
        if isinstance(v, bool):
            ap.add_argument("--" + k.replace("_", "-"), action="store_true")
        else:
            ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args()
    cfg = {k: getattr(a, k) for k in DEFAULT_CONFIG}
    classes = list(range(1, a.num_classes))
    alphas = {k: cfg["alpha"] for k in classes}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(out / "config.json", "w"), indent=1)

    torch.manual_seed(cfg["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va = SliceDataset(a.train), SliceDataset(a.val)
    sampler = ClasswiseBatchSampler(tr.slice_classes(), classes, cfg["num_cal_per_class"], cfg["num_pred"], cfg["seed"])
    model = UNet2D(a.in_channels, a.num_classes)
    model.load_state_dict(torch.load(a.base_ckpt, map_location="cpu", weights_only=True))
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    best, t0 = float("inf"), time.time()
    for ep in range(cfg["epochs"]):
        model.train(); L = []
        for _ in range(cfg["steps_per_epoch"]):
            cal_idx, pred_idx = sampler.next()
            img, msk = tr.get_batch(np.concatenate([cal_idx, pred_idx]))
            r = jcrt_loss(model, img.to(dev), msk.to(dev), len(cal_idx), classes, alphas, cfg["T"], cfg["w_anchor"],
                          cfg["label_smoothing"], cfg["detach_lambda"])
            if not torch.isfinite(r["total"]):
                raise FloatingPointError(f"non-finite loss at epoch {ep}")
            opt.zero_grad(); r["total"].backward(); opt.step()
            L.append(r["total"].item())
        preds = collect_predictions(model, va, dev)
        lams = calibrate(preds, classes, alphas)
        if any(v is None for v in lams.values()):
            raise RuntimeError(f"validation CRC infeasible at alpha={cfg['alpha']}: {lams}")
        res = evaluate(preds, lams)
        mean_fpr = float(np.mean([res[k]["fpr_patient_mean"] for k in classes]))
        print(json.dumps({"epoch": ep, "train_loss": float(np.mean(L)), "val_mean_fpr": mean_fpr,
                          "val_lambda": lams, "elapsed_s": round(time.time() - t0)}), flush=True)
        if mean_fpr < best:
            best = mean_fpr
            torch.save(model.state_dict(), out / "best.pt")
        torch.save(model.state_dict(), out / "last.pt")


if __name__ == "__main__":
    main()
