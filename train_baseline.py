"""Train the softmax baseline that J-CRT starts from.

CE(label smoothing 0.02) + soft Dice, Adam, batch 16, cosine LR 1e-3 -> 1e-6 per step, grad clip 1.0, 200 epochs.
Checkpoint selection on the validation split: max mean (over foreground classes) pooled argmax Dice.

    python train_baseline.py --train data/train --val data/val --in-channels 4 --num-classes 4 --out runs/baseline
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from jcrt import SliceDataset, UNet2D, anchor_loss

CFG = {"label_smoothing": 0.02, "lr": 1e-3, "lr_min": 1e-6, "batch_size": 16, "epochs": 200, "grad_clip_norm": 1.0,
       "seed": 0}


@torch.no_grad()
def val_dice(model, ds: SliceDataset, device, num_classes: int, batch_size: int = 64) -> float:
    model.eval()
    inter = np.zeros(num_classes); pv = np.zeros(num_classes); gv = np.zeros(num_classes)
    for s in range(0, len(ds), batch_size):
        img, msk = ds.get_batch(np.arange(s, min(s + batch_size, len(ds))))
        am = model(img.to(device)).argmax(1).cpu()
        for k in range(1, num_classes):
            p, g = am == k, msk == k
            inter[k] += (p & g).sum().item(); pv[k] += p.sum().item(); gv[k] += g.sum().item()
    return float(np.mean([2 * inter[k] / max(pv[k] + gv[k], 1) for k in range(1, num_classes)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--val", required=True)
    ap.add_argument("--in-channels", type=int, required=True); ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=CFG["epochs"])
    a = ap.parse_args()
    cfg = dict(CFG, epochs=a.epochs)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"]); rng = np.random.default_rng(cfg["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va = SliceDataset(a.train), SliceDataset(a.val)
    model = UNet2D(a.in_channels, a.num_classes).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    spe = len(tr) // cfg["batch_size"]
    total = cfg["epochs"] * spe
    lr_at = lambda t: cfg["lr_min"] + 0.5 * (cfg["lr"] - cfg["lr_min"]) * (1 + math.cos(math.pi * t / total))
    best, step, t0 = -1.0, 0, time.time()
    for ep in range(cfg["epochs"]):
        model.train(); perm = rng.permutation(len(tr)); L = []
        for s in range(spe):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            img, msk = tr.get_batch(perm[s * cfg["batch_size"]:(s + 1) * cfg["batch_size"]])
            loss = anchor_loss(model(img.to(dev)), msk.to(dev), cfg["label_smoothing"])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"]); opt.step()
            L.append(loss.item()); step += 1
        d = val_dice(model, va, dev, a.num_classes)
        print(json.dumps({"epoch": ep, "train_loss": float(np.mean(L)), "val_mean_dice": d,
                          "elapsed_s": round(time.time() - t0)}), flush=True)
        if d > best:
            best = d
            torch.save(model.state_dict(), out / "best.pt")
        torch.save(model.state_dict(), out / "last.pt")


if __name__ == "__main__":
    main()
