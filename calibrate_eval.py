"""Post-hoc class-wise CRC: calibrate one threshold per class on the calibration split, then report FNR / FPR /
Dice on the test split. Works for any checkpoint (baseline or J-CRT).

    python calibrate_eval.py --calib data/calib --test data/test --in-channels 4 --num-classes 4 \
        --ckpt runs/jcrt/best.pt --alpha 0.1
"""

from __future__ import annotations

import argparse
import json

import torch

from jcrt import SliceDataset, UNet2D, calibrate, collect_predictions, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--in-channels", type=int, required=True); ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.1])
    ap.add_argument("--out", default=None, help="optional JSON output path")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2D(a.in_channels, a.num_classes)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True))
    model.to(dev)
    classes = list(range(1, a.num_classes))
    cal = collect_predictions(model, SliceDataset(a.calib), dev)
    test = collect_predictions(model, SliceDataset(a.test), dev)
    results = {}
    for alpha in a.alpha:
        lams = calibrate(cal, classes, {k: alpha for k in classes})
        results[str(alpha)] = evaluate(test, lams)
    print(json.dumps(results, indent=1))
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
