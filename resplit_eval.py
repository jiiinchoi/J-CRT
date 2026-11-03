"""Repeated calibration/test re-splitting (the evaluation protocol of the paper).

The trained models are fixed. The calibration and test patients are pooled and randomly re-split B times with the
original sizes; in each split class-wise CRC (jcrt/crc.py, Eq. (2)) is run on the calibration part and the
patient-mean FNR / FPR are measured on the test part. The splits depend only on --split-seed and the sorted patient
ids, so every method evaluated with the same seed and data sees identical partitions (paired comparison).

Several checkpoints (e.g. training seeds of one method) can be given: per split their results are averaged, and the
reported SD is over splits of this seed-mean, as in the paper's tables.

    python resplit_eval.py --calib data/calib --test data/test --in-channels 4 --num-classes 4 \
        --ckpt runs/jcrt_s0/best.pt runs/jcrt_s1/best.pt runs/jcrt_s2/best.pt --alpha 0.05 0.1 0.2 --out jcrt_resplit.npz

Implementation: one forward pass per checkpoint; per patient and class the fp16 scores of positive and negative voxels
are stored as histograms over all fp16 values in [0, 1], so each split only sums histograms. Thresholds and metrics are
identical to crc.calibrate / crc.evaluate on the same fp16 probabilities (checked on the original split with --check).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from jcrt import SliceDataset, UNet2D, calibrate, collect_predictions, evaluate

NBV = 15361  # fp16 bit patterns 0x0000 .. 0x3C00 (= 1.0): every possible softmax score
VALS = np.arange(NBV, dtype=np.uint16).view(np.float16).astype(np.float32)


@torch.no_grad()
def histograms(model, ds: SliceDataset, device, K: int, bs: int = 64):
    """sorted patient ids, pos / neg [P, K-1, NBV] counts of the fp16 score p_c on voxels with label == c / != c."""
    per = ds.patient_slices(); pids = sorted(per)
    pos = np.zeros((len(pids), K - 1, NBV), np.int64); neg = np.zeros_like(pos)
    model.eval()
    for i, pid in enumerate(pids):
        idx = np.asarray(per[pid])
        for s in range(0, len(idx), bs):
            img, msk = ds.get_batch(idx[s:s + bs])
            p = torch.softmax(model(img.to(device)).float(), 1)[:, 1:].half().cpu().numpy().view(np.uint16)
            g = msk.numpy()
            for k in range(K - 1):
                m = g == k + 1
                pos[i, k] += np.bincount(p[:, k][m], minlength=NBV)[:NBV]
                neg[i, k] += np.bincount(p[:, k][~m], minlength=NBV)[:NBV]
    return pids, pos, neg


def tables(pos, neg):
    """Per patient, class and candidate threshold VALS[b]: FNR (score < lam among positives), FPR (score >= lam among
    negatives)."""
    z = np.zeros(pos.shape[:2] + (1,), np.int64)
    pos_lt = np.concatenate([z, np.cumsum(pos, -1)[..., :-1]], -1)
    neg_lt = np.concatenate([z, np.cumsum(neg, -1)[..., :-1]], -1)
    npos, nneg = pos.sum(-1), neg.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        fnr = pos_lt / npos[..., None]
        fpr = (nneg[..., None] - neg_lt) / nneg[..., None]
    return {"npos": npos, "has": pos > 0, "fnr": fnr, "fpr": fpr}


def crc_index(T, ids, k, alpha):
    """Eq. (2) on the patients `ids` that contain class k: largest candidate (a positive score) with
    (N R + 1) / (N + 1) < alpha; None if alpha <= 1 / (N + 1)."""
    ids = ids[T["npos"][ids, k] > 0]
    N = len(ids)
    if alpha <= 1 / (N + 1):
        return None
    R = T["fnr"][ids, k].mean(0)
    feas = np.nonzero(((N * R + 1) / (N + 1) < alpha) & T["has"][ids, k].any(0))[0]
    return int(feas.max())


def metrics(T, ids, k, b):
    present = ids[T["npos"][ids, k] > 0]
    return float(VALS[b]), float(T["fnr"][present, k, b].mean()), float(T["fpr"][ids, k, b].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--in-channels", type=int, required=True); ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--ckpt", nargs="+", required=True, help="one or more checkpoints of the same method (e.g. seeds)")
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ap.add_argument("--n-splits", type=int, default=1000)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="optional .npz with the per-split results (for paired comparisons)")
    ap.add_argument("--check", action="store_true",
                    help="also run crc.calibrate / crc.evaluate on the original split and assert identical results")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K, classes = a.num_classes, list(range(1, a.num_classes))
    cal_ds, test_ds = SliceDataset(a.calib, in_ram=False), SliceDataset(a.test, in_ram=False)

    tabs, pids = [], None
    for ck in a.ckpt:
        model = UNet2D(a.in_channels, K)
        model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True)); model.to(dev)
        cp, cpos, cneg = histograms(model, cal_ds, dev, K)
        tp, tpos, tneg = histograms(model, test_ds, dev, K)
        if set(cp) & set(tp):
            raise ValueError("calibration and test splits share patients")
        T = tables(np.concatenate([cpos, tpos]), np.concatenate([cneg, tneg]))
        nc, P = len(cp), len(cp) + len(tp)
        if a.check:  # the histogram path must reproduce crc.py on the original split
            cal0, test0 = np.arange(nc), np.arange(nc, P)
            lams = calibrate(collect_predictions(model, cal_ds, dev), classes, {k: a.alpha[0] for k in classes})
            ref = evaluate(collect_predictions(model, test_ds, dev), lams)
            for k in classes:
                b = crc_index(T, cal0, k - 1, a.alpha[0])
                if b is None:
                    assert ref[k] is None; continue
                lam, fnr, fpr = metrics(T, test0, k - 1, b)
                assert lam == ref[k]["lambda"] and abs(fnr - ref[k]["fnr_patient_mean"]) < 1e-9 \
                    and abs(fpr - ref[k]["fpr_patient_mean"]) < 1e-9, (ck, k, (lam, fnr, fpr), ref[k])
            print(f"check passed: {ck}", flush=True)
        if pids is not None and pids != cp + tp:
            raise ValueError("all checkpoints must be evaluated on the same calibration/test data")
        tabs.append(T); pids = cp + tp
        del model; torch.cuda.empty_cache()

    rng = np.random.default_rng(a.split_seed)
    perms = np.stack([rng.permutation(P) for _ in range(a.n_splits)])
    # res[alpha][class]: [n_ckpt, n_splits, 3] = (lambda, FNR, FPR); NaN where CRC is infeasible
    res = {al: {k: np.full((len(tabs), a.n_splits, 3), np.nan) for k in classes} for al in a.alpha}
    for s, perm in enumerate(perms):
        cal, tst = perm[:nc], perm[nc:]
        for j, T in enumerate(tabs):
            for k in classes:
                for al in a.alpha:
                    b = crc_index(T, cal, k - 1, al)
                    if b is not None:
                        res[al][k][j, s] = metrics(T, tst, k - 1, b)

    summary = {}
    for al in a.alpha:
        summary[str(al)] = {}
        for k in classes:
            x = np.nanmean(res[al][k], 0)                       # seed-mean per split
            ck_fnr = np.nanmean(res[al][k][..., 1], 1)          # mean FNR of each checkpoint
            seed_means = np.nanmean(res[al][k][..., 2], 1)      # mean FPR of each checkpoint
            summary[str(al)][k] = {
                "fnr_mean": float(np.nanmean(x[:, 1])), "fnr_range_over_ckpts": [float(ck_fnr.min()), float(ck_fnr.max())],
                "fpr_mean": float(np.nanmean(x[:, 2])), "fpr_sd_over_splits": float(np.nanstd(x[:, 2])),
                "fpr_sd_over_ckpts": float(np.std(seed_means, ddof=1)) if len(tabs) > 1 else None,
                "frac_splits_fnr_gt_alpha": float(np.nanmean(x[:, 1] > al)),
                "n_feasible_splits": int((~np.isnan(x[:, 1])).sum())}
            v = summary[str(al)][k]
            print(f"alpha={al:<5g} class {k}: FNR {v['fnr_mean']:.3f}  FPR {v['fpr_mean']:.4f} +- {v['fpr_sd_over_splits']:.4f}"
                  f"  (feasible {v['n_feasible_splits']}/{a.n_splits})", flush=True)
    if a.out:
        np.savez_compressed(a.out, perms=perms, patients=np.array(pids), n_calib=nc,
                            **{f"{al:g}|{k}": res[al][k] for al in a.alpha for k in classes})
        with open(a.out.rsplit(".", 1)[0] + "_summary.json", "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
