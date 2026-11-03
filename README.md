# J-CRT: Joint Conformal Risk Training for Multi-class Segmentation

Official code for *J-CRT: Jointly Training a Single Network for Efficient Class-Wise Conformal Risk Control in
Multi-class Medical Image Segmentation*.

Post-hoc conformal risk control (CRC) can guarantee a per-class false-negative rate (FNR), but its calibrated thresholds
often produce many false positives. J-CRT fine-tunes a single softmax network so that each class has fewer false
positives **after** class-wise CRC at the target FNR.

## Method

Each training step simulates calibration on a mini-batch and penalizes the false positives its thresholds would produce.
For every foreground class $c$:

- $\lambda_c(\theta)$: differentiable CRC threshold computed on the $c$-positive slices of $D_\text{cal}$
- $L_{\text{FP},c}$: sigmoid surrogate of the voxel FPR of class $c$ on $D_\text{pred}$ at $\lambda_c(\theta)$

$$L(\theta) = \frac{1}{C}\sum_c L_{\text{FP},c}(\theta) + w\,[\text{CE}_\text{ls} + \text{Dice}]$$

Here $C$ is the number of foreground classes. Gradients flow through both the scores and the thresholds. At test time, the
thresholds come from ordinary patient-level class-wise CRC on a held-out calibration set, which gives marginal expected
FNR control for each class under exchangeability.

## Install

```bash
pip install -r requirements.txt
```

Requires PyTorch >= 2.0 and NumPy >= 1.24.

## Data format

Create one folder per split (`train`, `val`, `calib`, `test`), each containing:

| file | content |
|---|---|
| `images.npy` | float `[N, C, H, W]`, normalized 2D slices (H and W divisible by 16) |
| `masks.npy` | int `[N, H, W]`, labels `0..K-1` (0 = background) |
| `patients.txt` | `N` lines, the patient id of each slice |

Patients must not be shared between splits. `--num-classes` is `K`, the number of output channels including background.

## Usage

The examples use 4 input channels and 3 foreground classes (`--num-classes 4`).

### 1. Train the softmax baseline

```bash
python train_baseline.py --train data/train --val data/val --in-channels 4 --num-classes 4 --out runs/baseline
```

### 2. Fine-tune with J-CRT

Repeat with `--seed` for each training seed:

```bash
python train_jcrt.py --train data/train --val data/val --in-channels 4 --num-classes 4 \
    --base-ckpt runs/baseline/best.pt --out runs/jcrt_s0 --seed 0
```

To train without the threshold gradient (ablation), add `--detach-lambda`. $\lambda_c(\theta)$ is still recomputed at every
step but is detached.

### 3. Evaluate with repeated re-splitting (paper protocol)

```bash
python resplit_eval.py --calib data/calib --test data/test --in-channels 4 --num-classes 4 \
    --ckpt runs/jcrt_s0/best.pt runs/jcrt_s1/best.pt runs/jcrt_s2/best.pt \
    --alpha 0.05 0.1 0.2 --out jcrt_resplit.npz
```

The models stay fixed. Calibration and test patients are pooled and re-split 1,000 times with the original sizes. In each
split, class-wise CRC runs on the calibration part and patient-level FNR / FPR are measured on the test part.

- **Paired comparisons:** the same `--split-seed` and data give every method identical partitions.
- **Several checkpoints:** results are averaged per split, and the reported SD is over splits of this seed-mean.
- **Outputs:** `--out` writes `jcrt_resplit.npz` and `jcrt_resplit_summary.json`. In the `.npz`, each key `"<alpha>|<class>"`
  holds an array `[n_ckpt, n_splits, 3]` of (lambda, FNR, FPR).
- **Sanity check:** `--check` asserts that the results match `jcrt/crc.py` on the original split.

### Evaluate a single split

```bash
python calibrate_eval.py --calib data/calib --test data/test --in-channels 4 --num-classes 4 \
    --ckpt runs/jcrt_s0/best.pt --alpha 0.05 0.1 0.2
```

For post-hoc CRC, use `--ckpt runs/baseline/best.pt` with either evaluation script.

## Hyper-parameters

J-CRT defaults are defined in `jcrt/loss.py`, and all can be overridden from the command line.

| option | default | meaning |
|---|---|---|
| `--lr` | 1e-4 | Adam learning rate |
| `--epochs`, `--steps-per-epoch` | 50, 250 | fixed training budget |
| `--alpha` | 0.1 | target FNR during training and checkpoint selection |
| `--num-cal-per-class` | 12 | $k$: $c$-positive slices per class in $D_\text{cal}$ |
| `--num-pred` | 16 | $m$: slices in $D_\text{pred}$ |
| `--T` | 0.01 | temperature of the FP surrogate |
| `--w-anchor` | 1.0 | $w$: weight of the segmentation loss |
| `--label-smoothing` | 0.02 | label smoothing of the cross-entropy |

The best J-CRT checkpoint has the lowest mean validation FPR after class-wise CRC at `--alpha`. The baseline trains for 200
epochs (batch 16, cosine LR 1e-3 to 1e-6, gradient clipping 1.0), and its best checkpoint has the highest validation Dice.

### Feasibility limits

CRC at level $\alpha$ needs more than $1/\alpha - 1$ units per class (at least 20 at $\alpha = 0.05$):

- **Evaluation:** calibration patients containing the class. Infeasible classes are reported as `null` (`NaN` in re-splits).
- **Checkpoint selection:** validation patients containing the class. Training stops with an error otherwise.
- **Training:** slices containing the class in $D_\text{cal}$. The default $k = 12$ always satisfies this at $\alpha = 0.1$.
  For `--alpha 0.05`, use `--num-cal-per-class 20`.

Evaluating at several $\alpha$ does not require retraining.

## Datasets used in the paper

All splits are patient-level (seed 42) and stratified by class presence.

| dataset | modality | patients | foreground classes | train / val / calib / test | calib patients per class: min–max [original] |
|---|---|---|---|---|---|
| BraTS-METS | MRI (T1n, T1c, T2w, T2f) | 328 | NETC, SNFH, ET | 200 / 30 / 49 / 49 | 23–38 [34], 47–49 [48], 49 [49] |
| HNTS-MRG 2024 (pre-RT) | MRI (T2w) | 150 | GTVp, GTVn | 76 / 16 / 30 / 28 | 25–30 [27], 22–30 [26] |
| KiTS23 | CT | 489 | kidney, tumor, cyst | 294 / 49 / 73 / 73 | 73 [73], 73 [73], 27–46 [37] |
| BUS-BRA | ultrasound | 1,064 (1,875 images) | benign, malignant | 532 / 106 / 212 / 214 | 127–160 [144], 52–85 [68] |

Calibration patients per class count only the patients that contain the class, because CRC is undefined for the others.
The range is over the 1,000 re-splits, and the value in brackets is the original split. The feasibility limit at
$\alpha = 0.05$ holds in every re-split. HNTS-MRG and BUS-BRA use a 50/10/20/20 split so that the sparsest class keeps at
least 20 calibration patients.

### Preprocessing (2D slices)

- **BraTS-METS**: the 328 cases already in the canonical 240 x 240 x 155, 1 mm space; per case and sequence, z-score over
  the non-zero (skull-stripped brain) voxels; all axial slices containing brain; 240 x 240.
- **HNTS-MRG**: pre-RT T2w volume; per volume, intensities clipped to the [0.5, 99.5] percentiles and scaled to [-1, 1];
  all axial slices; padded to square and resized to 256 x 256.
- **KiTS23**: HU clipped to [-200, 300] and scaled to [-1, 1]; axial slices within +-20 mm of the labeled kidney region,
  sampled every ~3 mm; padded to square and resized to 256 x 256.
- **BUS-BRA**: intensities scaled to [-1, 1]; padded to square and resized to 256 x 256; the 1-2 views of a patient are
  pooled into one calibration unit.

## Repository structure

| file | content |
|---|---|
| `train_baseline.py` | trains the softmax baseline |
| `train_jcrt.py` | J-CRT fine-tuning and checkpoint selection |
| `resplit_eval.py` | repeated calibration/test re-splitting (paper protocol) |
| `calibrate_eval.py` | class-wise CRC and evaluation on a single split |
| `jcrt/loss.py` | differentiable threshold, FP surrogate, anchor loss, class-wise batch sampler, J-CRT objective |
| `jcrt/crc.py` | patient-level class-wise CRC and FNR / FPR / Dice evaluation |
| `jcrt/model.py` | 2D U-Net |
| `jcrt/data.py` | slice dataset for the format above |

This repository contains J-CRT, its threshold-gradient ablation, and post-hoc CRC. It does not include the independent
binary CRT and shared sigmoid CRT baselines, the other backbones, the preprocessing scripts, or the figure scripts.

## Acknowledgments

`get_lambda_theta` and `fpr_loss` follow the reference implementation of conformal risk training by Yeh et al.
([chrisyeh96/conformal-risk-training](https://github.com/chrisyeh96/conformal-risk-training)).

## License

MIT
