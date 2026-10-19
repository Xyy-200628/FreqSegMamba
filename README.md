# FreqSegMamba

**Frequency-Prior-Guided Asymmetric Dual-Branch Visual State Space Network for Crack Segmentation**

FreqSegMamba is a topology-aware state space model for industrial structural crack segmentation. It addresses three challenges of applying Mamba/SSM to crack images: (1) weak crack signals drowned by strong background noise, (2) fine boundary degradation caused by exponential state decay of SSMs, and (3) topological discontinuity introduced by fixed-direction serialized scanning.

---

## Method Overview

The network consists of three synergistic pathways:

| Component | Role |
|---|---|
| **WAMamba backbone** | An isotropic visual state space encoder built on the **WAVSS** (Wavelet-Aware Visual State Space) module. WAVSS decomposes features into low-/high-frequency subbands via the discrete wavelet transform (DWT), processes the LL subband with the selective scanning engine, and enhances the LH/HL/HH detail subbands with bottleneck convolutions, before reconstructing via the inverse DWT. |
| **TPAS** | **Topology-Preserving Anisotropic Scanning.** Four SASS-anchored adaptive scan paths that *locally bend along the crack orientation field* so that physically adjacent crack pixels remain contiguous in the serialized sequence, preventing the SSM state decay from breaking crack topology. |
| **Freq-ADB** | **Frequency-Aware Adaptive Directional Branch** — an asymmetric detail branch composed of an Edge Prior Extractor (EPE), a Detail Pyramid (DP), and Cross-Feature Alignment Fusion (**CFAF**) modules, which injects explicit geometric edge priors into the backbone to sharpen crack boundaries. |

The total loss is a weighted combination of the main segmentation loss and a weakly supervised boundary loss:

```
L_total = 0.1 · L_BCE + 0.9 · L_Dice + 0.1 · L_Edge
```

The Dice-to-BCE ratio (9:1) and the boundary weight (λ = 0.1) were both determined by sensitivity-ablation experiments.

---

## Requirements

- Python 3.10
- PyTorch (CUDA build)
- [mamba-ssm](https://github.com/state-spaces/mamba)
- mmcv / mmengine
- timm, transformers
- opencv-python, numpy, tqdm

A conda environment is recommended. The code has been tested with Python 3.10.6 and PyTorch on NVIDIA A6000 GPUs.

---

## Datasets

Prepare the dataset in the following layout:

```
data/<DatasetName>/
├── train_img/    train_lab/
├── val_img/      val_lab/
└── test_img/     test_lab/
```

Supported datasets: **Crack500**, **CrackMap**, **DeepCrack**, **TUT**. Pass the dataset root via `--dataset_path`.

---

## Training

```bash
python main.py \
  --dataset_path /path/to/Crack500 \
  --epochs 50 \
  --seed 42 \
  --BCELoss_ratio 0.1 --DiceLoss_ratio 0.9 --edge_weight 0.1 \
  --tpas_mode adaptive4 \
  --tag my_run \
  --device cuda
```

Key defaults: AdamW optimizer, initial LR 5×10⁻⁴, PolyLR scheduler, batch size 1, input size 512×512, GroupNorm normalization. The best checkpoint (selected on the validation set) is saved to `checkpoints/weights/<timestamp>_<tag>_Dataset-><Dataset>/checkpoint_best.pth`.

## Evaluation

```bash
python final_test.py \
  --checkpoint checkpoints/weights/<dir>/checkpoint_best.pth \
  --dataset_path /path/to/Crack500 \
  --tpas_mode adaptive4 --no_adp \
  --device cuda
```

> **Important:** `--tpas_mode` and `--no_adp` must match the training configuration, otherwise the model architecture will not match the checkpoint.

The script prints `mIoU / ODS / OIS / F1 / Precision / Recall`.

## Single-Image Inference

```bash
python inference_one.py \
  --img /path/to/image.png \
  --checkpoint checkpoints/weights/<dir>/checkpoint_best.pth \
  --tpas_mode adaptive4 --no_adp \
  --device cuda
```

The binary segmentation mask (white crack on black background) is saved to `results/single_test/`.

---

## Ablation Experiments

Three ready-to-run scripts are provided:

| Script | Purpose |
|---|---|
| `run_component_ablation.sh` | Component ablation: baseline → +TPAS → +WAVSS → +Freq-ADB → full (5 rows) |
| `run_tpas_mode_ablation.sh` | TPAS scan-mode ablation |
| `run_loss_ablation.sh` | Loss-function ablation (Dice:BCE ratio + boundary weight λ) |

The ablation switches correspond to the three components:

| Switch | Effect |
|---|---|
| `--no_tpas` | Disable TPAS → fixed SASS scanning (baseline) |
| `--no_wavelet` | Disable WAVSS → full-resolution SSM scanning |
| `--no_adp` | Disable Freq-ADB (ADP-Stream + CFAF + boundary head) |

---

## Directory Structure

```
FreqSegMamba/
├── main.py               # training entry point
├── engine.py             # single-epoch training loop
├── test.py               # test-set evaluation
├── final_test.py         # final evaluation (prints mIoU/ODS/OIS/F1/P/R)
├── inference_one.py      # single-image inference
├── eval_compute.py       # FLOPs / parameter counting
├── models/               # model definitions
│   ├── decoder.py        # decoder + loss + model assembly
│   ├── tpas.py           # TPAS adaptive scanning
│   ├── edge_utils.py     # ADP-Stream + CFAF (edge-prior branch)
│   └── MFS.py / GBC.py / PAF.py / DySample.py / wavelet_utils.py
├── mmcls/WAVSS_dev/      # WAVSS backbone
├── datasets/             # data loading
├── util/                 # utility functions
├── eval/                 # evaluation metrics
└── run_*.sh              # ablation scripts
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{freQsegmamba,
  title={{FreqSegMamba}: Topology-Aware State Space Modeling for Crack Segmentation},
  author={},
  journal={IEEE Transactions on Industrial Informatics},
  year={2026},
  note={under review}
}
```

---

## License

See [LICENSE](LICENSE).
