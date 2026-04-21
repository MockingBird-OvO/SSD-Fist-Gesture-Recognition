# Realtime CV — SSD Gesture Detection

A real-time object detection system built on a custom **SSD (Single Shot MultiBox Detector)** architecture, trained with **5-fold cross-validation** and deployed as a live camera inference pipeline. The project currently targets hand gesture recognition (specifically the "Fist" class) but is fully configurable for any COCO-format dataset.

---

## Features

- Custom lightweight SSD with a 4-stage feature pyramid backbone
- 5-fold cross-validation training with per-fold best model checkpointing
- Ensemble inference across all fold models for improved robustness
- Rich data augmentation pipeline (photometric distortion, flips, rotation, blur)
- Hard negative mining with focal-style loss weighting
- Real-time camera detection using OpenCV

---

## Project Structure

```
code/
├── train.py                  # K-fold training entry point
├── predict.py                # Real-time camera inference (ensemble)
├── requirements.txt
│
├── _00_config/
│   └── config.py             # All hyperparameters and paths
│
├── _01_data/
│   ├── dataloader.py         # COCO dataset loader with augmentation & caching
│   ├── labels.json           # Class label definitions
│   └── result_0.json         # COCO-format annotation file
│
├── _02_utils/
│   ├── box_ops.py            # NMS and bounding box utilities
│   ├── encoder.py            # Box encode/decode (variances)
│   └── matcher.py            # Anchor-to-ground-truth IoU matching
│
├── _03_model/
│   ├── model.py              # LargeScaleSSD network definition
│   ├── loss.py               # MultiBoxLoss (localization + classification)
│   └── anchorGenerator.py    # Prior/anchor box generation
│
├── _04_training/
│   ├── models/               # Saved model weights (best_N.pth, latest_N.pth)
│   └── loss/                 # Training loss plots per fold
│
└── _05_inference/
    └── *.png                 # Sample inference output images
```

---

## Installation

**Python 3.9+ recommended.**

```bash
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| `torch >= 2.0.0` | Core deep learning framework |
| `torchvision >= 0.15.0` | Image transforms (`v2` API) |
| `opencv-python >= 4.8.0` | Camera capture and visualization |
| `Pillow >= 10.0.0` | Image loading |
| `pycocotools >= 2.0.7` | COCO annotation parsing |
| `numpy`, `matplotlib`, `tqdm` | Utilities and training plots |

---

## Configuration

All settings are centralized in `_00_config/config.py`. Key parameters:

| Setting | Default | Description |
|---|---|---|
| `IMAGE_ROOT` | *(set your path)* | Root directory of training images |
| `ANN_FILE` | `_01_data/result_0.json` | COCO-format annotation file |
| `NUM_CLASSES` | `2` | Number of classes (excluding background) |
| `INPUT_SIZE` | `(480, 640)` | Model input resolution (H × W) |
| `K_FOLD` | `5` | Number of cross-validation folds |
| `EPOCHS` | `40` | Training epochs per fold |
| `BATCH_SIZE` | `32` | Batch size |
| `LR` | `1e-3` | Initial learning rate (Adam) |
| `AUG_FACTOR` | `8` | Augmentation multiplier for training set |

**Before training**, update `IMAGE_ROOT` in `config.py` to point to your local image directory.

---

## Usage

### Training

```bash
python train.py
```

This runs 5-fold cross-validation. For each fold:
- Trains for up to 40 epochs with early stopping (patience = 8)
- Saves `best_N.pth` (lowest actual validation loss in final 20 epochs)
- Saves `latest_N.pth` (final epoch weights)
- Outputs a loss curve plot to `_04_training/loss/loss_N.png`

### Real-Time Inference

```bash
python predict.py
```

Loads all 5 best-fold models as an ensemble and opens the default camera. Detection boxes and confidence scores are rendered live. Press **`q`** to quit.

---

## Model Architecture

`LargeScaleSSD` is a 4-scale single-shot detector with a custom lightweight backbone:

```
Input [B, 3, 480, 640]
  │
  ├─ Stage 1 → [B, 48, 120, 160]   (1/4 scale,  f1)
  ├─ Stage 2 → [B, 96,  60,  80]   (1/8 scale,  f2)
  ├─ Stage 3 → [B, 192, 30,  40]   (1/16 scale, f3)
  └─ Stage 4 → [B, 192, 15,  20]   (1/32 scale, f4)
         │
  Per scale: regression head + classification head
         │
  Outputs: loc_data [B, total_anchors, 4]
           conf_data [B, total_anchors, num_classes]
```

Each detection head consists of two feature-extraction conv layers followed by an output conv, with Dropout and BatchNorm throughout. Weights are initialized with Xavier uniform; classification biases are set using a prior probability of 0.01 to prevent early training collapse.

---

## Training Details

- **Optimizer**: Adam with weight decay `1e-4`
- **LR Schedule**: MultiStepLR, decayed at epochs 20 and 32 (gamma = 0.3)
- **Loss**: MultiBoxLoss = localization (SmoothL1) + classification (cross-entropy with hard negative mining, neg:pos ratio = 3:1)
- **Anchor matching**: IoU threshold = 0.5
- **Gradient clipping**: max norm = 5.0
- **Cache refresh**: augmented training cache regenerated every 20 epochs

---

## Inference Details

At inference time, all 5 fold models run in parallel (ensemble). Their anchor predictions are concatenated before a single NMS pass:

1. Softmax over class logits
2. Threshold by confidence (`CONF_THRESHOLD = 0.5`)
3. Decode predicted offsets into absolute boxes
4. NMS with IoU threshold `0.5`, keeping up to 100 boxes

---

## Data Format

Annotations must follow the **COCO JSON format**. Labels are defined in `_01_data/labels.json`:

```json
[
  { "id": 0, "name": "Fist" }
]
```

Class IDs in the label file are 0-indexed; the model internally reserves index 0 for background and shifts all labels by +1.

---

## Output Files

| Path | Description |
|---|---|
| `_04_training/models/best_N.pth` | Best checkpoint for fold N |
| `_04_training/models/latest_N.pth` | Final epoch checkpoint for fold N |
| `_04_training/loss/loss_N.png` | Loss curves (total + components) for fold N |