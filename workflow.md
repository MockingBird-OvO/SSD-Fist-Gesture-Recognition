# Project Workflow: Realtime CV Object Detection

## Overview
This is an SSD-based object detection system with k-fold cross-validation training and real-time inference via camera feed.

---

## 1. Data Pipeline

### Data Loading & Augmentation
```
get_k_fold_data(ann_file, image_root, batch_size, k, fold)
    ↓
COCOImageDataset(coco, img_ids, image_root, is_train, aug_factor)
    ↓
├─ _read_raw(img_id)
│   ├─ Load image from disk
│   ├─ Load COCO annotations
│   ├─ Extract bboxes [x, y, x+w, y+h]
│   └─ Extract class labels
    ↓
refresh_cache(aug_factor=8)
    ├─ Apply random photometric distortion (prob=0.8)
    ├─ Apply random horizontal flip (prob=0.5)
    ├─ Resize to INPUT_SIZE (480, 640)
    ├─ Normalize with ImageNet stats
    └─ Generate 8x augmented versions of training data
    ↓
DataLoader(batch_size=32)
    └─ Returns: (images, targets) per batch
```

---

## 2. Model Architecture & Forward Pass

### LargeScaleSSD Network
```
Input Image [B, 3, 480, 640]
    ↓
stage1: Conv blocks + MaxPool → [B, 48, 120, 160]  (f1)
    ↓
stage2: Conv stride=2 → [B, 96,  60,  80]  (f2)
    ↓
stage3: Conv stride=2 → [B, 192, 30,  40]  (f3)
    ↓
stage4: Conv stride=2 → [B, 192, 15,  20]  (f4)
    ↓
┌─────────────────────────────────────────┐
│ For each feature map f1, f2, f3, f4:    │
├─────────────────────────────────────────┤
│ Localization Head (reg_heads[i])        │
│   → outputs: [B, num_anchors, 4]        │
│   (∆cx, ∆cy, ∆w, ∆h offsets)           │
│                                         │
│ Classification Head (cls_heads[i])      │
│   → outputs: [B, num_anchors, C]        │
│   (logits for each class)               │
└─────────────────────────────────────────┘
    ↓
Concatenate all scales
    ↓
Returns: (loc_data, conf_data)
```

---

## 3. Training Pipeline

```
for fold in range(K_FOLD):
    ↓
    train_loader, val_loader = get_k_fold_data()
        ↓
    for epoch in range(EPOCHS):
        ↓
        ┌─────────────────────────────────────────┐
        │ TRAINING PHASE                          │
        ├─────────────────────────────────────────┤
        │ for images, targets in train_loader:    │
        │   ↓                                     │
        │   out = net(images)  ← Forward pass    │
        │   ↓                                     │
        │   loss_l, loss_c, total_loss =          │
        │     criterion(out, targets, priors)    │
        │   ↓                                     │
        │   total_loss.backward()                │
        │   ↓                                     │
        │   clip_grad_norm_(parameters)          │
        │   ↓                                     │
        │   optimizer.step()                      │
        └─────────────────────────────────────────┘
        ↓
        ┌─────────────────────────────────────────┐
        │ VALIDATION PHASE                        │
        ├─────────────────────────────────────────┤
        │ for images, targets in val_loader:      │
        │   ↓                                     │
        │   out = net(images)  ← Forward pass    │
        │   ↓                                     │
        │   loss_l, loss_c = criterion(...)      │
        │   ↓                                     │
        │   track validation loss                │
        └─────────────────────────────────────────┘
        ↓
        if val_loss < best_val_loss:
            ↓
            save_model(net.state_dict())
        ↓
        scheduler.step()  ← Update learning rate
    ↓
    Clear GPU cache for next fold
```

### Loss Computation
```
MultiBoxLoss(out, targets, priors):
    ├─ Match predictions to ground truth
    │   ├─ jaccard(predicted_boxes, gt_boxes) → IoU matrix
    │   ├─ Find best IoU matches (overlap ≥ 0.5)
    │   └─ Assign class labels & localization targets
    │
    ├─ Localization Loss (Smooth L1)
    │   └─ Regress [∆cx, ∆cy, ∆w, ∆h] for matched boxes
    │
    ├─ Classification Loss (CrossEntropy)
    │   ├─ Hard negative mining (NEG_RATIO=3)
    │   └─ Balance foreground vs background
    │
    └─ return: (loss_l, loss_c, total_loss)
```

---

## 4. Inference Pipeline

### Setup
```
camera_detect():
    ↓
    Load config from Config class
    ↓
    Load label_map from JSON
    ↓
    Initialize LargeScaleSSD(num_classes=2)
    ↓
    net.load_state_dict(torch.load(WEIGHT_PATH))
    ↓
    net.eval()  ← Set to evaluation mode
    ↓
    priors = create_prior_boxes()
```

### Per-Frame Inference

```
while camera feed:
    ↓
    ret, frame = cap.read()  ← Raw BGR frame (480, 640, 3)
    ↓
    ┌─────────────────────────────────────────┐
    │ PREPROCESSING                           │
    ├─────────────────────────────────────────┤
    │ 1. Convert BGR → RGB                    │
    │ 2. PIL.Image conversion                 │
    │ 3. tv_tensors.Image wrapping            │
    │ 4. Apply eval_transform:                │
    │    ├─ Resize(480, 640)                  │
    │    ├─ ToImage()                         │
    │    ├─ ToDtype(float32, scale=True)      │
    │    └─ Normalize(ImageNet stats)         │
    │ 5. Unsqueeze batch dimension            │
    │ 6. Move to device (GPU)                 │
    └─────────────────────────────────────────┘
    ↓
    ┌─────────────────────────────────────────┐
    │ ENSEMBLE INFERENCE                      │
    ├─────────────────────────────────────────┤
    │ for each model in ensemble:             │
    │   loc_i, conf_i = model(img)            │
    │   append to all_locs, all_confs         │
    │                                         │
    │ loc_data  = cat(all_locs,  dim=1)       │
    │ conf_data = cat(all_confs, dim=1)       │
    │                                         │
    │ Output shapes:                          │
    │ - loc_data:  [1, N*num_priors, 4]       │
    │ - conf_data: [1, N*num_priors, 2]       │
    └─────────────────────────────────────────┘
    ↓
    ┌─────────────────────────────────────────┐
    │ POST-PROCESSING (fully vectorized)      │
    ├─────────────────────────────────────────┤
    │                                         │
    │ 1. conf_preds = softmax(conf_data[0])   │
    │    → [N*num_priors, num_classes]        │
    │    loc_preds = loc_data[0]              │
    │    → [N*num_priors, 4]                  │
    │                                         │
    │ 2. Vectorized confidence filter:        │
    │    scores_all, cls_ids_all =            │
    │      conf_preds[:, 1:].max(dim=1)       │
    │    ├─ scores_all:  [N*num_priors]       │
    │    │   best non-background score        │
    │    │   per anchor                       │
    │    └─ cls_ids_all: [N*num_priors]       │
    │        corresponding class (0-indexed)  │
    │                                         │
    │ 3. Boolean mask threshold filter:       │
    │    mask = scores_all > CONF_THRESHOLD   │
    │    filtered_locs   = loc_preds[mask]    │
    │    filtered_priors = priors_expanded    │
    │                          [mask]         │
    │    filtered_scores = scores_all[mask]   │
    │    filtered_cls    = cls_ids_all[mask]  │
    │                              + 1        │
    │    (all ops stay on GPU, no Python loop)│
    │                                         │
    │ 4. Decode filtered anchors only:        │
    │    filtered_boxes = decode(             │
    │      filtered_locs,                     │
    │      filtered_priors, VARIANCES)        │
    │    ├─ Apply ∆cx,∆cy,∆w,∆h offsets      │
    │    ├─ Scale back to [0,1] normalized    │
    │    └─ Output: [K, 4] (xyxy)            │
    │    (K << N*num_priors after masking)    │
    │                                         │
    │ 5. NMS on K candidates:                 │
    │    keep = nms(filtered_boxes,           │
    │               filtered_scores,          │
    │               overlap, top_k)           │
    │    └─ List: [(box, score, cls_id)...]   │
    │                                         │
    │ 6. Draw boxes on frame                  │
    │    └─ Assign color per class            │
    └─────────────────────────────────────────┘
    ↓
    Display on screen
```

## 5. Utility Functions Map

### Box Operations
```
box_ops.py:
├─ intersect(box_a, box_b)
│   │  Input: box_a [N, 4], box_b [M, 4] (xyxy format)
│   │  Process: Compute intersection area
│   └─ Output: [N, M] intersection areas
│
├─ jaccard(box_a, box_b)
│   │  Input: box_a [N, 4], box_b [M, 4] (xyxy format)
│   │  Process: IoU = Intersection / Union
│   └─ Output: [N, M] IoU values
│       (Used in: Loss function matching)
│
├─ point_form(boxes)
│   │  Input: boxes [N, 4] (cx, cy, w, h)
│   └─ Output: boxes [N, 4] (xmin, ymin, xmax, ymax)
│       (Used in: encode/decode operations)
│
├─ center_size(boxes)
│   │  Input: boxes [N, 4] (xmin, ymin, xmax, ymax)
│   └─ Output: boxes [N, 4] (cx, cy, w, h)
│       (Used in: encode/decode operations)
│
└─ nms(boxes, scores, overlap=0.5, top_k=200)
    │  Input: boxes [num_priors, 4] (xyxy)
    │         scores [num_priors]
    │         overlap threshold, max boxes to keep
    │  Process:
    │  ├─ Sort boxes by score (descending)
    │  ├─ Iteratively select highest-scoring box
    │  ├─ Remove boxes with IoU > overlap_threshold
    │  └─ Repeat until none remain
    └─ Output: keep indices of surviving boxes
        (Used in: Inference post-processing)
```

### Model Components
```
model.py:
├─ LargeScaleSSD(num_classes)
│   │  __init__:
│   │  ├─ Build 4-stage feature extraction
│   │  ├─ Build classification heads × 4
│   │  ├─ Build localization heads × 4
│   │  ├─ Xavier weight initialization
│   │  └─ Background bias initialization
│   │
│   └─ forward(x) → (loc_data, conf_data)
│
└─ create_prior_boxes()
    │  Input: Config anchors (scales, aspect ratios, feature maps)
    │  Process:
    │  ├─ For each feature map (4 scales)
    │  ├─ Generate anchor boxes at each spatial location
    │  ├─ Apply aspect ratios and scales
    │  └─ Normalize to [0, 1] range
    └─ Output: [num_total_priors, 4] (cx, cy, w, h)
        (Used in: Training loss, decoding predictions)
```

### Encoding/Decoding
```
encoder.py:
├─ encode(matched_boxes, priors)
│   │  Input: matched_boxes [N, 4], priors [N, 4]
│   │  Process: Convert box format to offsets
│   │  ├─ Convert both to center-size format
│   │  ├─ Compute: (∆cx, ∆cy, ∆w, ∆h)
│   │  ├─ Scale offsets by VARIANCES [0.1, 0.2]
│   │  └─ Clamp values for stability
│   └─ Output: offsets [N, 4]
│       (Used in: Training targets)
│
└─ decode(offsets, priors, variances)
    │  Input: offsets [N, 4], priors [N, 4], variances
    │  Process: Reverse encoding operation
    │  ├─ Unscale by VARIANCES
    │  ├─ Apply offsets to prior boxes
    │  ├─ Clamp w, h to avoid invalid boxes
    │  └─ Normalize to [0, 1] range
    └─ Output: predicted_boxes [N, 4]
        (Used in: Inference post-processing)
```

### Matching
```
matcher.py:
└─ Matcher(overlap_threshold=0.5)
    │  forward(priors, targets):
    │  ├─ jaccard(priors, targets) → IoU matrix
    │  ├─ Find best IoU for each prior
    │  ├─ Assign positive/negative labels
    │  └─ Return matched targets & confidence
    └─ Output: (matched_boxes, matched_conf)
        (Used in: Loss computation)
```

---

## 6. Configuration Control

All hyperparameters centralized in `config.py`:


---

## 7. Complete End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING MODE                              │
│                                                                  │
│  config.py                                                       │
│    ↓                                                              │
│  train.py                                                        │
│    ├─→ create_prior_boxes() [from model.py]                    │
│    ├─→ MultiBoxLoss(num_classes) [from model.py]               │
│    ├─→ get_k_fold_data() [from dataloader.py]                  │
│    │   ├─→ COCOImageDataset()                                  │
│    │   │   ├─→ _read_raw()                                     │
│    │   │   └─→ refresh_cache() [augmentation]                  │
│    │   └─→ DataLoader(batch_size)                              │
│    │                                                             │
│    └─→ for fold, epoch:                                         │
│        ├─→ LargeScaleSSD() forward                             │
│        ├─→ Matcher() [from matcher.py]                         │
│        │   └─→ jaccard() [from box_ops.py] (IoU matching)     │
│        ├─→ encode() [from encoder.py]                          │
│        ├─→ MultiBoxLoss backward                               │
│        └─→ save best_1.pth                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     INFERENCE MODE                              │
│                                                                  │
│  config.py                                                       │
│    ↓                                                             │
│  predict.py → camera_detect()                                   │
│    ├─→ Load ensemble models from WEIGHT_PATHS                   │
│    ├─→ create_prior_boxes() → priors_expanded                   │
│    │   └─→ priors.repeat(num_models, 1)                         │
│    ├─→ Load label_map from labels.json                          │
│    │                                                             │
│    └─→ while camera:                                            │
│        ├─→ Preprocess frame:                                    │
│        │   └─→ v2.Compose transforms                            │
│        ├─→ ensemble_forward() → (loc_data, conf_data)           │
│        │   └─→ cat all models' outputs on anchor dim            │
│        ├─→ softmax(conf_data) → conf_preds                      │
│        ├─→ conf_preds[:,1:].max(dim=1) → scores, cls_ids        │
│        ├─→ boolean mask > CONF_THRESHOLD                        │
│        ├─→ decode() on masked anchors only                      │
│        │   └─→ convert offsets to absolute boxes               │
│        ├─→ nms(filtered_boxes, filtered_scores)                 │
│        └─→ Draw & Display                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---



## 8. Dependencies Flow

```
┌─ train.py
│  ├─ config.py
│  ├─ _03_model/model.py (LargeScaleSSD, MultiBoxLoss, create_prior_boxes)
│  ├─ _01_data/dataloader.py (get_k_fold_data, COCOImageDataset)
│  └─ _02_utils/
│     ├─ matcher.py (matching logic)
│     ├─ encoder.py (encode/decode)
│     └─ box_ops.py (jaccard, intersect)
│
├─ predict.py
│  ├─ config.py
│  ├─ _03_model/model.py
│  ├─ _02_utils/box_ops.py (nms, decode)
│  └─ labels.json
│
└─ config.py
   ├─ _04_training/models/best_1.pth (saved weights)
   ├─ _01_data/result_0.json (annotations)
   ├─ _01_data/labels.json (class labels)
   └─ Image dataset directory
```