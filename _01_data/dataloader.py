import os
from typing import List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from torchvision.transforms import v2
from torchvision import tv_tensors
from _00_config.config import Config
import gc

class COCOImageDataset(Dataset):
    def __init__(self, coco: COCO, img_ids: List[int], image_root: str, is_train: bool = True, aug_factor: int = None):
        self.coco = coco
        self.img_ids = img_ids
        self.image_root = image_root
        self.is_train = is_train
        self.aug_factor = aug_factor if aug_factor is not None else Config.AUG_FACTOR  # Save factor

        cat_ids = sorted(self.coco.getCatIds())
        self.cat_to_idx = {cat_id: i for i, cat_id in enumerate(cat_ids)}

        # Define v2 transformation pipeline (keep unchanged)
        if is_train:
            self.transforms = v2.Compose([
                v2.RandomPhotometricDistort(p=Config.AUGMENTATION_PHOTOMETRIC_P),
                #v2.RandomGrayscale(p=Config.AUGMENTATION_GRAYSCALE_P),
                v2.GaussianBlur(kernel_size=Config.AUGMENTATION_BLUR_KERNEL_SIZE, sigma=Config.AUGMENTATION_BLUR_SIGMA),
                #v2.RandomZoomOut(fill=Config.AUGMENTATION_ZOOMOUT_FILL, p=Config.AUGMENTATION_ZOOMOUT_P, side_range=Config.AUGMENTATION_ZOOMOUT_SIDE_RANGE),
                #v2.RandomIoUCrop(),
                v2.RandomHorizontalFlip(p=Config.AUGMENTATION_HFLIP_P),
                v2.RandomVerticalFlip(p=Config.AUGMENTATION_VFLIP_P),
                v2.RandomRotation(degrees=Config.AUGMENTATION_ROTATION_DEGREES),
                v2.Resize(size=Config.INPUT_SIZE),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=Config.NORMALIZATION_MEAN, std=Config.NORMALIZATION_STD)
            ])
        else:
            self.transforms = v2.Compose([
                v2.Resize(size=Config.INPUT_SIZE),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=Config.NORMALIZATION_MEAN, std=Config.NORMALIZATION_STD)
            ])

        # 1. Raw data cache
        self.raw_cache = []
        print(f"Loading raw images to memory...")
        for img_id in self.img_ids:
            self.raw_cache.append(self._read_raw(img_id))
            
        # 2. Training tensor cache
        self.cached_images = None
        self.cached_targets = None
        gc.collect()
        self.refresh_cache()

    def _read_raw(self, img_id):
        img_info = self.coco.loadImgs(img_id)[0]
        relative_path = img_info['file_name'].split('images/')[-1]
        img_path = os.path.join(self.image_root, relative_path)
        img = Image.open(img_path).convert('RGB')
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_to_idx[ann['category_id']] + 1)
        
        return img, torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4), torch.as_tensor(labels, dtype=torch.int64)

    def refresh_cache(self, pbar=None):
        """Key method: Called once every 20 epochs, generates aug_factor times more data"""
        self.cached_images = None
        self.cached_targets = None
        if self.is_train:
            msg = f"Refresh memory augmented data (factor: {self.aug_factor})..."
            if pbar is not None:
                pbar.write(msg)
            else:
                print(msg)
        
        tmp_imgs = []
        tmp_targets = []
        
        # Only training set needs multi-fold augmentation, validation set needs 1 fold
        actual_factor = self.aug_factor if self.is_train else 1
        
        for _ in range(actual_factor):
            for img, boxes, labels in self.raw_cache:
                t_img = tv_tensors.Image(img)
                target = {
                    "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=t_img.shape[-2:]),
                    "labels": labels
                }
                
                if self.transforms:
                    t_img, target = self.transforms(t_img, target)
                
                if target["boxes"].shape[0] == 0:
                    target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                    target["labels"] = torch.zeros((0,), dtype=torch.int64)
                
                _, h, w = t_img.shape
                boxes_norm = target["boxes"] / torch.tensor([w, h, w, h], dtype=torch.float32)
                final_target = torch.cat([boxes_norm, target["labels"].unsqueeze(1).to(torch.float32)], dim=1)
                
                tmp_imgs.append(t_img)
                tmp_targets.append(final_target)
            
        self.cached_images = torch.stack(tmp_imgs)
        self.cached_targets = tmp_targets
        if self.is_train:
            msg=f"Refresh complete! Current single-round sample count: {len(tmp_imgs)}"
            if pbar is not None:
                pbar.write(msg)
            else:
                print(msg)
        gc.collect()

    def __len__(self):
        return len(self.cached_targets) if self.cached_targets is not None else 0

    def __getitem__(self, idx):
        return self.cached_images[idx], self.cached_targets[idx]


def collate_fn(batch): #ensures different number of boxes per image can be handled
    return tuple(zip(*batch))

def get_k_fold_data(ann_file: str, image_root: str,
                    batch_size: int = 32, num_workers: int = 0, # Data in memory, worker set to 0 is faster
                    k: int = 5, i: int = 0):
    
    if k <= 1:
        raise ValueError("k must be > 1 for k-fold cross validation")
    if not (0 <= i < k):
        raise ValueError("fold index i must satisfy 0 <= i < k")

    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    n = len(img_ids)

    fold_size = n // k
    val_start = i * fold_size
    val_end = n if i == k - 1 else (i + 1) * fold_size

    val_ids = img_ids[val_start:val_end]
    train_ids = img_ids[:val_start] + img_ids[val_end:]

    ds_train = COCOImageDataset(coco, train_ids, image_root, is_train=True)
    ds_val = COCOImageDataset(coco, val_ids, image_root, is_train=False)

    train_loader = DataLoader(ds_train, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(ds_val, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    
    return train_loader, val_loader