import torch
from math import sqrt
from _00_config.config import Config

def create_prior_boxes():
    # Corresponding to 4 scales in your LargeScaleSSD
    feature_maps = Config.FEATURE_MAPS
    # Basic size of anchors at each scale (proportion of original image)
    # Shallow layers see small objects (0.1), deep layers see large objects (0.9)
    scales = Config.ANCHOR_SCALES
    # Aspect ratios for each point (First more square, use 1:1, 1:2, 2:1)
    aspect_ratios = Config.ASPECT_RATIOS

    priors = []
    for i, (_, (f_h, f_w)) in enumerate(feature_maps.items()):
        for y in range(f_h):
            for x in range(f_w):
                # Compute normalized center point coordinates (0~1)
                cx = (x + 0.5) / f_w
                cy = (y + 0.5) / f_h
                
                s = scales[i]
                # Generate one box for each ratio
                for ar in aspect_ratios[i]:
                    priors.append([cx, cy, s * sqrt(ar), s / sqrt(ar)])
                    # SSD convention: for 1:1 ratio, add another slightly larger box
                    if ar == 1.0:
                        s_prime = sqrt(s * scales[i+1]) if i < 3 else s * 1.1
                        priors.append([cx, cy, s_prime, s_prime])

    return torch.tensor(priors) # [105000, 4]
