import torch

def encode(matched, priors, variances=[0.1, 0.2]):
    """
    Encode ground truth boxes as offsets relative to prior boxes
    matched: Matched ground truth boxes [105000, 4] (xmin, ymin, xmax, ymax)
    priors: Preset anchor boxes [105000, 4] (cx, cy, w, h)
    """
    # Convert to center point format
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # Encode center point offset
    g_cxcy /= (variances[0] * priors[:, 2:])
    
    # Encode width and height offset
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    
    return torch.cat([g_cxcy, g_wh], 1) # [105000, 4]

def decode(loc, priors, variances=[0.1, 0.2]):
    """
    Decode model output offsets (loc) to real coordinates (xmin, ymin, xmax, ymax)
    
    Args:
        loc: Output from model regression branch [105000, 4] (predicted offsets)
        priors: Preset anchor boxes [105000, 4] (cx, cy, w, h)
        variances: Scaling factors, must be consistent with encode
    """

    # 1. Decode center point (cx, cy)
    # Core formula: predicted_offset * variance * anchor_width + anchor_center
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])
    ), 1)

    # 2. Convert (cx, cy, w, h) back to (xmin, ymin, xmax, ymax)
    # This format is needed for drawing in cv2 or PIL
    decoded_boxes = torch.cat((
        boxes[:, :2] - boxes[:, 2:] / 2,  # xmin, ymin
        boxes[:, :2] + boxes[:, 2:] / 2   # xmax, ymax
    ), 1)

    return decoded_boxes