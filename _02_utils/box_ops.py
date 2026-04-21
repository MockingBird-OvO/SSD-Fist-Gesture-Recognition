import torch

def intersect(box_a, box_b):
    """ Compute intersection area of two boxes
    box_a: [N, 4] (xmin, ymin, xmax, ymax)
    box_b: [M, 4] (xmin, ymin, xmax, ymax)
    """
    n = box_a.size(0)
    m = box_b.size(0)
    # Find top-left and bottom-right corners of intersection area
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(n, m, 2),
                       box_b[:, 2:].unsqueeze(0).expand(n, m, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(n, m, 2),
                       box_b[:, :2].unsqueeze(0).expand(n, m, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    """ Compute Jaccard IoU (Intersection over Union)
    A ∩ B / (A + B - A ∩ B)
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union

def point_form(boxes):
    """ Convert (cx, cy, w, h) to (xmin, ymin, xmax, ymax) """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1) # xmax, ymax

def center_size(boxes):
    """ Convert (xmin, ymin, xmax, ymax) to (cx, cy, w, h) """
    return torch.cat(((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2]), 1)   # w, h


def nms(boxes, scores, overlap=0.5, top_k=200):
    """
    Non-Maximum Suppression
    Args:
        boxes: (tensor) Decoded predicted box coordinates [num_priors, 4] -> (x1, y1, x2, y2)
        scores: (tensor) Class confidence scores [num_priors]
        overlap: (float) IoU threshold, boxes above this are considered duplicates
        top_k: (int) Maximum number of boxes to keep
    Returns:
        keep: (tensor) Indices of kept boxes
    """
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    
    # Sort scores in ascending order
    v, idx = scores.sort(0)  # ascending
    idx = idx[-top_k:]  # Take indices of top_k highest scores
    
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # Select current highest scoring index
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # Remove already saved index
        
        # Calculate IoU between current box and remaining boxes
        xx1 = x1[idx].clamp(min=x1[i])
        yy1 = y1[idx].clamp(min=y1[i])
        xx2 = x2[idx].clamp(max=x2[i])
        yy2 = y2[idx].clamp(max=y2[i])
        
        w = (xx2 - xx1).clamp(min=0.0)
        h = (yy2 - yy1).clamp(min=0.0)
        inter = w * h
        
        rem_areas = area[idx]
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # [idx.size(0)]

        # Keep only boxes with IoU less than threshold (non-overlapping boxes)
        idx = idx[IoU.le(overlap)]
        
    return keep[:count]