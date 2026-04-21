from .box_ops import jaccard, point_form
from .encoder import encode

def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """
    truths: Ground truth box coordinates [N, 4] (xmin, ymin, xmax, ymax)
    priors: Preset anchor boxes [105000, 4] (cx, cy, w, h)
    labels: Ground truth labels [N]
    loc_t:  Tensor to store encoded coordinate targets [batch_idx, 105000, 4]
    conf_t: Tensor to store label targets [batch_idx, 105000]
    idx:    Current batch index
    """
    # 1. Compute IoU matrix [N, 105000]
    overlaps = jaccard(truths, point_form(priors))
    if truths.size(0) == 0:
        # If no boxes, fill background class (usually 0) at all positions
        conf_t[idx] = 0 
        loc_t[idx] = 0
        return

    # 2. [Key] Ensure each ground truth box matches at least one anchor
    # best_prior_overlap: maximum IoU matched to each ground truth box [N]
    # best_prior_idx: index of maximum IoU anchor for each ground truth box [N]
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    
    # [105000] Maximum IoU for each anchor and corresponding ground truth box index
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    
    # Compress dimensions
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)

    # Force "best matched" anchors to have very high IoU (ensure they are selected as positive samples)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2) 
    
    # Ensure each ground truth box is assigned to its best anchor
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    # 3. Extract matched ground truth box coordinates and labels
    matches = truths[best_truth_idx]          # [105000, 4]
    conf = labels[best_truth_idx]             # [105000] labels+1, because 0 is background
    
    # Set background (0) for IoU below threshold
    conf[best_truth_overlap < threshold] = 0 

    # 4. Encode coordinate offsets
    loc = encode(matches, priors, variances)
    
    # 5. Store to target tensor
    loc_t[idx] = loc
    conf_t[idx] = conf