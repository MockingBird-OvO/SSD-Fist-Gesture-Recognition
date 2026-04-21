import torch
import torch.nn as nn
import torch.nn.functional as F
from _02_utils import match
from _00_config.config import Config

class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh=None, neg_ratio=None, alpha=None):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh if overlap_thresh is not None else Config.OVERLAP_THRESHOLD
        self.neg_ratio = neg_ratio if neg_ratio is not None else Config.NEG_RATIO
        self.variances = Config.VARIANCES
        self.alpha = alpha if alpha is not None else Config.ALPHA  # Dynamic weight coefficient

    def forward(self, predictions, targets, priors):
        loc_data, conf_data = predictions
        batch_size = loc_data.size(0)
        num_priors = priors.size(0)

        loc_t = torch.zeros(batch_size, num_priors, 4).to(loc_data.device)
        conf_t = torch.zeros(batch_size, num_priors, dtype=torch.long).to(loc_data.device)

        for idx in range(batch_size):
            truths = targets[idx][:, :-1]
            labels = targets[idx][:, -1]
            match(self.threshold, truths, priors, self.variances, labels, loc_t, conf_t, idx)

        pos = conf_t > 0  # [Batch, 105000]
        num_pos = pos.sum(dim=1, keepdim=True) 

        # 2. Localization loss
        pos_idx = pos.unsqueeze(2).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # 3. Hard negative mining
        conf_p = conf_data.view(-1, self.num_classes)
        loss_c_raw = F.cross_entropy(conf_p, conf_t.view(-1), reduction='none').view(batch_size, -1)
        
        loss_c_raw[pos] = 0  
        _, loss_idx = loss_c_raw.sort(1, descending=True) 
        _, idx_rank = loss_idx.sort(1) 

        num_neg = torch.clamp(self.neg_ratio * num_pos.float(), min=10, max=pos.size(1) - 1).long()
        neg = (idx_rank < num_neg) & (~pos)

        # 4. Classification loss
        pos_neg_mask = pos | neg
        conf_p_final = conf_data[pos_neg_mask.unsqueeze(2).expand_as(conf_data)].view(-1, self.num_classes)
        conf_t_final = conf_t[pos_neg_mask]
        loss_c = F.cross_entropy(conf_p_final, conf_t_final, reduction='sum')

        # 5. Normalize total loss

        N = num_pos.sum().item()
        if N == 0: 
            print("Warning: No positive samples in this batch!")
        N = max(1, N)

        loss_l = loss_l / N
        loss_c = loss_c / N

        # 6. Compute weighted total loss
        total_loss = loss_c + (self.alpha * loss_l)

        return loss_l, loss_c, total_loss # <--- Return three values