import torch
import torch.nn as nn
import math
from _00_config.config import Config

class LargeScaleSSD(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        
        # --- Stage 1: Extract high-resolution features (120x160) ---
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), # -> 240x320
            nn.Conv2d(32, Config.STAGE_CHANNELS[0], 3, padding=1), nn.BatchNorm2d(Config.STAGE_CHANNELS[0]), nn.ReLU(),
            nn.MaxPool2d(2), # -> 120x160
        )
        
        # --- Stage 2: Extract medium-resolution features (60x80) ---
        self.stage2 = nn.Sequential(
            nn.Conv2d(Config.STAGE_CHANNELS[0], Config.STAGE_CHANNELS[1], 3, stride=2, padding=1), nn.BatchNorm2d(Config.STAGE_CHANNELS[1]), nn.ReLU(), 
        )
        
        # --- Stage 3: Extract low-resolution features (30x40) ---
        self.stage3 = nn.Sequential(
            nn.Conv2d(Config.STAGE_CHANNELS[1], Config.STAGE_CHANNELS[2], 3, stride=2, padding=1), nn.BatchNorm2d(Config.STAGE_CHANNELS[2]), nn.ReLU(), 
        )
        
        # --- Stage 4: Global features (15x20) ---
        self.stage4 = nn.Sequential(
            nn.Conv2d(Config.STAGE_CHANNELS[2], Config.STAGE_CHANNELS[3], 3, stride=2, padding=1), nn.BatchNorm2d(Config.STAGE_CHANNELS[3]), nn.ReLU(), 
        )

        self.num_anchors = Config.NUM_ANCHORS 

        # --- 4-scale regression heads ---
        def make_head(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=1),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=Config.DROPOUT_P),   # New Dropout
                nn.Conv2d(in_ch, in_ch, 3, padding=1),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, 3, padding=1), # Reduced to 2 feature layers + 1 output
            )

        self.cls_heads = nn.ModuleList([
            make_head(Config.STAGE_CHANNELS[0],  self.num_anchors[0] * num_classes),
            make_head(Config.STAGE_CHANNELS[1], self.num_anchors[1] * num_classes),
            make_head(Config.STAGE_CHANNELS[2], self.num_anchors[2] * num_classes),
            make_head(Config.STAGE_CHANNELS[3], self.num_anchors[3] * num_classes),
        ])

        self.reg_heads = nn.ModuleList([
            make_head(Config.STAGE_CHANNELS[0],  self.num_anchors[0] * 4),
            make_head(Config.STAGE_CHANNELS[1], self.num_anchors[1] * 4),
            make_head(Config.STAGE_CHANNELS[2], self.num_anchors[2] * 4),
            make_head(Config.STAGE_CHANNELS[3], self.num_anchors[3] * 4),
        ])
        # ====================== Key modification: Initialization logic ======================
        # 1. Basic weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 2. Inject "background bias": Force model to initially predict background, break classification deadlock
        prior_p = Config.PRIOR_P  # Initially assume only this probability of object
        bias_val = math.log((1 - prior_p) / prior_p)  # About 4.6

        for head in self.cls_heads:
            with torch.no_grad():
                b = head[-1].bias.view(-1, self.num_classes)  # [-1] takes last layer of Sequential
                b[:, 0] = bias_val
                b[:, 1:] = -bias_val
                head[-1].bias.copy_(b.view(-1))

        # ===============================================================
    
    def forward(self, x):
        batch_size = x.size(0)
        
        f1 = self.stage1(x) 
        f2 = self.stage2(f1) 
        f3 = self.stage3(f2) 
        f4 = self.stage4(f3) 
        
        features = [f1, f2, f3, f4]
        regs, clss = [], []

        for i, feat in enumerate(features):
            r = self.reg_heads[i](feat).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
            c = self.cls_heads[i](feat).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_classes)
            regs.append(r)
            clss.append(c)

        return torch.cat(regs, dim=1), torch.cat(clss, dim=1)