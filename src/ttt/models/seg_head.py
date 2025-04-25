"""Segmentation head for point cloud segmentation."""

import torch.nn as nn


class SegHead(nn.Module):
    """Segmentation head for point cloud segmentation"""
    
    def __init__(self, backbone_out_channels, num_classes):
        super(SegHead, self).__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x):
        return self.seg_head(x)