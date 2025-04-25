# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import torch
import torch.nn as nn
import sonata
import plotly.graph_objects as go

try:
    import flash_attn
except ImportError:
    flash_attn = None


# ScanNet Meta data
VALID_CLASS_IDS_20 = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    14,
    16,
    24,
    28,
    33,
    34,
    36,
    39,
)


CLASS_LABELS_20 = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
)

SCANNET_COLOR_MAP_20 = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),
    14: (247.0, 182.0, 210.0),
    15: (66.0, 188.0, 102.0),
    16: (219.0, 219.0, 141.0),
    17: (140.0, 57.0, 197.0),
    18: (202.0, 185.0, 52.0),
    19: (51.0, 176.0, 203.0),
    20: (200.0, 54.0, 131.0),
    21: (92.0, 193.0, 61.0),
    22: (78.0, 71.0, 183.0),
    23: (172.0, 114.0, 82.0),
    24: (255.0, 127.0, 14.0),
    25: (91.0, 163.0, 138.0),
    26: (153.0, 98.0, 156.0),
    27: (140.0, 153.0, 101.0),
    28: (158.0, 218.0, 229.0),
    29: (100.0, 125.0, 154.0),
    30: (178.0, 127.0, 135.0),
    32: (146.0, 111.0, 194.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    35: (96.0, 207.0, 209.0),
    36: (227.0, 119.0, 194.0),
    37: (213.0, 92.0, 176.0),
    38: (94.0, 106.0, 211.0),
    39: (82.0, 84.0, 163.0),
    40: (100.0, 85.0, 144.0),
}

CLASS_COLOR_20 = [SCANNET_COLOR_MAP_20[id] for id in VALID_CLASS_IDS_20]


class SegHead(nn.Module):
    def __init__(self, backbone_out_channels, num_classes):
        super(SegHead, self).__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x):
        return self.seg_head(x)


if __name__ == "__main__":
    # set random seed
    sonata.utils.set_seed(24525867)


    
    # Load model
    if flash_attn is not None:
        model = sonata.load("./models/sonata/sonata.pth")


    # Load linear probing seg head
    ckpt = sonata.load(
        "./models/sonata/sonata_linear_prob_head_sc.pth", ckpt_only=True
    )
    seg_head = SegHead(**ckpt["config"]).cuda()
    seg_head.load_state_dict(ckpt["state_dict"])
    # Load default data transform pipline
    transform = sonata.transform.default()
    # Load data
    point = sonata.data.load("sample1")
    point.pop("segment200")
    segment = point.pop("segment20")
    point["segment"] = segment  # two kinds of segment exist in ScanNet, only use one
    original_coord = point["coord"].copy()
    point = transform(point)
    
    device = torch.device("cuda")
    model = model.to(device)
    seg_head = seg_head.to(device)
    
    # Inference
    model.eval()
    seg_head.eval()
    with torch.inference_mode():
        for key in point.keys():
            if isinstance(point[key], torch.Tensor):
                point[key] = point[key].cuda(non_blocking=True)
        # model forward:
        point = model(point)
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        feat = point.feat
        seg_logits = seg_head(feat)
        pred = seg_logits.argmax(dim=-1).data.cpu().numpy()
        color = np.array(CLASS_COLOR_20)[pred]

    # Extract point cloud coordinates and colors
    coords = point.coord.cpu().detach().numpy()
    colors = color / 255.0  # Normalize colors to 0-1 range

    # Create the colorscale for the legend
    colorscale = []
    for i, (class_id, label) in enumerate(zip(VALID_CLASS_IDS_20, CLASS_LABELS_20)):
        rgb_color = np.array(SCANNET_COLOR_MAP_20[class_id]) / 255.0
        colorscale.append([i/len(CLASS_LABELS_20), f'rgb({rgb_color[0]*255},{rgb_color[1]*255},{rgb_color[2]*255})'])
    
    # Create markers for each class for the legend
    legend_traces = []
    for i, (class_id, label) in enumerate(zip(VALID_CLASS_IDS_20, CLASS_LABELS_20)):
        rgb_color = np.array(SCANNET_COLOR_MAP_20[class_id]) / 255.0
        legend_traces.append(
            go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(
                    size=10,
                    color=f'rgb({rgb_color[0]*255},{rgb_color[1]*255},{rgb_color[2]*255})'
                ),
                name=label,
                showlegend=True
            )
        )
    
    # Create the 3D scatter plot for the point cloud
    point_cloud = go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=[f'rgb({c[0]*255},{c[1]*255},{c[2]*255})' for c in colors],
            opacity=0.8
        ),
        showlegend=False
    )
    
    # Combine all traces
    traces = [point_cloud] + legend_traces
    
    # Create the figure
    fig = go.Figure(data=traces)
    
    # Update layout for better visualization
    fig.update_layout(
        title="Sonata Semantic Segmentation",
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Save as HTML file
    fig.write_html("sem_seg.html")
    
    # Optional: Display figure interactively if running in a notebook
    # fig.show()
    
    print("Point cloud visualization saved as 'sem_seg.html'")