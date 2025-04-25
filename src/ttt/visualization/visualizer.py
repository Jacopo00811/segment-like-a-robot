"""Visualization utilities for point cloud segmentation."""

import numpy as np
import torch
import plotly.graph_objects as go
from models.metadata import ScanNetMetadata


class Visualizer:
    """Class to handle visualizations"""
    
    def __init__(self, model, seg_head, device, transform):
        """
        Initialize visualizer
        
        Args:
            model: Model to visualize outputs from
            seg_head: Segmentation head
            device: Device to use
            transform: Transform to apply to data
        """
        self.model = model
        self.seg_head = seg_head
        self.device = device
        self.transform = transform
        
    def visualize_sample(self, data_dict, sample_idx, scene_memory, output_path="tta_memory_sem_seg_last_sample.html"):
        """
        Visualize segmentation results for a sample
        
        Args:
            data_dict: Data dictionary
            sample_idx: Sample index
            scene_memory: Scene memory containing indices
            output_path: Path to save visualization
        """
        # Switch to evaluation mode
        self.model.eval()
        self.seg_head.eval()
        
        # Get original point data
        og_point = self.transform(data_dict)

        for key in og_point.keys():
            if isinstance(og_point[key], torch.Tensor):
                og_point[key] = og_point[key].to(self.device, non_blocking=True)
        
        with torch.no_grad():
            point = self.model(og_point)
            
            # Recover hierarchical features
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            
            # Get segmentation prediction
            feat = point.feat
            seg_logits = self.seg_head(feat)
            pred = seg_logits.argmax(dim=-1).cpu().numpy()
            class_colors = np.array(ScanNetMetadata.get_class_colors())[pred]
            
            # Create and save visualization
            self._create_plotly_visualization(
                point.coord.cpu().numpy(),
                class_colors / 255.0,
                sample_idx,
                scene_memory.get_scene_indices(),
                output_path
            )
    
    def _create_plotly_visualization(self, coords, colors, sample_idx, memory_indices, output_path):
        """
        Create Plotly visualization for point cloud segmentation
        
        Args:
            coords: Point coordinates
            colors: Point colors
            sample_idx: Sample index
            memory_indices: Memory indices
            output_path: Path to save visualization
        """
        # Create markers for the legend
        legend_traces = []
        for i, (class_id, label) in enumerate(zip(ScanNetMetadata.VALID_CLASS_IDS_20, ScanNetMetadata.CLASS_LABELS_20)):
            rgb_color = np.array(ScanNetMetadata.SCANNET_COLOR_MAP_20[class_id]) / 255.0
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
        
        # Create scatter plot
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
        
        # Combine traces
        traces = [point_cloud] + legend_traces
        
        # Create figure
        fig = go.Figure(data=traces)
        
        # Update layout
        fig.update_layout(
            title=f"Sonata Semantic Segmentation with Memory-Based TTA - Sample (Index: {sample_idx})",
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
        
        # Add memory information to the figure as annotation
        fig.add_annotation(
            text=f"Memory contains scenes: {memory_indices}",
            xref="paper", yref="paper",
            x=0.5, y=0.02,
            showarrow=False,
            font=dict(size=12)
        )
        
        # Save visualization
        fig.write_html(output_path)
        print(f"Point cloud visualization saved as '{output_path}'")