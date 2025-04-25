import numpy as np
import torch
import torch.nn as nn
import sonata
import plotly.graph_objects as go
from torch.optim import AdamW
from pointcept.datasets.transform import Compose
import torch.nn.functional as F
from pointcept.datasets import build_dataset, collate_fn
from pointcept.datasets.scannet import ScanNetDataset
import tqdm
import os
from collections import deque

try:
    import flash_attn
except ImportError:
    flash_attn = None


class ScanNetMetadata:
    """Class to hold ScanNet dataset metadata"""
    
    VALID_CLASS_IDS_20 = (
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39,
    )

    CLASS_LABELS_20 = (
        "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window",
        "bookshelf", "picture", "counter", "desk", "curtain", "refrigerator", 
        "shower curtain", "toilet", "sink", "bathtub", "otherfurniture",
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

    @classmethod
    def get_class_colors(cls):
        """Get class colors for visualization"""
        return [cls.SCANNET_COLOR_MAP_20[id] for id in cls.VALID_CLASS_IDS_20]


class SegHead(nn.Module):
    """Segmentation head for point cloud segmentation"""
    
    def __init__(self, backbone_out_channels, num_classes):
        super(SegHead, self).__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x):
        return self.seg_head(x)


class SceneMemory:
    """Class to manage scene memory for test-time adaptation"""
    
    def __init__(self, max_size=5):
        """
        Initialize a sliding window memory for scene indices
        
        Args:
            max_size: Maximum number of scene indices to keep in memory
        """
        self.max_size = max_size
        self.indices = deque(maxlen=max_size)
        
    def add_scene_index(self, index):
        """
        Add scene index to memory
        
        Args:
            index: The dataset index of the scene
        """
        if index not in self.indices:
            self.indices.append(index)
        
    def get_scene_indices(self):
        """
        Return all scene indices in memory
        
        Returns:
            List of scene indices
        """
        return list(self.indices)
    
    def is_empty(self):
        """Check if memory is empty"""
        return len(self.indices) == 0
    
    def size(self):
        """Return current memory size"""
        return len(self.indices)


class DatasetProvider:
    """Class to manage dataset and data loading"""
    
    def __init__(self, data_root="/dtu/blackhole/0e/169006/ScanNet/preprocessed"):
        self.data_root = data_root
        
    def build_test_loader(self):
        """Build test data loader"""
        test_dataset = ScanNetDataset(
            split="val",
            data_root=self.data_root,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,  # Keep batch size 1 for point clouds of varying sizes
            shuffle=False,
            num_workers=4,  # Increased from 1 to 4 for better performance
            pin_memory=True,
            collate_fn=collate_fn,
        )
        return test_loader, test_dataset
    
    @staticmethod
    def process_point_for_model(data_dict, device):
        """Process a data dictionary to get it ready for the model"""
        point_dict = {}
        for key, val in data_dict.items():
            if key == 'segment_20':
                point_dict['segment'] = val
            elif key != 'segment_200':  # Skip segment_200
                point_dict[key] = val
        
        # Convert to device
        for key in point_dict.keys():
            if isinstance(point_dict[key], torch.Tensor):
                point_dict[key] = point_dict[key].to(device, non_blocking=True)
        
        return point_dict


class ModelProvider:
    """Class to load and manage models"""
    
    def __init__(self, device):
        self.device = device
        
    def load_model(self, model_path="./models/sonata/sonata.pth"):
        """Load the backbone model"""
        if flash_attn is None:
            print("Warning: flash_attn is not installed. Performance might be affected.")
        
        model = sonata.load(model_path)
        model = model.to(self.device)
        
        return model
    
    def load_seg_head(self, ckpt_path="./models/sonata/sonata_linear_prob_head_sc.pth"):
        """Load the segmentation head"""
        ckpt = sonata.load(ckpt_path, ckpt_only=True)
        seg_head = SegHead(**ckpt["config"]).to(self.device)
        seg_head.load_state_dict(ckpt["state_dict"])
        
        return seg_head
    
    def configure_model_for_adaptation(self, model):
        """Configure model for test-time adaptation by freezing/unfreezing parameters"""
        # Freeze all model parameters
        for name, param in model.named_parameters():
            param.requires_grad = False
        
        # Unfreeze parameters in later encoder blocks for adaptation
        for name, param in model.named_parameters():
            if "enc.enc4" in name:
                param.requires_grad = True
        
        self.print_trainable_parameters(model)
        
        return model
    
    def print_trainable_parameters(self, model):
        """Print information about trainable parameters"""
        print("Trainable Parameters:")
        trainable_count = 0
        total_count = 0
        for name, param in model.named_parameters():
            total_count += param.numel()
            if param.requires_grad:
                trainable_count += param.numel()
                print(f"Layer: {name} | Size: {param.size()} | Requires Grad: {param.requires_grad}")
        
        print(f"Total parameters: {total_count}, Trainable parameters: {trainable_count} ({trainable_count/total_count*100:.2f}%)")


class TransformProvider:
    """Class to manage data transforms"""
    
    @staticmethod
    def get_default_transform():
        """Get default transform"""
        return sonata.transform.default()
    
    @staticmethod
    def get_test_time_transform():
        """Get transform for test-time augmentation"""
        ttt_config = [
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color", "inverse"),
                feat_keys=("coord", "color", "normal"),
            ),
        ]
        
        return Compose(ttt_config)


class TestTimeAdapter:
    """Class to manage test-time adaptation"""
    
    def __init__(self, model, device, transform, tt_transform, learning_rate=1e-5):
        self.model = model
        self.device = device
        self.transform = transform
        self.tt_transform = tt_transform
        self.lr = learning_rate
        
    def adapt_model(self, scene_memory, test_dataset, epochs=1):
        """Adapt the model using scenes in memory"""
        self.model.train()
        trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        optimizer = AdamW(trainable_params, lr=self.lr)
        
        memory_indices = scene_memory.get_scene_indices()
        print(f"Memory contains indices: {memory_indices}")
        
        # Perform test-time adaptation using all scenes in memory
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Process each scene in memory
            for memory_idx in memory_indices:
                # Get scene data from memory
                memory_data_dict = test_dataset.get_data(memory_idx)
                
                # Get original point data
                og_point = self.transform(memory_data_dict.copy())
                for key in og_point.keys():
                    if isinstance(og_point[key], torch.Tensor):
                        og_point[key] = og_point[key].to(self.device, non_blocking=True)
                
                # Apply test-time transform for augmentation
                perturbed_point = self.tt_transform(memory_data_dict.copy())
                for key in perturbed_point.keys():
                    if isinstance(perturbed_point[key], torch.Tensor):
                        perturbed_point[key] = perturbed_point[key].to(self.device, non_blocking=True)
                
                # Forward pass and compute loss
                optimizer.zero_grad()
                
                # Forward pass with original point
                og_features = self.model(og_point)
                
                # Forward pass with perturbed point
                perturbed_features = self.model(perturbed_point)
                
                # Consistency loss: make embeddings from original and perturbed versions similar
                avg_og_features = og_features['feat'].mean(dim=0, keepdim=True)
                avg_perturbed_features = perturbed_features['feat'].mean(dim=0, keepdim=True)
                
                loss_tensor = 1 - F.cosine_similarity(avg_og_features, avg_perturbed_features)
                loss = loss_tensor.mean()
                
                # Print loss for this memory scene
                print(f"  Scene {memory_idx} - Loss: {loss.item():.4f}")
                
                # Backprop and optimize
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Report average loss across all memory scenes
            avg_loss = epoch_loss / len(memory_indices)
            print(f"Memory training - Epoch {epoch}, Average Loss: {avg_loss:.4f}")
            
            # Zero gradients between epochs
            optimizer.zero_grad()


class Visualizer:
    """Class to handle visualizations"""
    
    def __init__(self, model, seg_head, device, transform):
        self.model = model
        self.seg_head = seg_head
        self.device = device
        self.transform = transform
        
    def visualize_sample(self, data_dict, sample_idx, scene_memory, output_path="tta_memory_sem_seg_last_sample.html"):
        """Visualize segmentation results for a sample"""
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
        """Create Plotly visualization for point cloud segmentation"""
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


class SonataTestTimeAdapter:
    """Main class to orchestrate the test-time adaptation process"""
    
    def __init__(self, 
                 model_path="./models/sonata/sonata.pth",
                 seg_head_path="./models/sonata/sonata_linear_prob_head_sc.pth", 
                 data_root="/dtu/blackhole/0e/169006/ScanNet/preprocessed",
                 memory_size=5,
                 seed=24525867):
        # Set random seed
        sonata.utils.set_seed(seed)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.data_provider = DatasetProvider(data_root)
        self.model_provider = ModelProvider(self.device)
        self.transform_provider = TransformProvider()
        
        # Load model and segmentation head
        self.model = self.model_provider.load_model(model_path)
        self.model = self.model_provider.configure_model_for_adaptation(self.model)
        self.seg_head = self.model_provider.load_seg_head(seg_head_path)
        
        # Get transforms
        self.transform = self.transform_provider.get_default_transform()
        self.tt_transform = self.transform_provider.get_test_time_transform()
        
        # Initialize adaptor
        self.adapter = TestTimeAdapter(
            self.model, 
            self.device, 
            self.transform, 
            self.tt_transform
        )
        
        # Initialize visualizer
        self.visualizer = Visualizer(
            self.model,
            self.seg_head,
            self.device,
            self.transform
        )
        
        # Initialize scene memory
        self.scene_memory = SceneMemory(max_size=memory_size)
        
    def run(self, num_samples=20, memory_tta_epochs=1):
        """Run the test-time adaptation process"""
        # Load dataset and dataloader
        test_loader, test_dataset = self.data_provider.build_test_loader()
        print(f"Test loader with {len(test_loader)} samples")
        
        # Limit number of samples for testing
        num_samples = min(num_samples, len(test_loader))
        
        # Test-time adaptation loop
        last_sample_data = None
        
        for sample_idx in tqdm.tqdm(range(num_samples)):
            # Process current sample
            data_dict = test_dataset.get_data(sample_idx)
            
            # Add current sample index to memory
            self.scene_memory.add_scene_index(sample_idx)
            
            # Store the current sample data for later visualization (only the last one)
            last_sample_data = {
                'sample_idx': sample_idx,
                'data_dict': data_dict
            }
            
            # Perform test-time adaptation using all scenes in memory
            self.adapter.adapt_model(self.scene_memory, test_dataset, epochs=memory_tta_epochs)
        
        # After processing all samples, generate plot only for the last one
        if last_sample_data is not None:
            print(f"Creating visualization for the last sample (index: {last_sample_data['sample_idx']})")
            
            self.visualizer.visualize_sample(
                last_sample_data['data_dict'],
                last_sample_data['sample_idx'],
                self.scene_memory
            )


def main():
    """Main entry point"""
    # Initialize and run the adapter
    adapter = SonataTestTimeAdapter()
    adapter.run(num_samples=20, memory_tta_epochs=2)


if __name__ == "__main__":
    main()