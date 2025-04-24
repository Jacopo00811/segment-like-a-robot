import glob
import os
from pointcept.datasets.scannet import ScanNetDataset
from collections.abc import Sequence
import torch
import numpy as np
import torch.utils.data


class EgoSlicedScanNetDataset(ScanNetDataset):
    """
    Dataset class for ego-centric sliced ScanNet data.
    """
    
    def __init__(self, slice_sampling=1, filter_slices=True, min_size_threshold=0.15, **kwargs):
        """
        Initialize the dataset.
        Args:
            slice_sampling: Sample every nth slice
            filter_slices: Whether to filter out empty or small slices
            min_size_threshold: Minimum size threshold as proportion of the average slice size for a scene
            **kwargs: Additional arguments for ScanNetDataset
        """
        self.slice_sampling = max(1, slice_sampling)
        self.filter_slices = filter_slices
        self.min_size_threshold = min_size_threshold
        super().__init__(**kwargs)
        
        # After initialization, filter the data_list if needed
        if self.filter_slices:
            print(f"Filtering dataset: {len(self.data_list)} initial samples... \n")
            self.filter_empty_and_small_slices()
            print(f"Filtered dataset: {len(self.data_list)} valid samples")
    
    def get_data_list(self):
        """
        Override to handle sliced structure. Returns a list of paths to individual slices.
        """
        if isinstance(self.split, str):
            split_list = [self.split]
        elif isinstance(self.split, Sequence):
            split_list = self.split
        else:
            raise NotImplementedError

        data_list = []
        for split in split_list:
            if self.lr is not None:
                # Handle lr file if provided
                scene_paths = [os.path.join(self.data_root, split, scene) for scene in self.lr]
            else:
                # Get all scene directories in the split
                scene_paths = glob.glob(os.path.join(self.data_root, split, "scene*"))
            
            # For each scene, get slices with sampling
            for scene_path in scene_paths:
                slice_paths = sorted(glob.glob(os.path.join(scene_path, "filtered_point_cloud_slice_*")))
                # Apply sampling - take every nth slice
                sampled_slices = slice_paths[::self.slice_sampling]
                data_list.extend(sampled_slices)
                
        return data_list
    
    def calculate_avg_slice_sizes(self):
        """
        Calculate the average point count per slice for each scene.
        Returns:
            dict: Dictionary mapping scene paths to average slice sizes
        """
        scene_to_slices = {}
        
        # Group slices by scene
        for path in self.data_list:
            scene_path = os.path.dirname(path)
            if scene_path not in scene_to_slices:
                scene_to_slices[scene_path] = []
            scene_to_slices[scene_path].append(path)
        
        # Calculate average slice size per scene
        scene_avg_sizes = {}
        for scene_path, slice_paths in scene_to_slices.items():
            total_points = 0
            valid_slices = 0
            print(f"Processing scene: {scene_path}")
            for slice_path in slice_paths:
                try:
                    coord_path = os.path.join(slice_path, 'coord.npy')
                    if os.path.exists(coord_path):
                        coord = np.load(coord_path)
                        if coord.shape[0] > 0:
                            total_points += coord.shape[0]
                            valid_slices += 1
                except Exception as e:
                    pass  # Skip problematic slices
            
            if valid_slices > 0:
                scene_avg_sizes[scene_path] = total_points / valid_slices
            else:
                scene_avg_sizes[scene_path] = 0
        
        return scene_avg_sizes
    
    def filter_empty_and_small_slices(self):
        """
        Filter out empty slices and slices that are too small compared to the scene average.
        """
        original_count = len(self.data_list)
        
        scene_avg_sizes = self.calculate_avg_slice_sizes()
        
        # Filter slices
        valid_slices = []
        for path in self.data_list:
            try:
                coord_path = os.path.join(path, 'coord.npy')
                if not os.path.exists(coord_path):
                    continue
                
                coord = np.load(coord_path)
                
                # If slice is not empty
                if coord.shape[0] == 0:
                    continue
                
                # If slice is not too small compared to scene average
                scene_path = os.path.dirname(path)
                if scene_path in scene_avg_sizes and scene_avg_sizes[scene_path] > 0:
                    avg_size = scene_avg_sizes[scene_path]
                    if coord.shape[0] < avg_size * self.min_size_threshold:
                        continue
                
                valid_slices.append(path)
            except Exception as e:
                print(f"Skipping slice {path} due to error: {e}")
        
        self.data_list = valid_slices
        print(f"Filtered out {original_count - len(self.data_list)} slices. {len(self.data_list)} valid slices remain.")
    
    def get_data_name(self, idx):
        """
        Return a name that includes both scene and slice information
        """
        path = self.data_list[idx % len(self.data_list)]
        scene_name = os.path.basename(os.path.dirname(path))
        slice_name = os.path.basename(path)
        return f"{scene_name}_{slice_name}"


def single_sample_collate_fn(batch):
    """
    Collate function that processes one sample at a time and converts to the format
    expected by the PointTransformerV3 model.
    
    Assumes batch size=1 for inference.
    """
    assert len(batch) == 1, "This collate function only works with batch_size=1"
    sample = batch[0]
    result = {}
    
    # Convert NumPy arrays to PyTorch
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            result[key] = torch.from_numpy(value)
        else:
            result[key] = value
    
    # Add offset tensor if needed by the model
    point_count = result['coord'].shape[0]
    result['offset'] = torch.tensor([point_count])
    
    # Create batch tensor - this is required for the model's sparsify() function
    # For batch_size=1, all points belong to the same batch (batch 0)
    result['batch'] = torch.zeros(point_count, dtype=torch.int32)
    
    # Add required grid_size parameter, indicates the voxel size of each cell 
    result['grid_size'] = 0.02
    
    result['feat'] = torch.cat([result['color'], result['normal']], dim=1)

    return result






################ TESTING ####################
# if __name__ == "__main__":
#     PATH = "/dtu/blackhole/0e/169006/ScanNet/ego_sliced/preprocessed"
    
#     # Load all slices
#     dataset_all = EgoSlicedScanNetDataset(
#         split="val",
#         data_root=PATH,
#         slice_sampling=1
#     )
    
#     # Load every 3rd slice
#     dataset_sampled = EgoSlicedScanNetDataset(
#         split="val",
#         data_root=PATH,
#         slice_sampling=3
#     )
    
#     # Load every 5th slice
#     dataset_sparse = EgoSlicedScanNetDataset(
#         split="val",
#         data_root=PATH,
#         slice_sampling=5
#     )
    
#     print("\nSampling comparison:")
#     print(f"All slices dataset size: {len(dataset_all)} samples")
#     print(f"Every 3rd slice dataset size: {len(dataset_sampled)} samples")
#     print(f"Every 5th slice dataset size: {len(dataset_sparse)} samples")
    
#     dataset = dataset_all
#     print(f"\nAnalyzing dataset with slice_sampling={dataset.slice_sampling}")
#     print(f"Dataset contains {len(dataset)} samples")
    
#     sample = dataset[0]
#     print(f"Sample name: {dataset.get_data_name(0)}")
#     print(f"Sample contains keys: {list(sample.keys())}")
    
#     scenes = set([os.path.dirname(path) for path in dataset.data_list])
#     print(f"Number of unique scenes: {len(scenes)}")
#     print(f"Average slices per scene: {len(dataset.data_list)/len(scenes):.1f}")
    
#     # Define custom collate function to handle variable-sized point clouds

    
#     dataloader = torch.utils.data.DataLoader(
#             dataset,
#             batch_size=1,
#             shuffle=False,
#             num_workers=4,
#             collate_fn=single_sample_collate_fn,  # Use our custom collate function
#             pin_memory=True,
#             drop_last=False,
#             persistent_workers=True,
#         )
        
#     print(f"Dataloader contains {len(dataloader)} batches")
    
#     # Get first batch to verify
#     batch = next(iter(dataloader))
#     print(f"Batch contains keys: {list(batch.keys())}")
#     # print(f"Number of samples in batch: {len(batch['coord'])}")
#     # print(f"Sample point counts: {[coord.shape[0] for coord in batch['coord']]}")
#     print(f"Offset tensor: {batch['offset']}")
#     print(f"{type(batch["coord"][0])}")
