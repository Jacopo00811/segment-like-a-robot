import glob
import os
from pointcept.datasets.scannet import ScanNetDataset
from collections.abc import Sequence
import torch
import numpy as np
import torch.utils.data
from pointcept.datasets.utils import point_collate_fn


class EgoSlicedScanNetDataset(ScanNetDataset):
    """
    Dataset class for ego-centric sliced ScanNet data.
    """
    
    def __init__(self, slice_sampling=1, **kwargs):
        self.slice_sampling = max(1, slice_sampling)  
        super().__init__(**kwargs)
    
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
    
    def get_data_name(self, idx):
        """
        Return a name that includes both scene and slice information
        """
        path = self.data_list[idx % len(self.data_list)]
        scene_name = os.path.basename(os.path.dirname(path))
        slice_name = os.path.basename(path)
        return f"{scene_name}_{slice_name}"


def custom_collate_fn(batch):
    """
    Custom collate function that converts NumPy arrays to PyTorch tensors
    while keeping samples separate in lists.
    """
    result = {}
    for key in batch[0].keys():
        result[key] = []
        for sample in batch:
            if isinstance(sample[key], np.ndarray):
                result[key].append(torch.from_numpy(sample[key]))
            else:
                result[key].append(sample[key])
                
    # Offset calculation if needed 
    coords_count = [b['coord'].shape[0] for b in batch]
    offsets = torch.cumsum(torch.tensor(coords_count), dim=0)
    result['offset'] = offsets
    
    return result


if __name__ == "__main__":
    PATH = "/dtu/blackhole/0e/169006/ScanNet/ego_sliced/preprocessed"
    
    # Load all slices
    dataset_all = EgoSlicedScanNetDataset(
        split="val",
        data_root=PATH,
        slice_sampling=1
    )
    
    # Load every 3rd slice
    dataset_sampled = EgoSlicedScanNetDataset(
        split="val",
        data_root=PATH,
        slice_sampling=3
    )
    
    # Load every 5th slice
    dataset_sparse = EgoSlicedScanNetDataset(
        split="val",
        data_root=PATH,
        slice_sampling=5
    )
    
    print("\nSampling comparison:")
    print(f"All slices dataset size: {len(dataset_all)} samples")
    print(f"Every 3rd slice dataset size: {len(dataset_sampled)} samples")
    print(f"Every 5th slice dataset size: {len(dataset_sparse)} samples")
    
    dataset = dataset_all
    print(f"\nAnalyzing dataset with slice_sampling={dataset.slice_sampling}")
    print(f"Dataset contains {len(dataset)} samples")
    
    sample = dataset[0]
    print(f"Sample name: {dataset.get_data_name(0)}")
    print(f"Sample contains keys: {list(sample.keys())}")
    
    scenes = set([os.path.dirname(path) for path in dataset.data_list])
    print(f"Number of unique scenes: {len(scenes)}")
    print(f"Average slices per scene: {len(dataset.data_list)/len(scenes):.1f}")
    
    # Define custom collate function to handle variable-sized point clouds

    
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,  # Reduced batch size for testing
            shuffle=False,
            num_workers=4,
            collate_fn=custom_collate_fn,  # Use our custom collate function
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )
        
    print(f"Dataloader contains {len(dataloader)} batches")
    
    # Get first batch to verify
    batch = next(iter(dataloader))
    print(f"Batch contains keys: {list(batch.keys())}")
    print(f"Number of samples in batch: {len(batch['coord'])}")
    print(f"Sample point counts: {[coord.shape[0] for coord in batch['coord']]}")
    print(f"Offset tensor: {batch['offset']}")
    print(f"{type(batch["coord"][0])}")
