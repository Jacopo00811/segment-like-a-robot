"""Dataset provider for loading and managing dataset."""

import torch
from pointcept.datasets import collate_fn
from pointcept.datasets.scannet import ScanNetDataset


class DatasetProvider:
    """Class to manage dataset and data loading"""
    
    def __init__(self, data_root):
        """
        Initialize the dataset provider
        
        Args:
            data_root: Root directory containing dataset
        """
        self.data_root = data_root
        
    def build_test_loader(self):
        """
        Build test data loader
        
        Returns:
            test_loader: DataLoader for test set
            test_dataset: Test dataset
        """
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
        """
        Process a data dictionary to get it ready for the model
        
        Args:
            data_dict: Dictionary containing data
            device: Device to move tensors to
            
        Returns:
            point_dict: Processed dictionary
        """
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