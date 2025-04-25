"""Main orchestration class for test-time adaptation."""

import torch
import tqdm
import sonata

from models.model_provider import ModelProvider
from data.dataset_provider import DatasetProvider
from data.transform_provider import TransformProvider
from utils.scene_memory import SceneMemory
from adaptation.test_time_adapter import TestTimeAdapter
from visualization.visualizer import Visualizer


class SonataTestTimeAdapter:
    """Main class to orchestrate the test-time adaptation process"""
    
    def __init__(self, 
                 model_path,
                 seg_head_path, 
                 data_root,
                 memory_size=5,
                 seed=24525867):
        """
        Initialize the test-time adapter
        
        Args:
            model_path: Path to the model weights
            seg_head_path: Path to the segmentation head weights
            data_root: Root directory containing dataset
            memory_size: Maximum size of scene memory
            seed: Random seed for reproducibility
        """
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
        
    def run(self, num_samples=20, memory_tta_epochs=1, viz_output_path="tta_memory_sem_seg_last_sample.html"):
        """
        Run the test-time adaptation process
        
        Args:
            num_samples: Number of samples to process
            memory_tta_epochs: Number of epochs for memory-based adaptation
            viz_output_path: Path to save visualization
        """
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
                self.scene_memory,
                output_path=viz_output_path
            )