"""Test-time adaptation for point cloud segmentation."""

import torch
import torch.nn.functional as F
from torch.optim import AdamW


class TestTimeAdapter:
    """Class to manage test-time adaptation"""
    
    def __init__(self, model, device, transform, tt_transform, learning_rate=1e-5):
        """
        Initialize the test-time adapter
        
        Args:
            model: Model to adapt
            device: Device to use
            transform: Default transform
            tt_transform: Test-time transform for augmentation
            learning_rate: Learning rate for optimization
        """
        self.model = model
        self.device = device
        self.transform = transform
        self.tt_transform = tt_transform
        self.lr = learning_rate
        
    def adapt_model(self, scene_memory, test_dataset, epochs=1):
        """
        Adapt the model using scenes in memory
        
        Args:
            scene_memory: Scene memory containing indices
            test_dataset: Test dataset
            epochs: Number of adaptation epochs
        """
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