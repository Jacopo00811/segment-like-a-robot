"""Provider for loading and configuring models."""

import sonata
from models.seg_head import SegHead

try:
    import flash_attn
except ImportError:
    flash_attn = None


class ModelProvider:
    """Class to load and manage models"""
    
    def __init__(self, device):
        """
        Initialize model provider
        
        Args:
            device: Device to load models to
        """
        self.device = device
        
    def load_model(self, model_path):
        """
        Load the backbone model
        
        Args:
            model_path: Path to the model weights
            
        Returns:
            Loaded model
        """
        if flash_attn is None:
            print("Warning: flash_attn is not installed. Performance might be affected.")
        
        model = sonata.load(model_path)
        model = model.to(self.device)
        
        return model
    
    def load_seg_head(self, ckpt_path):
        """
        Load the segmentation head
        
        Args:
            ckpt_path: Path to the segmentation head weights
            
        Returns:
            Loaded segmentation head
        """
        ckpt = sonata.load(ckpt_path, ckpt_only=True)
        seg_head = SegHead(**ckpt["config"]).to(self.device)
        seg_head.load_state_dict(ckpt["state_dict"])
        
        return seg_head
    
    def configure_model_for_adaptation(self, model):
        """
        Configure model for test-time adaptation by freezing/unfreezing parameters
        
        Args:
            model: Model to configure
            
        Returns:
            Configured model
        """
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
        """
        Print information about trainable parameters
        
        Args:
            model: Model to analyze
        """
        print("Trainable Parameters:")
        trainable_count = 0
        total_count = 0
        for name, param in model.named_parameters():
            total_count += param.numel()
            if param.requires_grad:
                trainable_count += param.numel()
                print(f"Layer: {name} | Size: {param.size()} | Requires Grad: {param.requires_grad}")
        
        print(f"Total parameters: {total_count}, Trainable parameters: {trainable_count} ({trainable_count/total_count*100:.2f}%)")