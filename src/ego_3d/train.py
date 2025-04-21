import torch
import os
from importlib.util import spec_from_file_location, module_from_spec
from dataset import EgoSlicedScanNetDataset, single_sample_collate_fn
from tqdm import tqdm
from pointcept.models.default import DefaultSegmentorV2
import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_segmentation_metrics(pred, target, num_classes=20, ignore_index=-1):
    """
    Calculate semantic segmentation metrics
    
    Args:
        pred: torch.Tensor, prediction tensor [N]
        target: torch.Tensor, ground truth tensor [N]
        num_classes: int, number of classes
        ignore_index: int, index to ignore in metrics
        
    Returns:
        metrics: dict, containing IoU per class, mIoU, accuracy per class, mean accuracy, overall accuracy
    """
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    valid_mask = target != ignore_index
    pred_valid = pred[valid_mask]
    target_valid = target[valid_mask]
    
    conf_matrix = confusion_matrix(
        target_valid, 
        pred_valid, 
        labels=list(range(num_classes))
    )
    
    # Calculate IoU for each class
    # IoU = TP / (TP + FP + FN)
    intersection = np.diag(conf_matrix)
    union = np.sum(conf_matrix, axis=0) + np.sum(conf_matrix, axis=1) - intersection
    iou_per_class = intersection / (union + 1e-10)
    
    # Calculate accuracy for each class
    # Accuracy = TP / (TP + FN)
    total_per_class = np.sum(conf_matrix, axis=1)
    acc_per_class = intersection / (total_per_class + 1e-10)
    
    overall_acc = np.sum(intersection) / (np.sum(conf_matrix) + 1e-10)
    
    metrics = {
        'iou_per_class': iou_per_class,
        'miou': np.mean(iou_per_class),
        'acc_per_class': acc_per_class,
        'mean_acc': np.mean(acc_per_class),
        'overall_acc': overall_acc,
        'conf_matrix': conf_matrix
    }
    
    return metrics


def load_config_from_file(config_path):
    abs_config_path = os.path.abspath(config_path)
    spec = spec_from_file_location("config", abs_config_path)
    config_module = module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module


config_path = "/zhome/f9/0/168881/Desktop/segment-like-a-robot/Pointcept/configs/scannet/semseg-pt-v3m1-0-base.py"
weights_path = "models/model_best_PointTransformer_V3.pth"
data_path = "/dtu/blackhole/0e/169006/ScanNet/ego_sliced/preprocessed"

config_module = load_config_from_file(config_path)
model = DefaultSegmentorV2(
    num_classes=config_module.model["num_classes"],
    backbone_out_channels=config_module.model["backbone_out_channels"],
    backbone=config_module.model["backbone"],  # Use the configuration dictionary
    criteria=config_module.model["criteria"]
)

checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

# Prefix the keys with 'backbone.' if they don't already have it
backbone_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('backbone.'):
        backbone_state_dict[k] = v
    else:
        backbone_state_dict[f'backbone.{k}'] = v


# Load the weights into the model
model.load_state_dict(backbone_state_dict, strict=False)
print(f"Loaded weights from {weights_path}")

dataset = EgoSlicedScanNetDataset(
    split="val",
    data_root=data_path,
    slice_sampling=100,
    filter_slices=True,  # Enable filtering
    min_size_threshold=0.30  # Filter out slices smaller than 30% of scene average
)

dataloader_val = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    collate_fn=single_sample_collate_fn,
    pin_memory=True,
    drop_last=False,
    persistent_workers=True,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

all_predictions = []
all_targets = []
with torch.no_grad():
    for batch_idx, input_dict in enumerate(tqdm(dataloader_val, desc="Processing slices")):
        for key in input_dict:
            if isinstance(input_dict[key], torch.Tensor):
                if key == 'segment':
                    input_dict[key] = input_dict[key].long().to(device)
                else:
                    input_dict[key] = input_dict[key].to(device)
        
        sample_name = dataset.get_data_name(batch_idx)

        outputs = model(input_dict)
        
        seg_logits = outputs['seg_logits']
        seg_pred = seg_logits.argmax(dim=1)  # Get class predictions [N]
        
        all_predictions.append(seg_pred.cpu())
        all_targets.append(input_dict['segment'].cpu())
            

if all_predictions:
    # Concatenate all predictions and targets
    all_pred_tensor = torch.cat(all_predictions, dim=0)
    all_target_tensor = torch.cat(all_targets, dim=0)
    
    # Calculate metrics on the complete dataset
    final_metrics = calculate_segmentation_metrics(
        all_pred_tensor, 
        all_target_tensor,
        num_classes=config_module.model["num_classes"], 
        ignore_index=-1
    )
    
    print("\n===== FINAL EVALUATION RESULTS =====")
    print(f"mIoU: {final_metrics['miou']:.4f}")
    print(f"Overall Accuracy: {final_metrics['overall_acc']:.4f}")
    
    # Print per-class IoU for detailed analysis
    print("\nPer-class IoU:")
    for i, iou in enumerate(final_metrics['iou_per_class']):
        print(f"Class {i}: {iou:.4f}")
    
    # Print per-class accuracy
    print("\nPer-class Accuracy:")
    for i, acc in enumerate(final_metrics['acc_per_class']):
        print(f"Class {i}: {acc:.4f}")