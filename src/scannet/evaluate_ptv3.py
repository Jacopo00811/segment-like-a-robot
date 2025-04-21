import torch
from pointcept.datasets.scannet import ScanNetDataset
from ego_3d.dataset import single_sample_collate_fn
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3
import os
from importlib.util import spec_from_file_location, module_from_spec
from tqdm import tqdm
import numpy as np
from pointcept.datasets.transform import Compose, TRANSFORMS
from pointcept.datasets import collate_fn


def load_config_from_file(config_path):
    
    abs_config_path = os.path.abspath(config_path)
    spec = spec_from_file_location("config", abs_config_path)
    config_module = module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    return config_module


def load_weights(model, weights_path):
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded weights from {weights_path}")
    
    return model



config_path = "./Pointcept/configs/scannet/semseg-pt-v3m1-0-base.py" # use relative path
weights_path = "./models/model_best_PointTransformer_V3.pth"

config_module = load_config_from_file(config_path)

model_config = config_module.model["backbone"]
model = PointTransformerV3(
    in_channels=model_config["in_channels"],
    order=model_config["order"],
    stride=model_config["stride"],
    enc_depths=model_config["enc_depths"],
    enc_channels=model_config["enc_channels"],
    enc_num_head=model_config["enc_num_head"],
    enc_patch_size=model_config["enc_patch_size"],
    dec_depths=model_config["dec_depths"],
    dec_channels=model_config["dec_channels"],
    dec_num_head=model_config["dec_num_head"],
    dec_patch_size=model_config["dec_patch_size"],
    mlp_ratio=model_config["mlp_ratio"],
    qkv_bias=model_config["qkv_bias"],
    qk_scale=model_config["qk_scale"],
    attn_drop=model_config["attn_drop"],
    proj_drop=model_config["proj_drop"],
    drop_path=model_config["drop_path"],
    shuffle_orders=model_config["shuffle_orders"],
    pre_norm=model_config["pre_norm"],
    enable_rpe=model_config["enable_rpe"],
    enable_flash=model_config["enable_flash"],
    upcast_attention=model_config["upcast_attention"],
    upcast_softmax=model_config["upcast_softmax"],
    cls_mode=model_config["cls_mode"],
    pdnorm_bn=model_config["pdnorm_bn"],
    pdnorm_ln=model_config["pdnorm_ln"],
    pdnorm_decouple=model_config["pdnorm_decouple"],
    pdnorm_adaptive=model_config["pdnorm_adaptive"],
    pdnorm_affine=model_config["pdnorm_affine"],
    pdnorm_conditions=model_config["pdnorm_conditions"],
)

model = load_weights(model, weights_path)
model.eval()



DATASET_ROOT = "/dtu/blackhole/0e/169006/ScanNet/preprocessed"



dataset = ScanNetDataset(
    split='val',
    data_root=DATASET_ROOT,
)



dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        collate_fn=collate_fn,
    )




# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Create dictionaries to store results
all_predictions = {}
all_segment_ids = {}
all_confidences = {}
all_scene_metrics = {}


# Run inference test on entire validation set
with torch.no_grad():
    for idx, data_dict in enumerate(tqdm(dataloader, desc="Processing scenes")):

        print(data_dict)

        # Get the first (and only) item in the batch
        data_dict = data_dict[0]  
        
        # Extract scene information
        fragment_list = data_dict.pop("fragment_list")
        segment = data_dict.pop("segment")
        data_name = data_dict.pop("name")
        
        print(f"Processing scene {idx+1}/{len(dataloader)}: {data_name}")
        
        # Initialize prediction tensor for the entire scene
        pred = torch.zeros((segment.size, 20)).to(device)  # 20 classes for ScanNet
        
        # Process each fragment
        for i in range(len(fragment_list)):
            # Process fragment data
            input_dict = fragment_list[i]
            
            # Convert numpy arrays to tensors and move to device
            for key in input_dict.keys():
                if isinstance(input_dict[key], np.ndarray):
                    input_dict[key] = torch.from_numpy(input_dict[key]).to(device)
            
            # Get point indices for this fragment
            idx_part = input_dict["index"]
            
            # Forward pass
            output = model(input_dict)
            pred_logits = output["seg_logits"]  # Shape: [N, 20]
            
            # Apply softmax to get probabilities
            pred_part = torch.nn.functional.softmax(pred_logits, dim=1)
            
            # Add predictions to the overall scene prediction
            bs = 0
            for be in input_dict["offset"]:
                pred[idx_part[bs:be], :] += pred_part[bs:be]
                bs = be
        
        # Get final class predictions
        pred_labels = pred.max(1)[1].cpu().numpy()
        
        # Store results for this scene
        all_predictions[data_name] = pred_labels
        all_segment_ids[data_name] = segment.numpy()
        
        # Calculate metrics for this scene
        intersection, union, target = np.zeros(20), np.zeros(20), np.zeros(20)
        for i in range(20):
            intersection[i] = np.sum((pred_labels == i) & (segment.numpy() == i))
            union[i] = np.sum((pred_labels == i) | (segment.numpy() == i))
            target[i] = np.sum(segment.numpy() == i)
        
        # Calculate IoU scores
        iou_class = np.zeros(20)
        for i in range(20):
            if union[i] == 0:
                iou_class[i] = float('nan')
            else:
                iou_class[i] = intersection[i] / union[i]
        
        # Calculate mean IoU, accuracy
        valid_classes = ~np.isnan(iou_class)
        mean_iou = np.mean(iou_class[valid_classes])
        accuracy = np.sum(intersection) / np.sum(target)
        
        # Store metrics for this scene
        all_scene_metrics[data_name] = {
            'intersection': intersection,
            'union': union,
            'target': target,
            'iou_class': iou_class,
            'mean_iou': mean_iou,
            'accuracy': accuracy
        }
        
        print(f"Scene {data_name}: mIoU = {mean_iou:.4f}, Accuracy = {accuracy:.4f}")

# Calculate overall metrics
total_intersection = np.sum([metrics['intersection'] for metrics in all_scene_metrics.values()], axis=0)
total_union = np.sum([metrics['union'] for metrics in all_scene_metrics.values()], axis=0)
total_target = np.sum([metrics['target'] for metrics in all_scene_metrics.values()], axis=0)

# Calculate overall IoU scores
overall_iou_class = np.zeros(20)
for i in range(20):
    if total_union[i] == 0:
        overall_iou_class[i] = float('nan')
    else:
        overall_iou_class[i] = total_intersection[i] / total_union[i]

# Calculate overall mean IoU, accuracy
valid_classes = ~np.isnan(overall_iou_class)
overall_mean_iou = np.mean(overall_iou_class[valid_classes])
overall_accuracy = np.sum(total_intersection) / np.sum(total_target)

print("\nOverall Results:")
print(f"Mean IoU: {overall_mean_iou:.4f}")
print(f"Accuracy: {overall_accuracy:.4f}")

# Print per-class IoU
class_names = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", 
    "window", "bookshelf", "picture", "counter", "desk", "curtain", 
    "refrigerator", "shower curtain", "toilet", "sink", "bathtub", "otherfurniture"
]

print("\nPer-Class IoU:")
for i in range(20):
    print(f"Class {i} - {class_names[i]}: {overall_iou_class[i]:.4f}")


       
        
       