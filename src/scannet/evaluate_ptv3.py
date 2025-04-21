import torch
from pointcept.datasets.scannet import ScanNetDataset
from pointcept.datasets.utils import collate_fn
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3
import os
from importlib.util import spec_from_file_location, module_from_spec
from tqdm import tqdm
import numpy as np

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
    data_root=DATASET_ROOT
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
    for batch_idx, input_dict in enumerate(tqdm(dataloader, desc="Processing scenes")):
        # Move input data to device
        for key in input_dict:
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].to(device)
        
        # Get sample name
        sample_name = dataset.get_data_name(batch_idx)
        
        # Forward pass
        outputs = model(input_dict)
        
        # Process predictions - no need to index with [i] since there's only one sample
        if isinstance(outputs, dict):
            if "seg_logits" in outputs:
                logits = outputs["seg_logits"]
                pred_labels = torch.argmax(logits, dim=1).cpu().numpy()
                confidences = torch.softmax(logits, dim=1).max(dim=1)[0].cpu().numpy()
            else:
                continue
        else:
            logits = outputs
            pred_labels = torch.argmax(logits, dim=1).cpu().numpy()
            confidences = torch.softmax(logits, dim=1).max(dim=1)[0].cpu().numpy()
        
        # Store results
        all_predictions[sample_name] = pred_labels
        all_confidences[sample_name] = confidences
        
        # If ground truth is available
        if "segment" in input_dict:
            segment = input_dict["segment"].cpu().numpy()
            all_segment_ids[sample_name] = segment
            
            # Calculate metrics
            valid_mask = segment != -1
            correct = (pred_labels[valid_mask] == segment[valid_mask])
            accuracy = correct.sum() / valid_mask.sum() if valid_mask.sum() > 0 else 0
            all_scene_metrics[sample_name] = {
                "accuracy": float(accuracy),
                "num_points": int(valid_mask.sum())
            }

# Print summary statistics
if all_scene_metrics:
    accuracies = [metrics["accuracy"] for metrics in all_scene_metrics.values()]
    mean_accuracy = np.mean(accuracies)
    print(f"Mean slice accuracy: {mean_accuracy:.4f}")

print(f"Processed {len(all_predictions)} scenes")