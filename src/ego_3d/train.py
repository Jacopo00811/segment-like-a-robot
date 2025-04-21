import torch
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3
import os
from importlib.util import spec_from_file_location, module_from_spec
from dataset import EgoSlicedScanNetDataset, single_sample_collate_fn
from tqdm import tqdm

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


config_path = "/zhome/f9/0/168881/Desktop/segment-like-a-robot/Pointcept/configs/scannet/semseg-pt-v3m1-0-base.py"
weights_path = "models/model_best_PointTransformer_V3.pth"
data_path = "/dtu/blackhole/0e/169006/ScanNet/ego_sliced/preprocessed"

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

dataset = EgoSlicedScanNetDataset(
    split="val",
    data_root=data_path,
    slice_sampling=100,
    filter_slices=True,  # Enable built-in filtering
    min_size_threshold=0.15  # Filter out slices smaller than 15% of scene average
)
dataloader_val = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=single_sample_collate_fn,  # Use the custom collate function
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()


# Run inference
with torch.no_grad():
    for batch_idx, input_dict in enumerate(tqdm(dataloader_val, desc="Processing slices")):
        # Move input data to device
        for key in input_dict:
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].to(device)
        
        # Get sample name directly from dataset
        sample_name = dataset.get_data_name(batch_idx)
        
        # Forward pass
        print(f"Keys before: {input_dict.keys()}")
        outputs = model(input_dict)
        print(f"Keys after: {outputs.keys()}")