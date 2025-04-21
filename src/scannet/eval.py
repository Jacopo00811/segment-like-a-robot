import torch
import os
from importlib.util import spec_from_file_location, module_from_spec
from ego_3d.dataset import EgoSlicedScanNetDataset, single_sample_collate_fn
from tqdm import tqdm
from pointcept.models.default import DefaultSegmentorV2
import numpy as np
from sklearn.metrics import confusion_matrix
from pointcept.datasets.scannet import ScanNetDataset
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model




def build_val_loader(cfg):

    val_loader = None

    val_data = build_dataset(cfg.data['val'])

    val_sampler = None # None for now

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=val_sampler,
        collate_fn=collate_fn,
    )

    return val_loader

def build_model(cfg):
    model = build_model(cfg.model)
    model.eval()
    return model


def eval(cfg, model, val_loader):

    for idx, input_dict in enumerate(tqdm(val_loader, desc="Processing Scenes")):
        
        if idx > 2:
            break

        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
        
        with torch.no_grad():
            output_dict = model(input_dict)
        
        output = output_dict["seg_logits"]
        loss = output_dict["loss"]
        pred = output.max(1)(1)

        segment = input_dict["segment"]

        print("loss: ", loss.item())

    print("Test passed!")
    
def load_config_from_file(config_path):
    abs_config_path = os.path.abspath(config_path)
    spec = spec_from_file_location("config", abs_config_path)
    config_module = module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module


if __name__ == "__main__":

    cfg_path = "./Pointcept/configs/scannet/semseg-pt-v3m1-0-base.py"
    weights_path = "./models/model_best_PointTransformer_V3.pth"
    DATASET_ROOT = "/dtu/blackhole/0e/169006/ScanNet/preprocessed"

    cfg = load_config_from_file(cfg_path)

    cfg.data_root = DATASET_ROOT

    val_loader = build_val_loader(cfg)

    model = build_model(cfg)

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    backbone_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith("backbone."):
            backbone_state_dict[k] = v
        else:
            backbone_state_dict[f"backbone.{k}"] = v

    # Load the weights
    model.load_state_dict(backbone_state_dict, strict=False)
    print(f"Loaded model weights from {weights_path}")



