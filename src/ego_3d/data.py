from pathlib import Path

import numpy as np
import open3d as o3d
import typer
from torch.utils.data import Dataset
from load_config import load_cfg


import torch
from functools import partial

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../pointcept_repo")))

from PointTransformerV3.Pointcept.pointcept.datasets import build_dataset, point_collate_fn
from PointTransformerV3.Pointcept.pointcept.utils import comm
from PointTransformerV3.Pointcept.pointcept.datasets.defaults import worker_init_fn


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""


def build_dataloader(cfg, mode="train"):
    assert mode in ["train", "val"]
    dataset_cfg = cfg.data.train if mode == "train" else cfg.data.val
    dataset = build_dataset(dataset_cfg)

    sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset)
        if comm.get_world_size() > 1
        else None
    )

    init_fn = (
        partial(
            worker_init_fn,
            num_workers=cfg.num_worker_per_gpu,
            rank=comm.get_rank(),
            seed=cfg.seed,
        )
        if cfg.seed is not None
        else None
    )

    collate = (
        partial(point_collate_fn, mix_prob=cfg.mix_prob)
        if mode == "train"
        else None  # Use default or different for val
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size_per_gpu if mode == "train" else cfg.batch_size_val_per_gpu,
        shuffle=(sampler is None and mode == "train"),
        sampler=sampler,
        num_workers=cfg.num_worker_per_gpu,
        pin_memory=True,
        drop_last=True if mode == "train" else False,
        collate_fn=collate,
        worker_init_fn=init_fn,
        persistent_workers=True,
    )
    return dataloader




def print_point_cloud_stats(point_cloud):
    """
    Print statistics about point cloud data.
    """
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    normals = np.asarray(point_cloud.normals)

    print("--------------- SHAPE ---------------")
    print(points.shape, colors.shape, normals.shape)
    print("--------------- DATA ---------------")
    print(points[:5, :], colors[:5, :], normals[:5, :])
    print("--------------- CHECK NAN VALUES ---------------")
    print(np.isnan(points).any(), np.isnan(colors).any(), np.isnan(normals).any())
    print("--------------- BOUNDARIES ---------------")
    print(f"POINTS - Min: {points.min(axis=0)} Max: {points.max(axis=0)}")
    print(f"COLORS - Min: {colors.min(axis=0)} Max: {colors.max(axis=0)}")
    print(f"NORMS - Min: {normals.min(axis=0)} Max: {normals.max(axis=0)}")
    print("--------------- SIZE ---------------")
    print(f"POINTS: {points.nbytes / 1e6} MB")
    print(f"COLORS: {colors.nbytes / 1e6} MB")
    print(f"NORMS: {normals.nbytes / 1e6} MB")



def cut_point_cloud(point_cloud, camera_pos, phi, theta, dubleAlpha, dubleBeta):
    """
    Cut the point cloud based on the camera position and the specified angles. Shifts the origin to the camera position.
    Args:
        point_cloud: open3d.geometry.PointCloud
        camera_pos: camera position np.array([x, y, z]) of float
        phi: angle between the x-axis of the new coordinate system and the projection of the direction view vector on the xy-plane
        theta: angle between the z-axis of the new coordinate system and the direction view vector
        dubleAlpha: horizontal field of view
        dubleBeta: vertical field of view
    """
    # Convert all angles to radians
    phi = np.radians(phi)
    theta = np.radians(theta)
    dubleAlpha = np.radians(dubleAlpha)
    dubleBeta = np.radians(dubleBeta)
    
    # Extract the data from the point cloud object
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    norms = np.asarray(point_cloud.normals)

    # Shift the origin to the camera position
    points_centered = points - camera_pos

    # Calculate angle on the xy-plane and check if the point is inside the alpha range
    gammas = np.arctan2(points_centered[:, 1], points_centered[:, 0]) % (2 * np.pi)
    is_inside_alpha = np.logical_and(gammas >= phi - dubleAlpha/2, gammas <= phi + dubleAlpha/2)
    
    # Keep only the points that are inside the alpha range
    points_centered = points_centered[is_inside_alpha]
    colors = colors[is_inside_alpha]
    norms = norms[is_inside_alpha]
    gammas = gammas[is_inside_alpha]

    # Calculate the angle between the z-axis and the point
    omegas = np.arctan2(np.sqrt(np.pow(points_centered[:, 0], 2) + np.pow(points_centered[:, 1], 2))*np.cos(phi - gammas), points_centered[:, 2])
    is_inside_beta = np.logical_and(omegas >= theta - dubleBeta/2, omegas <= theta + dubleBeta/2)

    # Keep only the points that are also inside the beta range and shift the origin back to the camera position
    filtered_points = points_centered[is_inside_beta] + camera_pos
    filtered_colors = colors[is_inside_beta]
    filtered_normals = norms[is_inside_beta]

    # Create a new point cloud object with the filtered data
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(filtered_points)
    result.colors = o3d.utility.Vector3dVector(filtered_colors)
    result.normals = o3d.utility.Vector3dVector(filtered_normals)

    return result

    

if __name__ == "__main__":
    # Load the data
    pcd = o3d.io.read_point_cloud("data/raw/scene0000_00_vh_clean.ply")

    print_point_cloud_stats(pcd)
    center = pcd.get_center()
    filtered_points = cut_point_cloud(pcd, center, 45, 90, 40, 180)

    o3d.io.write_point_cloud("data/processed/visible_points.ply", filtered_points)
    
    
    
    cfg = load_cfg()
    train_loader = build_dataloader(cfg, mode="train")
    print(train_loader)

