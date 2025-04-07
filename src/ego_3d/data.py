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
    # Do not count NaN values
    print(f"POINTS - Min: {np.nanmin(points, axis=0)} Max: {np.nanmax(points, axis=0)}")
    print(f"COLORS - Min: {np.nanmin(colors, axis=0)} Max: {np.nanmax(colors, axis=0)}")
    print(f"NORMS - Min: {np.nanmin(normals, axis=0)} Max: {np.nanmax(normals, axis=0)}")
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
    Returns:
        open3d.geometry.PointCloud
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


def cut_point_cloud_npy(coords, colors, instances, normals, segment20, segment200,
                       camera_pos, phi, theta, dubleAlpha, dubleBeta):
    """
    Cut the point cloud based on the camera position and the specified angles.
    Args:
        coords: np.array of shape (N, 3) with XYZ coordinates
        colors: np.array of shape (N, 3) with RGB values
        instances: np.array of shape (N) with instance segmentation labels
        normals: np.array of shape (N, 3) with normal vectors
        segment20: np.array of shape (N) with 20-class segmentation labels
        segment200: np.array of shape (N) with 200-class segmentation labels
        camera_pos: camera position np.array([x, y, z]) of float
        phi: angle between the x-axis and the projection of view vector on xy-plane (degrees)
        theta: angle between the z-axis and the direction view vector (degrees)
        dubleAlpha: horizontal field of view (degrees)
        dubleBeta: vertical field of view (degrees)
        
    Returns:
        Tuple of filtered arrays (coords, colors, instances, normals, segment20, segment200)
    """
    # Convert all angles to radians
    phi = np.radians(phi)
    theta = np.radians(theta)
    dubleAlpha = np.radians(dubleAlpha)
    dubleBeta = np.radians(dubleBeta)
    
    # Scale colors in 0-1 range
    colors = colors / 255

    # Shift the origin to the camera position
    points_centered = coords - camera_pos

    # Calculate angle on the xy-plane and check if the point is inside the alpha range
    gammas = np.arctan2(points_centered[:, 1], points_centered[:, 0]) % (2 * np.pi)
    is_inside_alpha = np.logical_and(gammas >= phi - dubleAlpha/2, gammas <= phi + dubleAlpha/2)
    
    # Keep only the points that are inside the alpha range
    points_centered = points_centered[is_inside_alpha]
    filtered_colors = colors[is_inside_alpha]
    filtered_instances = instances[is_inside_alpha]
    filtered_normals = normals[is_inside_alpha]
    filtered_segment20 = segment20[is_inside_alpha]
    filtered_segment200 = segment200[is_inside_alpha]
    gammas = gammas[is_inside_alpha]

    # Calculate the angle between the z-axis and the point
    omegas = np.arctan2(np.sqrt(np.pow(points_centered[:, 0], 2) + np.pow(points_centered[:, 1], 2))*np.cos(phi - gammas), points_centered[:, 2])
    is_inside_beta = np.logical_and(omegas >= theta - dubleBeta/2, omegas <= theta + dubleBeta/2)

    # Keep only the points that are also inside the beta range and shift the origin back to the camera position
    filtered_coords = points_centered[is_inside_beta] + camera_pos
    filtered_colors = filtered_colors[is_inside_beta]
    filtered_instances = filtered_instances[is_inside_beta]
    filtered_normals = filtered_normals[is_inside_beta]
    filtered_segment20 = filtered_segment20[is_inside_beta]
    filtered_segment200 = filtered_segment200[is_inside_beta]

    return (filtered_coords, filtered_colors, filtered_instances, filtered_normals, 
            filtered_segment20, filtered_segment200)













    
if __name__ == "__main__":

    ##### HOW TO USE cut_point_cloud #####
    # pcd = o3d.io.read_point_cloud("data/raw/scene0000_00_vh_clean.ply")
    # print_point_cloud_stats(pcd)
    # center = pcd.get_center()
    # filtered_points = cut_point_cloud(pcd, center, 180, 90, 130, 150)
    # o3d.io.write_point_cloud("data/processed/visible_points.ply", filtered_points)

    ##### HOW TO USE cut_point_cloud_npy #####
    coords = np.load("data/raw/scene0706_00/coord.npy")
    colors = np.load("data/raw/scene0706_00/color.npy")
    instances = np.load("data/raw/scene0706_00/instance.npy")
    normals = np.load("data/raw/scene0706_00/normal.npy")
    segment20 = np.load("data/raw/scene0706_00/segment20.npy")
    segment200 = np.load("data/raw/scene0706_00/segment200.npy")

    # Rescale colors to 0-1 range if you want to save initial point cloud
    # colors = colors / 255

    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(coords)
    original_pcd.colors = o3d.utility.Vector3dVector(colors)
    original_pcd.normals = o3d.utility.Vector3dVector(normals)
    output_path = "data/processed/original_point_cloud.ply"
    o3d.io.write_point_cloud(output_path, original_pcd)

    camera_pos = np.array([1.0, 0.3, 0.6])
    phi = 90 
    theta = 90 
    dubleAlpha = 130
    dubleBeta = 150

    filtered_coords, filtered_colors, filtered_instances, filtered_normals, filtered_segment20, filtered_segment200 = cut_point_cloud_npy(
        coords, colors, instances, normals, segment20, segment200,
        camera_pos, phi, theta, dubleAlpha, dubleBeta
    )

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_coords)
    filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    filtered_pcd.normals = o3d.utility.Vector3dVector(filtered_normals)
    output_path = "data/processed/filtered_point_cloud.ply"
    o3d.io.write_point_cloud(output_path, filtered_pcd)
    
    o3d.io.write_point_cloud("data/processed/visible_points.ply", filtered_points)
    
    
    
    cfg = load_cfg()
    train_loader = build_dataloader(cfg, mode="train")
    print(train_loader)


