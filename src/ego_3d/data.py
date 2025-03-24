# from pathlib import Path

# import typer
# from torch.utils.data import Dataset


# class MyDataset(Dataset):
#     """My custom dataset."""

#     def __init__(self, data_path: Path) -> None:
#         self.data_path = data_path

#     def __len__(self) -> int:
#         """Return the length of the dataset."""

#     def __getitem__(self, index: int):
#         """Return a given sample from the dataset."""

#     def preprocess(self, output_folder: Path) -> None:
#         """Preprocess the raw data and save it to the output folder."""

# def preprocess(data_path: Path, output_folder: Path) -> None:
#     print("Preprocessing data...")
#     dataset = MyDataset(data_path)
#     dataset.preprocess(output_folder)


# if __name__ == "__main__":
#     typer.run(preprocess)



import numpy as np
import open3d as o3d

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
