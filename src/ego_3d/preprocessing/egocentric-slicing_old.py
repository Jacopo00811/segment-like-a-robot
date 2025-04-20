from pathlib import Path
import numpy as np
import open3d as o3d
from .data import cut_point_cloud_npy  # Adjust the import if needed

def main():
    # Define the path to the input raw scene (update this to your local path)
    raw_scene_dir = Path("data/raw/scene0706_00")
    
    # Load the .npy files containing point cloud data.
    coords = np.load(raw_scene_dir / "coord.npy")
    colors = np.load(raw_scene_dir / "color.npy")
    instances = np.load(raw_scene_dir / "instance.npy")
    normals = np.load(raw_scene_dir / "normal.npy")
    segment20 = np.load(raw_scene_dir / "segment20.npy")
    segment200 = np.load(raw_scene_dir / "segment200.npy")
    
    # For visualization, scale colors to 0-1 range.
    # Note: If you want to use original colors for slicing, keep colors unchanged.
    colors_vis = colors / 255.0
    
    # Create and save the original point cloud (using scaled colors for display)
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(coords)
    original_pcd.colors = o3d.utility.Vector3dVector(colors_vis)
    original_pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud("data/processed/original_point_cloud.ply", original_pcd)
    
    # Define the camera position and angular parameters for slicing.
    # Adjust these values as needed.
    camera_pos = np.array([1.0, 0.3, 0.6])  # Example camera position
    phi = 90         # Angle on the xy-plane (degrees)
    theta = 90       # Angle from vertical (degrees)
    dubleAlpha = 130 # Horizontal field of view (degrees)
    dubleBeta = 150  # Vertical field of view (degrees)
    
    # Perform slicing: filter the raw point cloud arrays based on view angles.
    filtered_coords, filtered_colors, filtered_instances, filtered_normals, filtered_segment20, filtered_segment200 = cut_point_cloud_npy(
        coords, colors, instances, normals, segment20, segment200,
        camera_pos, phi, theta, dubleAlpha, dubleBeta
    )
    
    # Create and save the filtered (sliced) point cloud.
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_coords)
    filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    filtered_pcd.normals = o3d.utility.Vector3dVector(filtered_normals)
    o3d.io.write_point_cloud("data/processed/filtered_point_cloud.ply", filtered_pcd)
    
    print("Slicing complete. Original and filtered point clouds have been saved.")

if __name__ == "__main__":
    main()