from pathlib import Path

import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
import os 
from tqdm import tqdm

from sens_reader.SensorData import SensorData

PROCESSED_OUTPUT_DIR = Path('/dtu/blackhole/0e/169006/ScanNet/ego_sliced/preprocessed/test/') # Ego-sliced processed scenes storage
PREPROCESSED_BASE_DIR = Path('/dtu/blackhole/0e/169006/ScanNet/preprocessed/val/')
RAW_SCANS_BASE_DIR = Path('/dtu/datasets2/ScanNet/ScanNetV2/scans/')
RAW_OUTPUT_DIR = Path('/dtu/blackhole/0e/169006/ScanNet/ego_sliced/raw/val/')

# Testing
# PROCESSED_OUTPUT_DIR = Path('data/raw') # Ego-sliced processed scenes storage
# PREPROCESSED_BASE_DIR = Path('/dtu/blackhole/0e/169006/ScanNet/preprocessed/val/')
# RAW_SCANS_BASE_DIR = Path('/dtu/datasets2/ScanNet/ScanNetV2/scans/')

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


# def build_dataloader(cfg, mode="train"):
#     assert mode in ["train", "val"]
#     dataset_cfg = cfg.data.train if mode == "train" else cfg.data.val
#     dataset = build_dataset(dataset_cfg)

#     sampler = torch.utils.data.distributed.DistributedSampler(dataset) if comm.get_world_size() > 1 else None

#     init_fn = (
#         partial(
#             worker_init_fn,
#             num_workers=cfg.num_worker_per_gpu,
#             rank=comm.get_rank(),
#             seed=cfg.seed,
#         )
#         if cfg.seed is not None
#         else None
#     )

#     collate = (
#         partial(point_collate_fn, mix_prob=cfg.mix_prob)
#         if mode == "train"
#         else None  # Use default or different for val
#     )

#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=cfg.batch_size_per_gpu if mode == "train" else cfg.batch_size_val_per_gpu,
#         shuffle=(sampler is None and mode == "train"),
#         sampler=sampler,
#         num_workers=cfg.num_worker_per_gpu,
#         pin_memory=True,
#         drop_last=True if mode == "train" else False,
#         collate_fn=collate,
#         worker_init_fn=init_fn,
#         persistent_workers=True,
#     )
#     return dataloader


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
    is_inside_alpha = np.logical_and(gammas >= phi - dubleAlpha / 2, gammas <= phi + dubleAlpha / 2)

    # Keep only the points that are inside the alpha range
    points_centered = points_centered[is_inside_alpha]
    colors = colors[is_inside_alpha]
    norms = norms[is_inside_alpha]
    gammas = gammas[is_inside_alpha]

    # Calculate the angle between the z-axis and the point
    omegas = np.arctan2(
        np.sqrt(np.pow(points_centered[:, 0], 2) + np.pow(points_centered[:, 1], 2)) * np.cos(phi - gammas),
        points_centered[:, 2],
    )
    is_inside_beta = np.logical_and(omegas >= theta - dubleBeta / 2, omegas <= theta + dubleBeta / 2)

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


def cut_point_cloud_npy(
    coords, colors, instances, normals, segment20, segment200, camera_pos, phi, theta, dubleAlpha, dubleBeta
):
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
    is_inside_alpha = np.logical_and(gammas >= phi - dubleAlpha / 2, gammas <= phi + dubleAlpha / 2)

    # Keep only the points that are inside the alpha range
    points_centered = points_centered[is_inside_alpha]
    filtered_colors = colors[is_inside_alpha]
    filtered_instances = instances[is_inside_alpha]
    filtered_normals = normals[is_inside_alpha]
    filtered_segment20 = segment20[is_inside_alpha]
    filtered_segment200 = segment200[is_inside_alpha]
    gammas = gammas[is_inside_alpha]

    # Calculate the angle between the z-axis and the point
    omegas = np.arctan2(
        np.sqrt(np.pow(points_centered[:, 0], 2) + np.pow(points_centered[:, 1], 2)) * np.cos(phi - gammas),
        points_centered[:, 2],
    )
    is_inside_beta = np.logical_and(omegas >= theta - dubleBeta / 2, omegas <= theta + dubleBeta / 2)

    # Keep only the points that are also inside the beta range and shift the origin back to the camera position
    filtered_coords = points_centered[is_inside_beta] + camera_pos
    filtered_colors = filtered_colors[is_inside_beta]
    filtered_instances = filtered_instances[is_inside_beta]
    filtered_normals = filtered_normals[is_inside_beta]
    filtered_segment20 = filtered_segment20[is_inside_beta]
    filtered_segment200 = filtered_segment200[is_inside_beta]

    return (
        filtered_coords,
        filtered_colors,
        filtered_instances,
        filtered_normals,
        filtered_segment20,
        filtered_segment200,
    )

def load_camera_poses(poses_root, scene_name):
    """
    Load camera poses from text files stored in poses_root/scene_name/
    
    Each text file is expected to contain a 4x4 matrix, one row per line,
    with values separated by whitespace.
    
    Args:
        poses_root (str or Path): Root directory where poses are stored.
        scene_name (str): The scene folder name.
    
    Returns:
        List[np.ndarray]: List of 4x4 camera pose matrices.
    """
    poses_dir = Path(poses_root) / scene_name
    if not poses_dir.exists(): raise FileNotFoundError(f"Poses directory {poses_dir} not found.")
    
    # assumes files with a .txt extension, sorted alphanumerically:
    pose_files = sorted(poses_dir.glob("*.txt"))
    camera_poses = []
    for pose_file in pose_files:
        pose = np.loadtxt(pose_file)
        if pose.shape != (4, 4):
            raise ValueError(f"Pose file {pose_file} does not have shape (4,4); got shape {pose.shape}")
        camera_poses.append(pose)
    return camera_poses

def extract_camera_poses(scene_name, sens_file):
    """
    Extract camera poses from a .sens file by first exporting them using SensorData,
    then loading the exported text files.
    
    The ScanNet .sens file contains the camera trajectory (and more). We can extract a list of 4x4 
    extrinsic matrices (transformation from world to camera coordinates).
    
    Per the ScanNet repository: RGB-D sensor stream containing color frames, depth frames, camera poses and other data

    Args:
        sens_file (Path): Path to the .sens file.
    
    Returns:
        List[np.ndarray]: A list of 4x4 numpy arrays representing camera extrinsic matrices.
    """

    # sd = SensorData(sens_file)
    # sd.export_poses(os.path.join('poses/', scene_name))
    
    # adjust to large storage space
    poses_root = os.path.join(PROCESSED_OUTPUT_DIR, "poses")
    poses_dir = Path(poses_root) / scene_name

    # if the poses directory does not exist, export the poses from the .sens file.
    if not poses_dir.exists():
        print(f"Poses for scene {scene_name} not found at {poses_dir}. Exporting from {sens_file}...")
        sd = SensorData(sens_file)
        # export pose text files into the given directory.
        # (Assumes SensorData.export_poses writes files named 0.txt, 1.txt, etc.)
        sd.export_poses(str(poses_dir))
    
    # Now load the poses from the text files
    return load_camera_poses(poses_root, scene_name)
    
def ego_slice(scene_name, path_to_scene, path_to_sens_file):    
    # ensure the output directory exists
    PROCESSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # load the .npy files containing point cloud data.
    coords = np.load(Path(path_to_scene) / "coord.npy")
    colors = np.load(Path(path_to_scene) / "color.npy")
    instances = np.load(Path(path_to_scene) / "instance.npy")
    normals = np.load(Path(path_to_scene) / "normal.npy")
    segment20 = np.load(Path(path_to_scene) / "segment20.npy")
    segment200 = np.load(Path(path_to_scene) / "segment200.npy")
    
    # for visualization, scale colors to 0-1 range.
    colors_vis = colors / 255.0
    
    # create and save the original point cloud (using scaled colors for display)
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(coords)
    original_pcd.colors = o3d.utility.Vector3dVector(colors_vis)
    original_pcd.normals = o3d.utility.Vector3dVector(normals)
    # o3d.io.write_point_cloud(str(PROCESSED_OUTPUT_DIR / "original_point_cloud.ply"), original_pcd)
    
    # determine the .sens file path.
    if path_to_sens_file.exists(): camera_poses = extract_camera_poses(scene_name, path_to_sens_file)
    else:
        print(f"[ERROR] Sens file {path_to_sens_file} not found. Using default camera pose.")
        pose1 = np.eye(4)
        pose2 = np.eye(4)
        pose2[:3, 3] = np.array([0.5, 0.0, 0.0])  # sample shift in x-direction
        camera_poses = [pose1, pose2]
    
    # define the camera position and angular parameters for slicing.
    # camera_pos = np.array([1.0, 0.3, 0.6]) # testing
    phi = 90         # Angle on the xy-plane (degrees)
    theta = 90       # Angle from vertical (degrees)
    dubleAlpha = 130 # Horizontal field of view (degrees)
    dubleBeta = 150  # Vertical field of view (degrees)
    
    # # perform slicing: filter the raw point cloud arrays based on view angles.
    # filtered_coords, filtered_colors, filtered_instances, filtered_normals, filtered_segment20, filtered_segment200 = cut_point_cloud_npy(
    #     coords, colors, instances, normals, segment20, segment200,
    #     camera_pos, phi, theta, dubleAlpha, dubleBeta
    # )
    
    
    # loop over every camera pose and slice the point cloud accordingly.
    for i, pose in enumerate(camera_poses):
        camera_pos = pose[:3, 3]  # extract the translation (camera position)
        (filtered_coords, filtered_colors, filtered_instances,
         filtered_normals, filtered_segment20, filtered_segment200) = cut_point_cloud_npy(
            coords, colors, instances, normals, segment20, segment200,
            camera_pos, phi, theta, dubleAlpha, dubleBeta
        )

        # saving raw
        # filtered_pcd = o3d.geometry.PointCloud()
        # filtered_pcd.points = o3d.utility.Vector3dVector(filtered_coords)
        # filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
        # filtered_pcd.normals = o3d.utility.Vector3dVector(filtered_normals)
        
        # out_dir = RAW_OUTPUT_DIR / scene_name
        # out_dir.mkdir(parents=True, exist_ok=True)
        # out_filename = out_dir / f"filtered_point_cloud_slice_{i:03d}.ply"
        # o3d.io.write_point_cloud(str(out_filename), filtered_pcd)

        # Saving Preprocessed
        processed_out_dir = PROCESSED_OUTPUT_DIR / f"{scene_name}_slice_{i:03d}"
        processed_out_dir.mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(str(processed_out_dir), "coord.npy"), filtered_coords)
        np.save(os.path.join(str(processed_out_dir), "color.npy"), filtered_colors)
        np.save(os.path.join(str(processed_out_dir), "instance.npy"), filtered_instances)
        np.save(os.path.join(str(processed_out_dir), "normal.npy"), filtered_normals)
        np.save(os.path.join(str(processed_out_dir), "segment20.npy"), filtered_segment20)
        np.save(os.path.join(str(processed_out_dir), "segment200.npy"), filtered_segment200)

        # print(f"Slice {i} saved to {out_filename}")
    
    print("Slicing complete. Filtered point cloud have been saved.")

if __name__ == "__main__":
    
    print(PROCESSED_OUTPUT_DIR)
    print(RAW_OUTPUT_DIR)
    
    """
    We will only use the validation scene/scan set for now. The preprocessed data
    is located in /dtu/blackhole/0e/169006/ScanNet/preprocessed/val/<scene> and 
    the raw scans (with .sens files) are located in 
    /dtu/datasets2/ScanNet/ScanNetV2/scans/<scene>/<scene>.sens.
    
    The validation scene set should be a subset of the raw scenes, so no errors should I arise,
    we assert some sanity checks nonetheless.
    
    We store the ego-sliced processed scenes in PROCESSED_OUTPUT_DIR
    """
    
    # loop over each scene directory in the preprocessed validation folder
    # ignore sub directories

    for scene_dir in tqdm(PREPROCESSED_BASE_DIR.iterdir()):
        if scene_dir.is_dir():
            scene_name = scene_dir.name
            print(f"Processing scene: {scene_name}")
            
            # construct the path to the .sens file for that scene from the raw scans folder
            path_to_sens_file = RAW_SCANS_BASE_DIR / scene_name / f"{scene_name}.sens"
            
            # check if the constructed .sens file exists
            if not path_to_sens_file.exists():
                print(f"[WARNING] Sens file for scene {scene_name} not found at {path_to_sens_file}. Skipping.")
                continue
            
            ego_slice(scene_name, scene_dir, path_to_sens_file)
    
    print("All scenes have been processed.")
    
    
    ######## RUN Ego-slicing for 1 scene ########
    # scene_name = 'scene0704_01' # Scene must be in validation set
    
    # # confirm path to preprocessed scene exists
    # path_to_scene = Path(f'/dtu/blackhole/0e/169006/ScanNet/preprocessed/val/{scene_name}') # TODO: We should export this to an environment variable or config
    # assert path_to_scene.exists(), f'Path to scene ({path_to_scene}) does not exist'
    
    # # confirm path to .sens file exist
    # path_to_sens_file = Path(f'/dtu/datasets2/ScanNet/ScanNetV2/scans/{scene_name}/{scene_name}.sens')
    # assert path_to_sens_file.exists(), f'Path to sens file ({path_to_sens_file}) does not exist'
    
    # ego_slice(scene_name, path_to_scene, path_to_sens_file)
    ###############################################
    
    
    
    
    # /dtu/datasets2/ScanNet/ScanNetV2/scans/scene0236_01
    
    ##### HOW TO USE cut_point_cloud #####
    # pcd = o3d.io.read_point_cloud("data/raw/scene0000_00_vh_clean.ply")
    # print_point_cloud_stats(pcd)
    # center = pcd.get_center()
    # filtered_points = cut_point_cloud(pcd, center, 180, 90, 130, 150)
    # o3d.io.write_point_cloud("data/processed/visible_points.ply", filtered_points)

    ##### HOW TO USE cut_point_cloud_npy #####
    # coords = np.load("data/raw/scene0706_00/coord.npy")
    # colors = np.load("data/raw/scene0706_00/color.npy")
    # instances = np.load("data/raw/scene0706_00/instance.npy")
    # normals = np.load("data/raw/scene0706_00/normal.npy")
    # segment20 = np.load("data/raw/scene0706_00/segment20.npy")
    # segment200 = np.load("data/raw/scene0706_00/segment200.npy")

    # # Rescale colors to 0-1 range if you want to save initial point cloud
    # # colors = colors / 255

    # original_pcd = o3d.geometry.PointCloud()
    # original_pcd.points = o3d.utility.Vector3dVector(coords)
    # original_pcd.colors = o3d.utility.Vector3dVector(colors)
    # original_pcd.normals = o3d.utility.Vector3dVector(normals)
    # output_path = "data/processed/original_point_cloud.ply"
    # o3d.io.write_point_cloud(output_path, original_pcd)

    # camera_pos = np.array([1.0, 0.3, 0.6])
    # phi = 90
    # theta = 90
    # dubleAlpha = 130
    # dubleBeta = 150

    # filtered_coords, filtered_colors, filtered_instances, filtered_normals, filtered_segment20, filtered_segment200 = (
    #     cut_point_cloud_npy(
    #         coords, colors, instances, normals, segment20, segment200, camera_pos, phi, theta, dubleAlpha, dubleBeta
    #     )
    # )

    # filtered_pcd = o3d.geometry.PointCloud()
    # filtered_pcd.points = o3d.utility.Vector3dVector(filtered_coords)
    # filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    # filtered_pcd.normals = o3d.utility.Vector3dVector(filtered_normals)
    # output_path = "data/processed/filtered_point_cloud.ply"
    # o3d.io.write_point_cloud(output_path, filtered_pcd)

    # o3d.io.write_point_cloud("data/processed/visible_points.ply", filtered_coords)

