import numpy as np
import matplotlib.pyplot as plt
import os


data_path = "/dtu/blackhole/0e/169006/ScanNet/ego_sliced/preprocessed"
single_prediction_path = "reports/pointcloud/scene0300_01_filtered_point_cloud_slice_900_pred.npy"

CLASS_LABELS_20 = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa", 
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
)

VALID_CLASS_IDS_20 = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39
)

SCANNET_COLOR_MAP_20 = {
    1: (174.0/255, 199.0/255, 232.0/255),  # wall
    2: (152.0/255, 223.0/255, 138.0/255),  # floor
    3: (31.0/255, 119.0/255, 180.0/255),   # cabinet
    4: (255.0/255, 187.0/255, 120.0/255),  # bed
    5: (188.0/255, 189.0/255, 34.0/255),   # chair
    6: (140.0/255, 86.0/255, 75.0/255),    # sofa
    7: (255.0/255, 152.0/255, 150.0/255),  # table
    8: (214.0/255, 39.0/255, 40.0/255),    # door
    9: (197.0/255, 176.0/255, 213.0/255),  # window
    10: (148.0/255, 103.0/255, 189.0/255), # bookshelf
    11: (196.0/255, 156.0/255, 148.0/255), # picture
    12: (23.0/255, 190.0/255, 207.0/255),  # counter
    14: (247.0/255, 182.0/255, 210.0/255), # desk
    16: (66.0/255, 188.0/255, 102.0/255),  # curtain
    24: (219.0/255, 219.0/255, 141.0/255), # refrigerator
    28: (140.0/255, 57.0/255, 197.0/255),  # shower curtain
    33: (202.0/255, 185.0/255, 52.0/255),  # toilet
    34: (51.0/255, 176.0/255, 203.0/255),  # sink
    36: (200.0/255, 54.0/255, 131.0/255),  # bathtub
    39: (92.0/255, 193.0/255, 61.0/255),   # otherfurniture
    -1: (0.0, 0.0, 0.0),                   # ignore/unknown
}

def visualize_slice(scene_folder, slice_name, prediction_path=None):
    """
    Visualize a point cloud slice with its ground truth and prediction
    
    Args:
        scene_folder: path to the scene folder
        slice_name: name of the slice
        prediction_path: path to the prediction file (optional)
    """
    slice_path = os.path.join(data_path, scene_folder, slice_name)

    coords = np.load(os.path.join(slice_path, "coord.npy"))
    colors = np.load(os.path.join(slice_path, "color.npy"))
    gt_labels = np.load(os.path.join(slice_path, "segment20.npy"))
    
    predictions = np.load(prediction_path)
    
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: Original Point Cloud with RGB colors
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colors, s=1)
    ax1.set_title('Original Point Cloud')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.grid(False)
    
    # Convert class indices to colors for ground truth
    gt_colors = np.zeros((len(gt_labels), 3))
    for i, label in enumerate(gt_labels):
        if label >= 0 and label < len(VALID_CLASS_IDS_20):
            class_id = VALID_CLASS_IDS_20[label]
            gt_colors[i] = SCANNET_COLOR_MAP_20[class_id]
    
    # Plot 2: Ground Truth Segmentation with correct colors
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=gt_colors, s=1)
    ax2.set_title('Ground Truth Segmentation')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.grid(False)
    
    # Create legend for ground truth
    unique_labels = np.unique(gt_labels)
    unique_labels = unique_labels[unique_labels >= 0]  # Filter out ignore index
    legend_elements = []
    for label in unique_labels:
        if label < len(CLASS_LABELS_20):
            class_name = CLASS_LABELS_20[label]
            class_id = VALID_CLASS_IDS_20[label]
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=SCANNET_COLOR_MAP_20[class_id], 
                                  markersize=8, label=f'{label}: {class_name}'))
    
    # Add legend outside the plot
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 3: Predicted Segmentation
    # Convert prediction indices to colors
    pred_colors = np.zeros((len(predictions), 3))
    for i, label in enumerate(predictions):
        if label >= 0 and label < len(VALID_CLASS_IDS_20):
            class_id = VALID_CLASS_IDS_20[label]
            pred_colors[i] = SCANNET_COLOR_MAP_20[class_id]
    
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=pred_colors, s=1)
    ax3.set_title('Predicted Segmentation')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.grid(False)
    
    # Legend for predictions
    unique_preds = np.unique(predictions)
    unique_preds = unique_preds[unique_preds >= 0]  # Filter out ignore index
    pred_legend_elements = []
    for label in unique_preds:
        if label < len(CLASS_LABELS_20):
            class_name = CLASS_LABELS_20[label]
            class_id = VALID_CLASS_IDS_20[label]
            pred_legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=SCANNET_COLOR_MAP_20[class_id],
                                    markersize=8, label=f'{label}: {class_name}'))
    
    ax3.legend(handles=pred_legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    os.makedirs('reports/figures', exist_ok=True)
    
    scene_folder = scene_folder.replace('val/', '')
    fig_path = f'reports/figures/{scene_folder}_{slice_name}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {fig_path}")

def visualize_scene(scene_folder, prediction_path=None, data_root=data_path):
    """
    Visualize a scene with its ground truth and prediction
    
    Args:
        scene_folder: path to the scene folder
        prediction_path: path to the prediction file (optional)
    """
    scene_path = os.path.join(data_root, scene_folder)

    print(f"loading from scene {scene_path}")

    coords = np.load(os.path.join(scene_path, "coord.npy"))
    colors = np.load(os.path.join(scene_path, "color.npy"))
    gt_labels = np.load(os.path.join(scene_path, "segment20.npy"))

    predictions = np.load(prediction_path)

    print(f"coords shape: {coords.shape}")
    print(f"predictions shape: {predictions.shape}")
    print(f"gt_labels shape: {gt_labels.shape}")

    fig = plt.figure(figsize=(18, 6))

    # Normalize the colors to 0-1 range if they're not already
    if colors.max() > 1.0:
        colors = colors / 255.0
    

    # Plot 1: Original Point Cloud with RGB colors
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colors, s=1)
    ax1.set_title('Original Point Cloud')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.grid(False)

    # Convert class indices to colors for ground truth
    gt_colors = np.zeros((len(gt_labels), 3))
    for i, label in enumerate(gt_labels):
        if label >= 0 and label < len(VALID_CLASS_IDS_20):
            class_id = VALID_CLASS_IDS_20[label]
            gt_colors[i] = SCANNET_COLOR_MAP_20[class_id]

    # Plot 2: Ground Truth Segmentation with correct colors
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=gt_colors, s=1)
    ax2.set_title('Ground Truth Segmentation')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.grid(False)

    # Create legend for ground truth
    unique_labels = np.unique(gt_labels)
    unique_labels = unique_labels[unique_labels >= 0]  # Filter out ignore index
    legend_elements = []
    for label in unique_labels:
        if label < len(CLASS_LABELS_20):
            class_name = CLASS_LABELS_20[label]
            class_id = VALID_CLASS_IDS_20[label]
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=SCANNET_COLOR_MAP_20[class_id],
                                  markersize=8, label=f'{label}: {class_name}'))
            
    # Add legend outside the plot
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    
    # Plot 3: Predicted Segmentation
    # Convert prediction indices to colors
    pred_colors = np.zeros((len(predictions), 3))
    for i, label in enumerate(predictions):
        if label >= 0 and label < len(VALID_CLASS_IDS_20):
            class_id = VALID_CLASS_IDS_20[label]
            pred_colors[i] = SCANNET_COLOR_MAP_20[class_id]
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=pred_colors, s=1)
    ax3.set_title('Predicted Segmentation')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.grid(False)

    # Legend for predictions
    unique_preds = np.unique(predictions)
    unique_preds = unique_preds[unique_preds >= 0]  # Filter out ignore index
    pred_legend_elements = []
    for label in unique_preds:
        if label < len(CLASS_LABELS_20):
            class_name = CLASS_LABELS_20[label]
            class_id = VALID_CLASS_IDS_20[label]
            pred_legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=SCANNET_COLOR_MAP_20[class_id],
                                    markersize=8, label=f'{label}: {class_name}'))
    
    ax3.legend(handles=pred_legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    os.makedirs('reports/scannet/figures', exist_ok=True)

    fig_path = f"reports/scannet/figures/{scene_folder}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {fig_path}")


    

if __name__ == "__main__":
    pred_filename = os.path.basename(single_prediction_path)
    # Assuming filename format is sceneXXXX_XX_filtered_point_cloud_slice_XXX_pred.npy
    parts = pred_filename.split('_')
    scene_id = f"{parts[0]}_{parts[1]}"
    slice_id = f"filtered_point_cloud_slice_{parts[-2]}"
    visualize_slice(f"val/{scene_id}", slice_id, single_prediction_path)


