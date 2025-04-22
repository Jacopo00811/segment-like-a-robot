import os
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
import argparse

# ScanNet classes and color mappings
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

def visualize_point_cloud(points, colors=None, title="Point Cloud", save_path="point_cloud.html"):
    """
    Visualizes a point cloud using Plotly.

    Args:
        points (numpy.ndarray): A Nx3 array of points.
        colors (numpy.ndarray, optional): A Nx3 array of colors. Defaults to None.
        title (str, optional): Title of the plot. Defaults to "Point Cloud".
        save_path (str, optional): Path to save the HTML file. Defaults to "point_cloud.html".

    Returns:
        None: Saves the plot as an HTML file.
    """
    # Convert colors to string format for plotly
    if colors is not None:
        colors_str = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' 
                      for r, g, b in colors]
    else:
        # Default color if none provided
        colors_str = ['rgb(100,100,100)'] * len(points)
    
    # Create the scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=colors_str,
            opacity=0.8
        ),
        hoverinfo='none'  # Disable hover for better performance with large point clouds
    )])
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'  # Preserves the data aspect ratio
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    # Save the plot
    plot(fig, filename=save_path, auto_open=False)
    print(f"Point cloud visualization saved to {save_path}")
    return save_path

def create_interactive_visualization(scene_folder, prediction_path, data_root=None, output_dir='reports/scannet/interactive'):
    """
    Create an interactive 3D visualization of point cloud predictions using Plotly
    
    Args:
        scene_folder: path to the scene folder
        prediction_path: path to the prediction file (.npy)
        data_root: base path to data (optional)
        output_dir: directory to save output HTML files
    
    Returns:
        Path to the saved HTML file
    """
    if data_root is None:
        # Try to determine data_root from scene_folder
        if os.path.isabs(scene_folder):
            data_root = os.path.dirname(os.path.dirname(scene_folder))
        else:
            data_root = '.'
    
    # Load data
    scene_path = os.path.join(data_root, scene_folder)
    print(f"Loading data from {scene_path}")
    
    coords = np.load(os.path.join(scene_path, "coord.npy"))
    predictions = np.load(prediction_path)
    
    # Print shapes for debugging
    print(f"Coordinates shape: {coords.shape}")
    print(f"Predictions shape: {predictions.shape}")
    
    # Handle size mismatch
    valid_points = min(len(coords), len(predictions))
    coords_subset = coords[:valid_points]
    pred_labels = predictions[:valid_points]
    
    # Convert prediction indices to colors
    pred_colors = np.zeros((valid_points, 3))
    class_counts = {}
    
    for i, label in enumerate(pred_labels):
        label_int = int(label)
        # Map the label to the ScanNet class ID
        if 0 <= label_int < len(VALID_CLASS_IDS_20):
            class_id = VALID_CLASS_IDS_20[label_int]
            if class_id in SCANNET_COLOR_MAP_20:
                pred_colors[i] = SCANNET_COLOR_MAP_20[class_id]
                
                # Count instances of each class
                class_name = CLASS_LABELS_20[label_int]
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
        else:
            # Use black for invalid labels
            pred_colors[i] = SCANNET_COLOR_MAP_20[-1]
    
    # Create directory for output if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the visualization
    scene_name = os.path.basename(scene_folder)
    html_path = os.path.join(output_dir, f"{scene_name}_interactive.html")
    
    # Generate hover text with point info and predicted class
    hover_text = []
    for i, label in enumerate(pred_labels):
        if 0 <= int(label) < len(CLASS_LABELS_20):
            class_name = CLASS_LABELS_20[int(label)]
            hover_text.append(f"Point {i}: {class_name}")
        else:
            hover_text.append(f"Point {i}: unknown")
    
    # Create the figure with hover information
    fig = go.Figure(data=[go.Scatter3d(
        x=coords_subset[:, 0],
        y=coords_subset[:, 1],
        z=coords_subset[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=[f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in pred_colors],
            opacity=0.8
        ),
        text=hover_text,
        hoverinfo='text'
    )])
    
    # Create a title with class distribution information
    title = f"3D Segmentation - {scene_name}<br>Showing {valid_points}/{len(coords)} points"
    
    # Add class distribution to the legend
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=60)
    )
    
    # Add the class legend as shapes
    legend_items = []
    top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Create an invisible trace for each class in the legend
    for idx, (class_name, count) in enumerate(top_classes):
        class_idx = CLASS_LABELS_20.index(class_name)
        class_id = VALID_CLASS_IDS_20[class_idx]
        color = SCANNET_COLOR_MAP_20[class_id]
        rgb_color = f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})'
        
        # Add invisible point with legend entry
        fig.add_trace(go.Scatter3d(
            x=[coords_subset[0, 0]],
            y=[coords_subset[0, 1]],
            z=[coords_subset[0, 2]],
            mode='markers',
            marker=dict(
                size=10,
                color=rgb_color,
                opacity=1
            ),
            name=f"{class_name} ({count} points)",
            showlegend=True,
            visible="legendonly"  # Points are hidden but shown in legend
        ))
    
    # Save the figure
    plot(fig, filename=html_path, auto_open=False)
    print(f"Interactive visualization saved to {html_path}")
    
    return html_path

def create_segmentation_comparison(scene_folder, prediction_path, data_root=None, output_dir='reports/scannet/interactive'):
    """
    Create an interactive comparison between original RGB point cloud, ground truth segmentation, 
    and predicted segmentation
    
    Args:
        scene_folder: path to the scene folder
        prediction_path: path to the prediction file (.npy)
        data_root: base path to data (optional)
        output_dir: directory to save output HTML files
    
    Returns:
        Path to the saved HTML file
    """
    if data_root is None:
        # Try to determine data_root from scene_folder
        if os.path.isabs(scene_folder):
            data_root = os.path.dirname(os.path.dirname(scene_folder))
        else:
            data_root = '.'
    
    # Load data
    scene_path = os.path.join(data_root, scene_folder)
    print(f"Loading data from {scene_path}")
    
    coords = np.load(os.path.join(scene_path, "coord.npy"))
    colors = np.load(os.path.join(scene_path, "color.npy"))
    gt_labels = np.load(os.path.join(scene_path, "segment20.npy"))
    predictions = np.load(prediction_path)
    
    # Print shapes for debugging
    print(f"Coordinates shape: {coords.shape}")
    print(f"Original colors shape: {colors.shape}")
    print(f"GT labels shape: {gt_labels.shape}")
    print(f"Predictions shape: {predictions.shape}")
    
    # Handle size mismatch for predictions
    valid_points = min(len(coords), len(predictions))
    
    # Create directory for output if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # File paths for individual visualizations
    scene_name = os.path.basename(scene_folder)
    rgb_path = os.path.join(output_dir, f"{scene_name}_rgb.html")
    gt_path = os.path.join(output_dir, f"{scene_name}_gt.html")
    pred_path = os.path.join(output_dir, f"{scene_name}_pred.html")
    
    # 1. Original RGB Point Cloud
    # Normalize colors if needed
    if colors.max() > 1.0:
        normalized_colors = colors / 255.0
    else:
        normalized_colors = colors
    
    visualize_point_cloud(
        coords, 
        normalized_colors, 
        title=f"Original RGB Point Cloud - {scene_name}", 
        save_path=rgb_path
    )
    
    # 2. Ground Truth Segmentation
    # Convert class indices to colors for ground truth
    gt_colors = np.zeros_like(coords)
    for i, label in enumerate(gt_labels):
        label_int = int(label)
        if 0 <= label_int < len(VALID_CLASS_IDS_20):
            class_id = VALID_CLASS_IDS_20[label_int]
            gt_colors[i] = SCANNET_COLOR_MAP_20[class_id]
        else:
            gt_colors[i] = SCANNET_COLOR_MAP_20[-1]  # Default color for unknown
    
    visualize_point_cloud(
        coords, 
        gt_colors, 
        title=f"Ground Truth Segmentation - {scene_name}", 
        save_path=gt_path
    )
    
    # 3. Predicted Segmentation
    # Convert prediction indices to colors
    pred_colors = np.zeros((valid_points, 3))
    for i, label in enumerate(predictions[:valid_points]):
        label_int = int(label)
        if 0 <= label_int < len(VALID_CLASS_IDS_20):
            class_id = VALID_CLASS_IDS_20[label_int]
            pred_colors[i] = SCANNET_COLOR_MAP_20[class_id]
        else:
            pred_colors[i] = SCANNET_COLOR_MAP_20[-1]  # Default color for unknown
    
    # Create a detailed interactive prediction visualization
    create_interactive_visualization(scene_folder, prediction_path, data_root, output_dir)
    
    return {
        "rgb": rgb_path,
        "ground_truth": gt_path,
        "prediction": pred_path
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create interactive 3D visualization of point cloud predictions')
    parser.add_argument('scene_folder', help='Path to the scene folder')
    parser.add_argument('prediction_path', help='Path to the prediction file (.npy)')
    parser.add_argument('--data_root', help='Base path to data (optional)', default=None)
    parser.add_argument('--output_dir', help='Directory to save output HTML files', default='reports/scannet/interactive')
    parser.add_argument('--comparison', action='store_true', help='Create comparison visualizations of RGB, GT, and predictions')
    
    args = parser.parse_args()
    
    if args.comparison:
        result = create_segmentation_comparison(args.scene_folder, args.prediction_path, args.data_root, args.output_dir)
        print(f"Created comparison visualizations at: {args.output_dir}")
    else:
        html_path = create_interactive_visualization(args.scene_folder, args.prediction_path, args.data_root, args.output_dir)
        print(f"Created interactive visualization at: {html_path}")