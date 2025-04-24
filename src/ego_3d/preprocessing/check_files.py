import os
import re

def count_scenes(directory_path):
    # Get all files/folders in the directory
    items = os.listdir(directory_path)
    
    # Use a set to store unique scene identifiers
    unique_scenes = set()
    
    # Regular expression to extract scene identifier (e.g., "scene0011" from "scene0011_00_slice_000")
    scene_pattern = re.compile(r'(scene\d+)_\d+_slice_\d+')
    
    for item in items:
        match = scene_pattern.match(item)
        if match:
            scene_id = match.group(1)
            unique_scenes.add(scene_id)
    
    return len(unique_scenes)

# Replace with the path to your 'preprocessed/test' directory
directory_path = '/dtu/blackhole/0e/169006/ScanNet/ego_sliced/preprocessed/val'
scene_count = count_scenes(directory_path)

print(f"Number of unique scenes: {scene_count}")