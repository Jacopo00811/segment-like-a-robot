import os
import re
import shutil

def process_directory(directory_path):
    # Get all directories in the path
    all_items = os.listdir(directory_path)
    folders = [item for item in all_items if os.path.isdir(os.path.join(directory_path, item))]
    
    # Group folders by scene number
    scenes = {}
    pattern = re.compile(r'scene(\d+)_(\d+)_slice_(\d+)')
    
    for folder in folders:
        match = pattern.match(folder)
        if match:
            scene_num, scene_part, slice_num = match.groups()
            scene_key = f"{scene_num}_{scene_part}"
            
            if scene_key not in scenes:
                scenes[scene_key] = set()
            
            scenes[scene_key].add(slice_num)
    
    # Find scenes missing either slice_000 or slice_001
    scenes_to_remove = []
    for scene_key, slices in scenes.items():
        if not ('000' in slices and '001' in slices):
            scenes_to_remove.append(scene_key)
    
    # Remove the folders for incomplete scenes
    folders_removed = []
    for folder in folders:
        match = pattern.match(folder)
        if match:
            scene_num, scene_part, _ = match.groups()
            scene_key = f"{scene_num}_{scene_part}"
            
            if scene_key in scenes_to_remove:
                folder_path = os.path.join(directory_path, folder)
                # Uncomment the line below to actually remove folders
                shutil.rmtree(folder_path)
                folders_removed.append(folder)
    
    print(f"The following folders would be removed (remove the # in the code to actually delete):")
    for folder in sorted(folders_removed):
        print(f"  - {folder}")

if __name__ == "__main__":
    # Change this to your directory path
    target_directory = "/dtu/blackhole/0e/169006/Mini-ScanNet/ego_sliced/preprocessed/train"
    process_directory(target_directory)