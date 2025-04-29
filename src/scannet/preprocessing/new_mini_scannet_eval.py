import os
import re
import shutil

SLICED_DIR = "/dtu/blackhole/0e/169006/Mini-ScanNet/ego_sliced/preprocessed/val"
UNSLICED_DIR = "/dtu/blackhole/0e/169006/Mini-ScanNet/preprocessed/val"
SOURCE_DIR = "/dtu/blackhole/0e/169006/ScanNet/preprocessed/val"

def main():
    print("Analyzing scene directories...")
    sliced_scenes = [f for f in os.listdir(SLICED_DIR) if os.path.isdir(os.path.join(SLICED_DIR, f))]
    
    # Extract base scene names (sceneXXXX_YY) from sliced scenes
    base_scene_pattern = r'(scene\d{4}_\d{2})_slice_\d{3}'
    required_base_scenes = set()
    
    for sliced_scene in sliced_scenes:
        match = re.match(base_scene_pattern, sliced_scene)
        if match:
            required_base_scenes.add(match.group(1))
    
    print(f"These are the base scenes found in sliced directory: \n {sorted(required_base_scenes)}")
    
    existing_unsliced_scenes = set([f for f in os.listdir(UNSLICED_DIR) 
                                   if os.path.isdir(os.path.join(UNSLICED_DIR, f))])
    
    print(f"These are the unsliced scenes found in unsliced directory: \n {sorted(existing_unsliced_scenes)}")

    # Find missing and extra scenes
    missing_scenes = required_base_scenes - existing_unsliced_scenes
    extra_scenes = existing_unsliced_scenes - required_base_scenes
    
    print(f"\nSUMMARY:")
    print(f"- Found {len(required_base_scenes)} unique base scenes from sliced directory")
    print(f"- Found {len(existing_unsliced_scenes)} scenes in unsliced directory")
    print(f"- Missing scenes: {len(missing_scenes)}")
    print(f"- Extra scenes: {len(extra_scenes)}")
    
    # Process missing scenes
    if missing_scenes:
        print("\nMissing scenes (need to be copied):")
        for scene in sorted(missing_scenes):
            print(f"- {scene}")
            
            # Check if scene exists in the main ScanNet dataset
            source_path = os.path.join(SOURCE_DIR, scene)
            if os.path.exists(source_path) and os.path.isdir(source_path):
                print(f"  Found in source directory - copying...")
                try:
                    shutil.copytree(source_path, os.path.join(UNSLICED_DIR, scene))
                    print(f"  Successfully copied {scene}")
                except Exception as e:
                    print(f"  Error copying: {str(e)}")
            else:
                print(f"  NOT FOUND in source directory: {SOURCE_DIR}")
    
    # Process extra scenes
    if extra_scenes:
        print("\nExtra scenes (should be removed):")
        for scene in sorted(extra_scenes):
            print(f"- {scene}")
            scene_path = os.path.join(UNSLICED_DIR, scene)
            print(f"  Removing {scene}...")
            shutil.rmtree(scene_path)
    
    print("\nDone!")
    
    # Safety prompt before making changes
    if missing_scenes or extra_scenes:
        choice = input("\nDo you want to:\n"
                      "1. Copy missing scenes (if found)\n"
                      "2. Remove extra scenes\n"
                      "3. Do both\n"
                      "4. Do nothing\n"
                      "Enter your choice (1-4): ")
        
        if choice == '1' or choice == '3':
            copy_missing_scenes(missing_scenes, SOURCE_DIR, UNSLICED_DIR)
        
        if choice == '2' or choice == '3':
            remove_extra_scenes(extra_scenes, UNSLICED_DIR)

def copy_missing_scenes(missing_scenes, source_dir, dest_dir):
    print("\nCopying missing scenes...")
    for scene in missing_scenes:
        source_path = os.path.join(source_dir, scene)
        dest_path = os.path.join(dest_dir, scene)
        
        if os.path.exists(source_path) and os.path.isdir(source_path):
            try:
                shutil.copytree(source_path, dest_path)
                print(f"✓ Successfully copied {scene}")
            except Exception as e:
                print(f"✗ Error copying {scene}: {str(e)}")
        else:
            print(f"✗ {scene} not found in source directory")

def remove_extra_scenes(extra_scenes, dir_path):
    print("\nRemoving extra scenes...")
    for scene in extra_scenes:
        scene_path = os.path.join(dir_path, scene)
        try:
            shutil.rmtree(scene_path)
            print(f"✓ Removed {scene}")
        except Exception as e:
            print(f"✗ Error removing {scene}: {str(e)}")

if __name__ == "__main__":
    main()