#!/usr/bin/env python3
import os
import re
import shutil
import argparse

def cleanup_checkpoints(checkpoint_dir, keep_latest=True, keep_best=True):
    if not os.path.exists(checkpoint_dir):
        print(f"Directory {checkpoint_dir} does not exist.")
        return

    files = os.listdir(checkpoint_dir)
    checkpoint_pattern = re.compile(r'checkpoint_(\d+)')
    
    checkpoints = []
    for f in files:
        match = checkpoint_pattern.match(f)
        if match:
            checkpoints.append((int(match.group(1)), f))
    
    checkpoints.sort(key=lambda x: x[0])
    
    if not checkpoints:
        print("No numbered checkpoints found.")
        return

    latest_step, latest_name = checkpoints[-1]
    
    to_delete = []
    for step, name in checkpoints:
        # Keep the latest one
        if keep_latest and step == latest_step:
            print(f"Keeping latest checkpoint: {name}")
            continue
        
        to_delete.append(name)

    # Note: 'best' and 'final' are not matched by the regex so they are safe
    print(f"Found {len(to_delete)} intermediate checkpoints to delete.")
    
    for name in to_delete:
        path = os.path.join(checkpoint_dir, name)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            print(f"Deleted: {name}")
        except Exception as e:
            print(f"Error deleting {name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanup intermediate SIFE checkpoints")
    parser.add_argument("--dir", type=str, default="checkpoints", help="Directory containing checkpoints")
    args = parser.parse_args()
    
    cleanup_checkpoints(args.dir)
