from pathlib import Path
import dataclasses
import json
import math
import numpy as np 
import os 
from torchvision.io import read_image
import torch
import tempfile
import time
import nibabel 
 


###this is the piece of code that transform JPG images into torch tensors.
### Bacause that is the expected input by most models. This will save us some time later
### The fucntion can be called during inference as well 
### This is useless
def get_volume(base_dir, save_torch_dir=None):
    """
    Process tomogram folders one at a time, saving each to disk without 
    keeping all volumes in memory.
    """
    # Create save directory if it doesn't exist
    if save_torch_dir is not None:
        os.makedirs(save_torch_dir, exist_ok=True)
    
    # Process each folder (each folder = one tomogram)
    for folder in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            volume_slices = []  # Reset for each folder so memory doesn't kill us 
            
            # Collect all image slices from this folder
            for image_name in sorted(os.listdir(folder_path)):
                image_path = os.path.join(folder_path, image_name)
                img_tensor = read_image(image_path)
                volume_slices.append(img_tensor)
            
            # Stack slices to create a 3D volume for this tomogram
            volume = torch.stack(volume_slices, dim=0)
            
            # Save if a save directory is provided
            if save_torch_dir is not None:
                # Create proper save path
                save_path = os.path.join(save_torch_dir, f"{folder}.pt")
                
                # Save the volume
                torch.save(volume, save_path)
                print(f"Saved tomogram for {folder} with shape {volume.shape} to {save_path}")
            
            # Clear memory
            del volume_slices
            del volume
            import gc
            gc.collect()
            torch.cuda.empty_cache()  # If using GPU
    
    print("All tomograms processed and saved successfully")
    return None  # No need to return volumes dictionary

# If you need to process a specific tomogram and return it:
def get_single_volume(base_dir, tomo_id, save_torch_dir=None):
    """
    Process a single tomogram and optionally save it.
    Returns the processed volume.
    """
    folder_path = os.path.join(base_dir, tomo_id)
    if not os.path.isdir(folder_path):
        print(f"Folder {folder_path} does not exist")
        return None
        
    volume_slices = []
    
    # Collect all image slices from this folder
    for image_name in sorted(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, image_name)
        img_tensor = read_image(image_path)
        volume_slices.append(img_tensor)
    
    # Stack slices to create a 3D volume
    volume = torch.stack(volume_slices, dim=0)
    
    # Save if a save directory is provided
    if save_torch_dir is not None:
        os.makedirs(save_torch_dir, exist_ok=True)
        save_path = os.path.join(save_torch_dir, f"{tomo_id}.pt")
        torch.save(volume, save_path)
        print(f"Saved tomogram for {tomo_id} with shape {volume.shape} to {save_path}")
    
    return volume





if __name__ == "__main__":
    get_volume(
        "/mnt/raid0/DS/byu-locating-bacterial-flagellar-motors-2025/train",
        save_torch_dir="/mnt/raid0/DS/byu-locating-bacterial-flagellar-motors-2025/ds_torch/train/"
    )
