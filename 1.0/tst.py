path = '/mnt/raid0/Kaggle/DS/new_byu-locating-bacterial-flagellar-motors-2025/test/tomo_00e047'
from torchvision.io import decode_image 
import torch
import os 
import cv2
import numpy as np


def tiling(path,tomo_id, z1, z2, y1, y2, x1, x2): 
#path should to directly to the tomogram file 
#We need the two diagonal points to figure out the cube
#Instead of working out the tiles in the function before training time, we find out the coordinates and extract them 
#live during the training. 
    tomo_path = os.path.join(path,tomo_id)
    volume = []
    all_images = [f for f in os.listdir(tomo_path) if os.path.isfile(os.path.join(tomo_path, f))]
    slice_to_file = {int(os.path.splitext(img)[0].replace('slice_', '')): img for img in all_images}
    for slice_idx in range(z1, z2):
            if slice_idx in slice_to_file:
                image = slice_to_file[slice_idx]
                full_path = os.path.join(tomo_path, image)
                img = decode_image(full_path)         
                #img = img[:, y1:y2, x1:x2]   
                volume.append(img)
            else:
                raise ValueError(f"Slice {slice_idx} missing for tomo_id {tomo_id}.")

    volume = torch.stack(volume, dim=0).float()# (D, 1, H, W)
    print(f"==>> volume: {volume.shape}")
    volume = volume.permute(1, 0, 2, 3)  #make this shape (C, Z, Y, X) ###Ensure that this is correct
    print(f"==>> volume: {volume.shape}")
    #Makes sure that it is 
    return volume




class TomogramTiler:
    def __init__(self, base_path):
        self.base_path = base_path
        self._slice_mappings = {}  # Cache directory listings
    
    def _get_slice_mapping(self, tomo_id):
        """Cache the slice-to-file mapping per tomogram"""
        if tomo_id not in self._slice_mappings:
            tomo_path = os.path.join(self.base_path, tomo_id)
            all_images = [f for f in os.listdir(tomo_path) 
                         if f.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
            
            slice_mapping = {}
            for img in all_images:
                try:
                    slice_idx = int(os.path.splitext(img)[0].replace('slice_', ''))
                    slice_mapping[slice_idx] = img
                except ValueError:
                    continue  # Skip files that don't match pattern
            
            self._slice_mappings[tomo_id] = slice_mapping
        
        return self._slice_mappings[tomo_id]
    
    def extract_tile(self, tomo_id, z1, z2, y1, y2, x1, x2):
        """Extract a 3D tile from tomogram slices"""
        slice_mapping = self._get_slice_mapping(tomo_id)
        tomo_path = os.path.join(self.base_path, tomo_id)
        
        # Pre-allocate volume tensor
        tile_depth = z2 - z1
        tile_height = y2 - y1
        tile_width = x2 - x1
        
        volume = torch.zeros((tile_depth, tile_height, tile_width), dtype=torch.float32)
        
        for i, slice_idx in enumerate(range(z1, z2)):
            if slice_idx not in slice_mapping:
                raise ValueError(f"Slice {slice_idx} missing for tomo_id {tomo_id}")
            
            image_file = slice_mapping[slice_idx]
            full_path = os.path.join(tomo_path, image_file)
            
            # Load as grayscale and crop in one step
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load image: {full_path}")
            
            # Crop and normalize
            cropped = img[y1:y2, x1:x2].astype(np.float32) / 255.0
            volume[i] = torch.from_numpy(cropped)
        
        # Add channel dimension: (C, Z, Y, X)
        return volume.unsqueeze(0)



import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

class TomogramInferenceDataset(Dataset):
    def __init__(self, tomogram_path, tile_size=(64, 64, 64), overlap=0.25, transform=None):
        """
        Dataset for systematic inference on entire tomograms
        
        Args:
            tomogram_path: Path to single tomogram directory containing slice images
            tile_size: (depth, height, width) of tiles to extract
            overlap: Fraction of overlap between adjacent tiles (0.0 to 0.5)
            transform: Optional transforms to apply to tiles
        """
        self.tomogram_path = tomogram_path
        self.tile_size = tile_size
        self.overlap = overlap
        self.transform = transform
        
        # Cache the slice mapping once during initialization
        self.slice_mapping = self._build_slice_mapping()
        
        # Get tomogram dimensions
        self.tomogram_shape = self._get_tomogram_shape()
        
        # Generate all tile coordinates for complete coverage
        self.tile_coordinates = self._generate_systematic_tiles()
        
        print(f"Initialized inference dataset:")
        print(f"  Tomogram shape: {self.tomogram_shape}")
        print(f"  Total tiles: {len(self.tile_coordinates)}")
        print(f"  Tile size: {self.tile_size}")
        print(f"  Overlap: {self.overlap}")
    
    def _build_slice_mapping(self):
        """Build slice index to filename mapping (cached for efficiency)"""
        all_files = os.listdir(self.tomogram_path)
        slice_mapping = {}
        
        for filename in all_files:
            if filename.endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                try:
                    # Extract slice index from filename (adapt to your naming convention)
                    slice_idx = int(os.path.splitext(filename)[0].replace('slice_', ''))
                    slice_mapping[slice_idx] = filename
                except ValueError:
                    continue  # Skip files that don't match expected pattern
        
        return slice_mapping
    
    def _get_tomogram_shape(self):
        """Determine tomogram dimensions without loading entire volume"""
        # Get number of slices
        num_slices = len(self.slice_mapping)
        
        # Load one slice to get spatial dimensions
        first_slice_idx = min(self.slice_mapping.keys())
        sample_path = os.path.join(self.tomogram_path, self.slice_mapping[first_slice_idx])
        sample_slice = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
        
        if sample_slice is None:
            raise ValueError(f"Could not load sample slice: {sample_path}")
        
        height, width = sample_slice.shape
        return (num_slices, height, width)  # (Z, Y, X)
    
    def _generate_systematic_tiles(self):
        """Generate tile coordinates for complete tomogram coverage"""
        z_range, y_range, x_range = self.tomogram_shape
        tile_depth, tile_height, tile_width = self.tile_size
        
        # Calculate stride based on overlap
        # overlap=0.25 means 75% stride (25% overlap)
        z_stride = max(1, int(tile_depth * (1 - self.overlap)))
        y_stride = max(1, int(tile_height * (1 - self.overlap)))
        x_stride = max(1, int(tile_width * (1 - self.overlap)))
        
        coordinates = []
        
        # Generate grid positions ensuring complete coverage
        z_positions = list(range(0, z_range, z_stride))
        y_positions = list(range(0, y_range, y_stride))
        x_positions = list(range(0, x_range, x_stride))
        
        # Add final positions to ensure we reach the edges
        if z_positions[-1] + tile_depth < z_range:
            z_positions.append(z_range - tile_depth)
        if y_positions[-1] + tile_height < y_range:
            y_positions.append(y_range - tile_height)
        if x_positions[-1] + tile_width < x_range:
            x_positions.append(x_range - tile_width)
        
        # Generate all tile coordinates
        for z_start in z_positions:
            for y_start in y_positions:
                for x_start in x_positions:
                    # Ensure tiles don't exceed boundaries
                    z1 = max(0, z_start)
                    y1 = max(0, y_start)
                    x1 = max(0, x_start)
                    
                    z2 = min(z_range, z1 + tile_depth)
                    y2 = min(y_range, y1 + tile_height)
                    x2 = min(x_range, x1 + tile_width)
                    
                    # Adjust start coordinates if tile is smaller than expected
                    if z2 - z1 < tile_depth:
                        z1 = max(0, z2 - tile_depth)
                    if y2 - y1 < tile_height:
                        y1 = max(0, y2 - tile_height)
                    if x2 - x1 < tile_width:
                        x1 = max(0, x2 - tile_width)
                    
                    coordinates.append((z1, y1, x1, z2, y2, x2))
        
        return coordinates
    
    def _extract_tile(self, z1, y1, x1, z2, y2, x2):
        """Extract a single tile using your existing tiling logic"""
        volume_slices = []
        
        for slice_idx in range(z1, z2):
            if slice_idx not in self.slice_mapping:
                raise ValueError(f"Missing slice {slice_idx} in tomogram")
            
            # Load slice
            slice_path = os.path.join(self.tomogram_path, self.slice_mapping[slice_idx])
            img = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                raise ValueError(f"Could not load slice: {slice_path}")
            
            # Crop spatial region and normalize
            cropped = img[y1:y2, x1:x2].astype(np.float32) / 255.0
            volume_slices.append(torch.from_numpy(cropped))
        
        # Stack into 3D volume and add channel dimension
        volume = torch.stack(volume_slices, dim=0)  # (Z, Y, X)
        volume = volume.unsqueeze(0)  # (1, Z, Y, X) - add channel dimension
        
        return volume
    
    def __len__(self):
        return len(self.tile_coordinates)
    
    def __getitem__(self, idx):
        """Get a single tile with its metadata"""
        z1, y1, x1, z2, y2, x2 = self.tile_coordinates[idx]
        
        # Extract tile
        tile = self._extract_tile(z1, y1, x1, z2, y2, x2)
        
        # Apply transforms if provided
        if self.transform:
            tile = self.transform(tile)
        
        return {
            'tile': tile,
            'coordinates': (z1, y1, x1, z2, y2, x2),
            'tile_index': idx,
            'center': ((z1 + z2) // 2, (y1 + y2) // 2, (x1 + x2) // 2)
        }
    
    def get_tomogram_shape(self):
        """Return the full tomogram dimensions"""
        return self.tomogram_shape
    
    def get_reconstruction_info(self):
        """Return information needed for reconstructing the full prediction"""
        return {
            'tomogram_shape': self.tomogram_shape,
            'tile_size': self.tile_size,
            'overlap': self.overlap,
            'total_tiles': len(self.tile_coordinates)
        }





# Usage
if __name__ == '__main__':
    path = '/mnt/raid0/Kaggle/DS/new_byu-locating-bacterial-flagellar-motors-2025/train'
    tomo_id = 'tomo_00e047'
    inferenceDS = TomogramInferenceDataset(os.path.join(path,tomo_id),tile_size=(256, 256, 256),overlap=0.0,transform=None)
    print(len(inferenceDS))
    for item in inferenceDS:
        a = item['tile'].shape
        b = item['coordinates']
        c = item['tile_index']
        d = item['center']
        print(f"==>> a: {a}")
        print(f"==>> b: {b}")
        print(f"==>> c: {c}")
        print(f"==>> d: {d}")
    tiler = TomogramTiler(path)
    z1, z2, y1, y2, x1, x2 = 50,100, 60,190, 70, 560
    tile = tiler.extract_tile(tomo_id, z1, z2, y1, y2, x1, x2)
    print(tile.shape)
