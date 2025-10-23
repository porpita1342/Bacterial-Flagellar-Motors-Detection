import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F 
import numpy as np
from typing import Union, Tuple, Optional, Callable, List, Dict
import cv2
import math
from monai.transforms import Compose, RandSpatialCrop, SpatialPad, Spacing
from monai.data import MetaTensor 


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
            
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load image: {full_path}")
            
            # Crop and normalize
            cropped = img[y1:y2, x1:x2].astype(np.float32) / 255.0
            volume[i] = torch.from_numpy(cropped)
        
        # Add channel dimension: (C, Z, Y, X)
        return volume.unsqueeze(0)

def get_rand_coords(tomo_row: pd.DataFrame,
                    tile_size: Union[int, Tuple[int, int, int]] = (32, 128, 128),
                    item_rng=None): 
    """Generate random tile coordinates within tomogram bounds"""
    z_range = tomo_row["Array shape (axis 0)"]
    y_range = tomo_row["Array shape (axis 1)"]
    x_range = tomo_row["Array shape (axis 2)"]

    tile_depth, tile_height, tile_width = tile_size[:3]
    if z_range < tile_depth or y_range < tile_height or x_range < tile_width:
        raise ValueError(f"Tile size {tile_size} exceeds tomogram bounds {(z_range, y_range, x_range)}")
    
    min_z_center = tile_depth // 2
    max_z_center = z_range - tile_depth // 2 - 1
    min_y_center = tile_height // 2  
    max_y_center = y_range - tile_height // 2 - 1
    min_x_center = tile_width // 2
    max_x_center = x_range - tile_width // 2 - 1
    
    tile_centre_z = item_rng.randint(min_z_center, max_z_center + 1)
    tile_centre_y = item_rng.randint(min_y_center, max_y_center + 1)
    tile_centre_x = item_rng.randint(min_x_center, max_x_center + 1)

    z1 = int(tile_centre_z - tile_depth // 2)
    y1 = int(tile_centre_y - tile_height // 2)
    x1 = int(tile_centre_x - tile_width // 2)

    z2 = int(tile_centre_z + tile_depth // 2)
    y2 = int(tile_centre_y + tile_height // 2)
    x2 = int(tile_centre_x + tile_width // 2)
    
    # Fixed: consistent coordinate ordering (Z, Y, X)
    return (tile_centre_z, tile_centre_y, tile_centre_x,
            int(z1), int(y1), int(x1), int(z2), int(y2), int(x2))

class CustomDataset(Dataset): 
    def __init__(self,
                train_df: pd.DataFrame,
                img_files_dir: str,
                tile_size: Union[int, Tuple[int, int, int]] = (96,96,96),
                positive_ratio: float = 0.2,
                transform: Optional[Callable] = None,
                dataset_size: int = 1000,
                seed: int = 42,
                target_voxel_spacing: Optional[float] = None):
        
        super().__init__()
        
        if 'Voxel spacing' not in train_df.columns:
            raise ValueError("The DataFrame must contain a 'Voxel spacing' column")

        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.epoch_seed = seed

        # Determine target voxel spacing
        if target_voxel_spacing is None:
            voxel_spacings = train_df['Voxel spacing'].values
            self.target_voxel_spacing = np.median(voxel_spacings)
            print(f"Auto-selected target voxel spacing: {self.target_voxel_spacing:.2f} Angstroms")
        else:
            self.target_voxel_spacing = target_voxel_spacing
            print(f"Using target voxel spacing: {self.target_voxel_spacing:.2f} Angstroms")
        
        # Store original dataframe for reference
        #self.unprocessed_df = train_df.copy()
        
        # Pre-process the dataframe to adjust coordinates and shapes
        self.df = self._preprocess_dataframe_with_voxel_spacing(train_df.copy())
        
        self.img_files_dir = img_files_dir
        self.tile_size = tile_size
        self.transform = transform
        self.dataset_size = dataset_size
        self.unique_tomo_ids = self.df['tomo_id'].unique()
        self.positive_ratio = positive_ratio 
        self.tiler = TomogramTiler(base_path=img_files_dir)
    def _preprocess_dataframe_with_voxel_spacing(self, df):
        """
        Pre-calculate what coordinates and shapes would be after voxel spacing normalization.
        This avoids complex coordinate transformations during training.
        """
        adjusted_df = df.copy()
        
        print("Pre-calculating adjusted coordinates for voxel spacing normalization...")
        
        for idx, row in adjusted_df.iterrows():
            original_spacing = row['Voxel spacing']
            scale_factor = original_spacing / self.target_voxel_spacing
            # Adjust coordinates 
            adjusted_coords = []
            # oui = eval(row['coordinates'])
            # print(f"{oui=}, {type(oui)=}")
            # print(f"{row['coordinates']=}, {type(row['coordinates'])=}")
            for coord in row['coordinates']:
                # print(f'{coord=} || {scale_factor=}')
                adjusted_coord = [
                    math.floor(coord[0] * scale_factor),  # Z
                    math.floor(coord[1] * scale_factor),  # Y  
                    math.floor(coord[2] * scale_factor)   # X
                ]
                if row['coordinates'][0][0] == -1:
                    adjusted_coords.append([-1,-1,-1])
                else:
                    adjusted_coords.append(adjusted_coord)

            adjusted_df.at[idx, 'coordinates'] = adjusted_coords
            
            # Adjust array shapes (tomogram dimensions after resampling)
            for axis in range(3):
                original_shape = row[f'Array shape (axis {axis})']
                new_shape = math.ceil(original_shape * scale_factor)
                adjusted_df.at[idx, f'Array shape (axis {axis})'] = new_shape
            
            # Update voxel spacing to target
            adjusted_df.at[idx, 'Voxel spacing'] = self.target_voxel_spacing
            # Store scale factor for later use
            adjusted_df.at[idx, 'scale_factor'] = scale_factor
        
        print(f"Adjusted coordinates for {len(adjusted_df)} tomograms")
        return adjusted_df

    def set_epoch(self, epoch):
        self.epoch_seed = self.seed + epoch
        self.rng = np.random.RandomState(self.epoch_seed)
    
    def __len__(self): 
        return self.dataset_size

    def _resample_tile(self, tile, scale_factor):
        if abs(scale_factor - 1.0) < 0.01:
            return tile 
        resampled = F.interpolate(
            tile.unsqueeze(0),  # Add batch dimension: (1, C, Z, Y, X)
            scale_factor=[scale_factor] * 3,
            mode='trilinear',
            align_corners=False
        ).squeeze(0)  # Remove batch dimension: (C, Z, Y, X)
        
        return resampled
    
    def _get_positive_tomogram(self, tomo_row, tomo_id, item_rng):
        if tomo_row['num_coords'] == 0:
            return None

        scale_factor = tomo_row['scale_factor']


        if int(tomo_row['num_coords']) == 1: 
            coord_idx = 0
        else:
            coord_idx = item_rng.randint(0, tomo_row['num_coords'] - 1)

        adjusted_label_z, adjusted_label_y, adjusted_label_x = map(int, tomo_row['coordinates'][coord_idx])

        z, y, x = self.tile_size
        bound = 5
        z_range = tomo_row["Array shape (axis 0)"]
        y_range = tomo_row["Array shape (axis 1)"]
        x_range = tomo_row["Array shape (axis 2)"]

        def get_shift(label, size, array_range, bound):
            half = size // 2
            dist_to_low = label
            dist_to_high = array_range - label
            max_neg_shift = -min(half - bound, dist_to_low)
            max_pos_shift = min(half - bound, dist_to_high)
            return item_rng.randint(max_neg_shift, max_pos_shift)

        shift_z = get_shift(adjusted_label_z, z, z_range, bound)
        shift_y = get_shift(adjusted_label_y, y, y_range, bound)
        shift_x = get_shift(adjusted_label_x, x, x_range, bound)

        center_z = adjusted_label_z + shift_z
        center_y = adjusted_label_y + shift_y
        center_x = adjusted_label_x + shift_x

        z1 = max(0, center_z - z // 2)
        y1 = max(0, center_y - y // 2)
        x1 = max(0, center_x - x // 2)
        z2 = min(z_range, z1 + z)
        y2 = min(y_range, y1 + y)
        x2 = min(x_range, x1 + x)

        if z2 - z1 < z:
            z1 = max(0, z2 - z)
        if y2 - y1 < y:
            y1 = max(0, y2 - y)
        if x2 - x1 < x:
            x1 = max(0, x2 - x)

        # Convert back to original coordinate space (before resampling)
        original_z1 = math.floor(z1 / scale_factor)
        original_z2 = math.ceil(z2 / scale_factor)
        original_y1 = math.floor(y1 / scale_factor)
        original_y2 = math.ceil(y2 / scale_factor)
        original_x1 = math.floor(x1 / scale_factor)
        original_x2 = math.ceil(x2 / scale_factor)

        # Extract tile and resample
        tile = self.tiler.extract_tile(
            tomo_id=tomo_id,
            z1=original_z1, z2=original_z2,
            y1=original_y1, y2=original_y2,
            x1=original_x1, x2=original_x2
        )
        tile = self._resample_tile(tile, scale_factor)
        tile = tile[:, :z, :y, :x]

        # Pad if undersized
        pad_z = z - tile.shape[1]
        pad_y = y - tile.shape[2]
        pad_x = x - tile.shape[3]
        if pad_z > 0 or pad_y > 0 or pad_x > 0:
            tile = F.pad(tile, (0, pad_x, 0, pad_y, 0, pad_z), mode='constant', value=0)

        # Calculate local coordinates in the resampled tile space
        local_coords = []
        for (coord_z, coord_y, coord_x) in tomo_row['coordinates']:
            if (z1 <= coord_z < z2 and y1 <= coord_y < y2 and x1 <= coord_x < x2):
                local_coord = (coord_z - z1, coord_y - y1, coord_x - x1)
                local_coords.append(local_coord)


        # Center of tile in adjusted space
        tile_center_z = (z1 + z2) // 2
        tile_center_y = (y1 + y2) // 2
        tile_center_x = (x1 + x2) // 2

        

        return tile, 1.0, local_coords, (tile_center_z, tile_center_y, tile_center_x)


    def _get_hard_negative(self, tomo_row, tomo_id, item_rng):
        """Generate hard negative tiles with voxel spacing normalization"""
        scale_factor = tomo_row['scale_factor']
        max_attempts = 50
        
        for attempt in range(max_attempts):
            # Use adjusted shapes for random coordinate generation
            adjusted_tomo_row = tomo_row.copy()
            
            tile_centre_z, tile_centre_y, tile_centre_x, z1, y1, x1, z2, y2, x2 = get_rand_coords(
                tomo_row=adjusted_tomo_row, 
                tile_size=self.tile_size,
                item_rng=item_rng
            )
            
            # Check if this tile contains any adjusted coordinates
            has_coords = False
            for (adjusted_coord_z, adjusted_coord_y, adjusted_coord_x) in tomo_row['coordinates']:
                if (z1 <= adjusted_coord_z < z2 and 
                    y1 <= adjusted_coord_y < y2 and 
                    x1 <= adjusted_coord_x < x2):
                    has_coords = True
                    break
            
            if not has_coords:
                original_z1 = math.floor(z1 / scale_factor)
                original_z2 = math.ceil(z2 / scale_factor)
                original_y1 = math.floor(y1 / scale_factor)
                original_y2 = math.ceil(y2 / scale_factor)
                original_x1 = math.floor(x1 / scale_factor)
                original_x2 = math.ceil(x2 / scale_factor)
                
                tile = self.tiler.extract_tile(
                    tomo_id=tomo_id,
                    z1=original_z1, z2=original_z2,
                    y1=original_y1, y2=original_y2,
                    x1=original_x1, x2=original_x2
                )
                
                tile = self._resample_tile(tile, scale_factor)
                tile = tile[:, :self.tile_size[0], :self.tile_size[1], :self.tile_size[2]]
                tile = tile[:, :self.tile_size[0], :self.tile_size[1], :self.tile_size[2]]

                pad_z = self.tile_size[0] - tile.shape[1]
                pad_y = self.tile_size[1] - tile.shape[2]
                pad_x = self.tile_size[2] - tile.shape[3]
                if pad_z > 0 or pad_y > 0 or pad_x > 0:
                    tile = F.pad(tile, (0, pad_x, 0, pad_y, 0, pad_z), mode='constant', value=0)
                pad_z = self.tile_size[0] - tile.shape[1]
                pad_y = self.tile_size[1] - tile.shape[2]
                pad_x = self.tile_size[2] - tile.shape[3]
                if pad_z > 0 or pad_y > 0 or pad_x > 0:
                    tile = F.pad(tile, (0, pad_x, 0, pad_y, 0, pad_z), mode='constant', value=0)
                return tile, 0.0, [(-1, -1, -1)], (tile_centre_z, tile_centre_y, tile_centre_x)
        
        return self._get_random_tomogram(tomo_row, tomo_id, item_rng)

    def _get_random_tomogram(self, tomo_row, tomo_id, item_rng):
        scale_factor = tomo_row['scale_factor']
        
        tile_centre_z, tile_centre_y, tile_centre_x, z1, y1, x1, z2, y2, x2 = get_rand_coords(
            tomo_row=tomo_row, 
            tile_size=self.tile_size,
            item_rng=item_rng
        )
        
        original_z1 = math.floor(z1 / scale_factor)
        original_z2 = math.ceil(z2 / scale_factor)
        original_y1 = math.floor(y1 / scale_factor)
        original_y2 = math.ceil(y2 / scale_factor)
        original_x1 = math.floor(x1 / scale_factor)
        original_x2 = math.ceil(x2 / scale_factor)
        tile = self.tiler.extract_tile(
            tomo_id=tomo_id,
            z1=original_z1, z2=original_z2,
            y1=original_y1, y2=original_y2,
            x1=original_x1, x2=original_x2
        )
        
        tile = self._resample_tile(tile, scale_factor)
        tile = tile[:, :self.tile_size[0], :self.tile_size[1], :self.tile_size[2]]

        pad_z = self.tile_size[0] - tile.shape[1]
        pad_y = self.tile_size[1] - tile.shape[2]
        pad_x = self.tile_size[2] - tile.shape[3]
        if pad_z > 0 or pad_y > 0 or pad_x > 0:
            tile = F.pad(tile, (0, pad_x, 0, pad_y, 0, pad_z), mode='constant', value=0)
        has_motor = 0.0
        local_coords = []
        
        for (adjusted_coord_z, adjusted_coord_y, adjusted_coord_x) in tomo_row['coordinates']:
            if (z1 <= adjusted_coord_z < z2 and 
                y1 <= adjusted_coord_y < y2 and 
                x1 <= adjusted_coord_x < x2):
                
                has_motor = 1.0
                local_coord = (
                    adjusted_coord_z - z1,
                    adjusted_coord_y - y1,
                    adjusted_coord_x - x1
                )
                local_coords.append(local_coord)
        
        if has_motor == 0.0:
            local_coords = [(-1, -1, -1)]
        
        return tile, has_motor, local_coords, (tile_centre_z, tile_centre_y, tile_centre_x)

    def __getitem__(self, idx):
        item_seed = (self.epoch_seed + idx) % (2**32 - 1)
        item_rng = np.random.RandomState(item_seed)
                
        if item_rng.random() < self.positive_ratio:
            tomo_row = self.df.loc[self.df['num_coords'] > 0].sample(n=1).iloc[0]
            tomo_id = tomo_row['tomo_id']
            tile, has_motor, local_coords, tile_origin = self._get_positive_tomogram(tomo_row, tomo_id, item_rng)

        else:
            tomo_row = self.df.loc[self.df['num_coords'] == 0].sample(n=1).iloc[0]
            tomo_id = tomo_row['tomo_id']
            tile, has_motor, local_coords, tile_origin = self._get_hard_negative(tomo_row, tomo_id, item_rng)            
            #print(f"2===>{local_coords=}")


        if int(has_motor) != 1: 
            local_coords = [(-1, -1, -1)]
            
        if self.transform is not None:
            tile = self.transform(tile)

        if not isinstance(tile, torch.Tensor):
            tile = torch.tensor(tile, dtype=torch.float32)
        local_coords = torch.tensor(local_coords, dtype=torch.float32)
        pad = torch.ones((20-local_coords.shape[0], 3)) * -1 
        local_coords = torch.concatenate([local_coords, pad], dim=0)


        return_dict = {
            "image_tile": tile,
            "has_motor": torch.tensor(has_motor, dtype=torch.float32),
            "local_coords": local_coords,
            "tile_origin": torch.tensor(tile_origin, dtype=torch.float32),            
            "tomo_id": tomo_id,
            "voxel_spacing": torch.tensor([self.target_voxel_spacing], dtype=torch.float32),  # Now consistent
            "original_shape": [tomo_row[f'Array shape (axis {i})'] for i in range(3)],  # Adjusted shapes
            "scale_factor": torch.tensor([tomo_row['scale_factor']], dtype=torch.float32),  # For debugging
        }
        # print('local_coords:',return_dict['local_coords'].shape,
        #       'image_tile',return_dict['image_tile'].shape,
        #       'target_voxel_spacing:', return_dict['voxel_spacing'],
        #       'tile_origin:', return_dict['tile_origin'],
        #       'scale_factor:', return_dict['scale_factor']
        #       )
        return return_dict
    

def preprocess_dataframe(df):
    """Group coordinates by tomo_id"""
    grouped = df.groupby('tomo_id')
    processed_df = []
    
    for tomo_id, group in grouped:
        coords = []
        for _, row in group.iterrows():
            #print(row)
            coords.append([row["Motor axis 0"], row["Motor axis 1"], row["Motor axis 2"]])
        
        if coords[0][0] == -1:
            num_coords = 0.0
        else:
            num_coords = len(coords)
        processed_df.append({
            'tomo_id': tomo_id,
            'coordinates': coords,
            'num_coords': num_coords,
            'Voxel spacing': group['Voxel spacing'].values[0],
            'Array shape (axis 0)': group['Array shape (axis 0)'].values[0],
            'Array shape (axis 1)': group['Array shape (axis 1)'].values[0],
            'Array shape (axis 2)': group['Array shape (axis 2)'].values[0],
        })
    
    return pd.DataFrame(processed_df)

if __name__ == '__main__':
    train_df = pd.read_csv('/home/porpita/BYU/DS/train_labels.csv')
    train_df = preprocess_dataframe(train_df)
    img_files_dir = "/home/porpita/BYU/DS/train"
    args = tile_size = (96,96,96)
    positive_ratio = 0.2
    transforms = None
    dataset_size = 500
    seed = 42


    args = {
        "tile_size": (48, 48, 48),
        "positive_ratio": 0.2,
        "transform": None,
        "dataset_size": 1000,
        "seed": 42
    }
    ds = CustomDataset(train_df, img_files_dir,**args)
    for i in range(1000): 
        print('-'*50)
        print(i)
        print(ds[i]['image_tile'].shape)
        print(ds[i]['local_coords'])
        print(ds[i]['tile_origin'])
        print(ds[i]['tomo_id'])






# class VolumeReconstructor:
#     """Helper class to reconstruct full volume from overlapping tiles"""
    
#     def __init__(self, volume_shape: Tuple[int, int, int], 
#                  tile_size: Tuple[int, int, int],
#                  overlap: float = 0.25,
#                  blend_mode: str = "gaussian"):
#         self.volume_shape = volume_shape
#         self.tile_size = tile_size
#         self.overlap = overlap
#         self.blend_mode = blend_mode
        
#         # Initialize output volume and weight map
#         self.output_volume = None
#         self.weight_map = None
        
#     def add_tile(self, tile_pred: torch.Tensor, position: Tuple[int, int, int, int, int, int]):
#         """Add a predicted tile to the reconstruction"""
#         z1, y1, x1, z2, y2, x2 = position
        
#         # Initialize volumes on first tile
#         if self.output_volume is None:
#             num_channels = tile_pred.shape[1] if len(tile_pred.shape) == 4 else tile_pred.shape[0]
#             self.output_volume = torch.zeros((num_channels,) + self.volume_shape, 
#                                            dtype=tile_pred.dtype, device=tile_pred.device)
#             self.weight_map = torch.zeros(self.volume_shape, 
#                                         dtype=torch.float32, device=tile_pred.device)
        
#         # Calculate weights for blending
#         tile_weights = self._get_tile_weights(z2-z1, y2-y1, x2-x1)
#         tile_weights = tile_weights.to(tile_pred.device)
        
#         # Add tile to output (handle both 3D and 4D tensors)
#         if len(tile_pred.shape) == 4:  # (C, Z, Y, X)
#             self.output_volume[:, z1:z2, y1:y2, x1:x2] += tile_pred * tile_weights
#         else:  # (Z, Y, X)
#             self.output_volume[z1:z2, y1:y2, x1:x2] += tile_pred * tile_weights
            
#         # Update weight map
#         self.weight_map[z1:z2, y1:y2, x1:x2] += tile_weights
    
#     def _get_tile_weights(self, tile_z: int, tile_y: int, tile_x: int) -> torch.Tensor:
#         """Generate weights for tile blending"""
#         if self.blend_mode == "gaussian":
#             # Create 3D Gaussian weights
#             z_weights = torch.exp(-0.5 * ((torch.arange(tile_z) - tile_z//2) / (tile_z//6))**2)
#             y_weights = torch.exp(-0.5 * ((torch.arange(tile_y) - tile_y//2) / (tile_y//6))**2)
#             x_weights = torch.exp(-0.5 * ((torch.arange(tile_x) - tile_x//2) / (tile_x//6))**2)
            
#             # Create 3D weight map
#             weights = torch.outer(torch.outer(z_weights, y_weights).flatten(), x_weights)
#             weights = weights.reshape(tile_z, tile_y, tile_x)
            
#         else:  # constant
#             weights = torch.ones((tile_z, tile_y, tile_x))
            
#         return weights
    
#     def get_final_volume(self) -> torch.Tensor:
#         """Get the final reconstructed volume"""
#         if self.output_volume is None:
#             raise ValueError("No tiles have been added yet")
        
#         # Normalize by weights to handle overlaps
#         # Add small epsilon to avoid division by zero
#         self.weight_map = torch.clamp(self.weight_map, min=1e-8)
        
#         if len(self.output_volume.shape) == 4:  # (C, Z, Y, X)
#             normalized_volume = self.output_volume / self.weight_map.unsqueeze(0)
#         else:  # (Z, Y, X)
#             normalized_volume = self.output_volume / self.weight_map
            
#         return normalized_volume
# class CustomDataset(Dataset): 
#     def __init__(self,
#                 train_df: pd.DataFrame,
#                 img_files_dir: str,
#                 tile_size: Union[int, Tuple[int, int, int]] = (32, 128, 128),
#                 positive_ratio: float = 0.2,
#                 transform: Optional[Callable] = None,
#                 dataset_size: int = 1000,
#                 seed: int = 42,
#                 target_voxel_spacing: Optional[float] = None):
        
#         super().__init__()
        
#         if 'Voxel spacing' not in train_df.columns:
#             raise ValueError("The DataFrame must contain a 'Voxel spacing' column")

#         self.seed = seed
#         self.rng = np.random.RandomState(seed)
#         self.epoch_seed = seed

#         # Determine target voxel spacing
#         if target_voxel_spacing is None:
#             voxel_spacings = train_df['Voxel spacing'].values
#             self.target_voxel_spacing = np.median(voxel_spacings)
#             print(f"Auto-selected target voxel spacing: {self.target_voxel_spacing:.2f} Angstroms")
#         else:
#             self.target_voxel_spacing = target_voxel_spacing
#             print(f"Using target voxel spacing: {self.target_voxel_spacing:.2f} Angstroms")
        
#         # Store original dataframe for reference
#         self.unprocessed_df = train_df.copy()
        
#         # Pre-process the dataframe to adjust coordinates and shapes
#         self.df = self._preprocess_dataframe_with_voxel_spacing(preprocess_dataframe(train_df))
        
#         self.img_files_dir = img_files_dir
#         self.tile_size = tile_size
#         self.transform = transform
#         self.dataset_size = dataset_size
#         self.unique_tomo_ids = self.df['tomo_id'].unique()
#         self.positive_ratio = positive_ratio 
#         self.tiler = TomogramTiler(base_path=img_files_dir)

#     def _preprocess_dataframe_with_voxel_spacing(self, df):
#         """
#         Pre-calculate what coordinates and shapes would be after voxel spacing normalization.
#         This avoids complex coordinate transformations during training.
#         """
#         adjusted_df = df.copy()
        
#         print("Pre-calculating adjusted coordinates for voxel spacing normalization...")
        
#         for idx, row in adjusted_df.iterrows():
#             original_spacing = row['Voxel spacing']
#             scale_factor = original_spacing / self.target_voxel_spacing
            
#             # Adjust coordinates
#             adjusted_coords = []
#             for coord in row['coordinates']:
#                 adjusted_coord = [
#                     math.floor(coord[0] * scale_factor),  # Z
#                     math.floor(coord[1] * scale_factor),  # Y  
#                     math.floor(coord[2] * scale_factor)   # X
#                 ]
#                 adjusted_coords.append(adjusted_coord)
            
#             adjusted_df.at[idx, 'coordinates'] = adjusted_coords
#             adjusted_df.at[idx, 'num_coords'] = len(adjusted_coords)
            
#             # Adjust array shapes (tomogram dimensions after resampling)
#             for axis in range(3):
#                 original_shape = row[f'Array shape (axis {axis})']
#                 new_shape = math.ceil(original_shape * scale_factor)
#                 adjusted_df.at[idx, f'Array shape (axis {axis})'] = new_shape
            
#             # Update voxel spacing to target
#             adjusted_df.at[idx, 'Voxel spacing'] = self.target_voxel_spacing
            
#             # Store scale factor for later use
#             adjusted_df.at[idx, 'scale_factor'] = scale_factor
        
#         print(f"Adjusted coordinates for {len(adjusted_df)} tomograms")
#         return adjusted_df

#     def set_epoch(self, epoch):
#         """Call this at the start of each epoch for consistent shuffling"""
#         self.epoch_seed = self.seed + epoch
#         self.rng = np.random.RandomState(self.epoch_seed)
    
#     def __len__(self): 
#         return self.dataset_size

#     def _resample_tile(self, tile, scale_factor):
#         """Resample individual tile to target voxel spacing using trilinear interpolation"""
#         if abs(scale_factor - 1.0) < 0.01:
#             return tile  # No resampling needed
        
#         # Use trilinear interpolation for smooth resampling
#         resampled = F.interpolate(
#             tile.unsqueeze(0),  # Add batch dimension: (1, C, Z, Y, X)
#             scale_factor=[scale_factor] * 3,
#             mode='trilinear',
#             align_corners=False
#         ).squeeze(0)  # Remove batch dimension: (C, Z, Y, X)
        
#         return resampled

#     def _get_positive_tomogram(self, tomo_row, tomo_id, item_rng):
#         """Extract positive tile with voxel spacing normalization"""
#         if tomo_row['num_coords'] == 0:
#             return None
        
#         # Get scale factor for this tomogram
#         scale_factor = tomo_row['scale_factor']
        
#         # Use adjusted coordinates for tile planning (in target voxel spacing)
#         coord_idx = item_rng.randint(0, tomo_row['num_coords'])
#         adjusted_label_z, adjusted_label_y, adjusted_label_x = map(int, tomo_row['coordinates'][coord_idx])
        
#         # Calculate tile bounds using adjusted coordinates and shapes
#         z, y, x = self.tile_size
#         max_shift = min(z//4, y//4, x//4, 5)
#         shift_z = item_rng.randint(-max_shift, max_shift + 1)
#         shift_y = item_rng.randint(-max_shift, max_shift + 1)
#         shift_x = item_rng.randint(-max_shift, max_shift + 1)
        
#         # Use adjusted array shapes (after resampling)
#         z_range = tomo_row["Array shape (axis 0)"]
#         y_range = tomo_row["Array shape (axis 1)"]
#         x_range = tomo_row["Array shape (axis 2)"]
        
#         # Calculate tile bounds in adjusted coordinate space
#         adjusted_z1 = math.floor(max(0, adjusted_label_z - z//2 + shift_z))
#         adjusted_y1 = math.floor(max(0, adjusted_label_y - y//2 + shift_y))
#         adjusted_x1 = math.floor(max(0, adjusted_label_x - x//2 + shift_x))
#         adjusted_z2 = math.ceil(min(z_range, adjusted_z1 + z))
#         adjusted_y2 = math.ceil(min(y_range, adjusted_y1 + y))
#         adjusted_x2 = math.ceil(min(x_range, adjusted_x1 + x))

#         # Adjust bounds if tile extends beyond limits
#         if adjusted_z2 - adjusted_z1 < z:
#             adjusted_z1 = max(0, adjusted_z2 - z)
#         if adjusted_y2 - adjusted_y1 < y:
#             adjusted_y1 = max(0, adjusted_y2 - y)
#         if adjusted_x2 - adjusted_x1 < x:
#             adjusted_x1 = max(0, adjusted_x2 - x)
        
#         # Convert back to original coordinate space for tile extraction
#         original_z1 = math.floor(adjusted_z1 / scale_factor)
#         original_z2 = math.ceil(adjusted_z2 / scale_factor)
#         original_y1 = math.floor(adjusted_y1 / scale_factor)
#         original_y2 = math.ceil(adjusted_y2 / scale_factor)
#         original_x1 = math.floor(adjusted_x1 / scale_factor)
#         original_x2 = math.ceil(adjusted_x2 / scale_factor)

        
#         # Extract tile from original images using original coordinates
#         tile = self.tiler.extract_tile(
#             tomo_id=tomo_id,
#             z1=original_z1, z2=original_z2,
#             y1=original_y1, y2=original_y2,
#             x1=original_x1, x2=original_x2
#         )
        
#         # Resample the extracted tile to target voxel spacing
#         tile = self._resample_tile(tile, scale_factor)

#         # Calculate local coordinates in the resampled tile space
#         local_coords = []
#         for (adjusted_coord_z, adjusted_coord_y, adjusted_coord_x) in tomo_row['coordinates']:
#             if (adjusted_z1 <= adjusted_coord_z < adjusted_z2 and 
#                 adjusted_y1 <= adjusted_coord_y < adjusted_y2 and 
#                 adjusted_x1 <= adjusted_coord_x < adjusted_x2):
                
#                 local_coord = (
#                     adjusted_coord_z - adjusted_z1,
#                     adjusted_coord_y - adjusted_y1,
#                     adjusted_coord_x - adjusted_x1
#                 )
#                 local_coords.append(local_coord)
        
#         # Calculate tile center in adjusted space
#         tile_center_z = (adjusted_z1 + adjusted_z2) // 2
#         tile_center_y = (adjusted_y1 + adjusted_y2) // 2
#         tile_center_x = (adjusted_x1 + adjusted_x2) // 2
        
#         return tile, 1.0, local_coords, (tile_center_z, tile_center_y, tile_center_x)

#     def _get_hard_negative(self, tomo_row, tomo_id, item_rng):
#         """Generate hard negative tiles with voxel spacing normalization"""
#         scale_factor = tomo_row['scale_factor']
#         max_attempts = 50
        
#         for attempt in range(max_attempts):
#             # Use adjusted shapes for random coordinate generation
#             adjusted_tomo_row = tomo_row.copy()
            
#             tile_centre_z, tile_centre_y, tile_centre_x, z1, y1, x1, z2, y2, x2 = get_rand_coords(
#                 tomo_row=adjusted_tomo_row, 
#                 tile_size=self.tile_size,
#                 item_rng=item_rng
#             )
            
#             # Check if this tile contains any adjusted coordinates
#             has_coords = False
#             for (adjusted_coord_z, adjusted_coord_y, adjusted_coord_x) in tomo_row['coordinates']:
#                 if (z1 <= adjusted_coord_z < z2 and 
#                     y1 <= adjusted_coord_y < y2 and 
#                     x1 <= adjusted_coord_x < x2):
#                     has_coords = True
#                     break
            
#             if not has_coords:
#                 # Convert to original coordinate space for extraction
#                 original_z1 = math.floor(z1 / scale_factor)
#                 original_z2 = math.ceil(z2 / scale_factor)
#                 original_y1 = math.floor(y1 / scale_factor)
#                 original_y2 = math.ceil(y2 / scale_factor)
#                 original_x1 = math.floor(x1 / scale_factor)
#                 original_x2 = math.ceil(x2 / scale_factor)
                
#                 # Extract and resample tile
#                 tile = self.tiler.extract_tile(
#                     tomo_id=tomo_id,
#                     z1=original_z1, z2=original_z2,
#                     y1=original_y1, y2=original_y2,
#                     x1=original_x1, x2=original_x2
#                 )
                
#                 tile = self._resample_tile(tile, scale_factor)
                
#                 return tile, 0.0, [(-1, -1, -1)], (tile_centre_z, tile_centre_y, tile_centre_x)
        
#         # Fallback to random tile if can't find true negative
#         return self._get_random_tomogram(tomo_row, tomo_id, item_rng)

#     def _get_random_tomogram(self, tomo_row, tomo_id, item_rng):
#         """Extract random tile with voxel spacing normalization"""
#         scale_factor = tomo_row['scale_factor']
        
#         # Use adjusted shapes for random coordinate generation
#         tile_centre_z, tile_centre_y, tile_centre_x, z1, y1, x1, z2, y2, x2 = get_rand_coords(
#             tomo_row=tomo_row, 
#             tile_size=self.tile_size,
#             item_rng=item_rng
#         )
        
#         # Convert to original coordinate space for extraction
#         original_z1 = math.floor(z1 / scale_factor)
#         original_z2 = math.ceil(z2 / scale_factor)
#         original_y1 = math.floor(y1 / scale_factor)
#         original_y2 = math.ceil(y2 / scale_factor)
#         original_x1 = math.floor(x1 / scale_factor)
#         original_x2 = math.ceil(x2 / scale_factor)
#         # Extract and resample tile
#         tile = self.tiler.extract_tile(
#             tomo_id=tomo_id,
#             z1=original_z1, z2=original_z2,
#             y1=original_y1, y2=original_y2,
#             x1=original_x1, x2=original_x2
#         )
        
#         tile = self._resample_tile(tile, scale_factor)
        
#         # Check for coordinates in adjusted space
#         has_motor = 0.0
#         local_coords = []
        
#         for (adjusted_coord_z, adjusted_coord_y, adjusted_coord_x) in tomo_row['coordinates']:
#             if (z1 <= adjusted_coord_z < z2 and 
#                 y1 <= adjusted_coord_y < y2 and 
#                 x1 <= adjusted_coord_x < x2):
                
#                 has_motor = 1.0
#                 local_coord = (
#                     adjusted_coord_z - z1,
#                     adjusted_coord_y - y1,
#                     adjusted_coord_x - x1
#                 )
#                 local_coords.append(local_coord)
        
#         if has_motor == 0.0:
#             local_coords = [(-1, -1, -1)]
        
#         return tile, has_motor, local_coords, (tile_centre_z, tile_centre_y, tile_centre_x)

#     def __getitem__(self, idx):
#         """Modified getitem to use pre-calculated adjusted coordinates"""
#         item_seed = (self.epoch_seed + idx) % (2**32 - 1)
#         item_rng = np.random.RandomState(item_seed)
        
#         tomo_id = item_rng.choice(self.unique_tomo_ids)
#         tomo_row = self.df[self.df['tomo_id'] == tomo_id].iloc[0]
        
#         should_be_positive = item_rng.random() < self.positive_ratio
        
#         if should_be_positive and tomo_row['num_coords'] > 0:
#             result = self._get_positive_tomogram(tomo_row, tomo_id, item_rng)
#             if result is not None:
#                 tile, has_motor, local_coords, tile_origin = result
#             else:
#                 tile, has_motor, local_coords, tile_origin = self._get_hard_negative(tomo_row, tomo_id, item_rng)
#         else:
#             tile, has_motor, local_coords, tile_origin = self._get_hard_negative(tomo_row, tomo_id, item_rng)

#         if int(has_motor) != 1: 
#             local_coords = [(-1, -1, -1)]
            
#         # Apply transforms
#         if self.transform is not None:
#             tile = self.transform(tile)

#         if not isinstance(tile, torch.Tensor):
#             tile = torch.tensor(tile, dtype=torch.float32)
        
#         return {
#             "image_tile": tile,
#             "has_motor": torch.tensor(has_motor, dtype=torch.float32),
#             "local_coords": torch.tensor(local_coords, dtype=torch.float32),
#             "tile_origin": torch.tensor(tile_origin, dtype=torch.float32),
#             "tomo_id": tomo_id,
#             "voxel_spacing": torch.tensor(self.target_voxel_spacing, dtype=torch.float32),  # Now consistent
#             "original_shape": [tomo_row[f'Array shape (axis {i})'] for i in range(3)],  # Adjusted shapes
#             "scale_factor": torch.tensor(tomo_row['scale_factor'], dtype=torch.float32),  # For debugging
#         }

# class CustomInferenceDataset(Dataset):
#     def __init__(self, 
#                  tomo_id: str,
#                  img_files_dir: str,
#                  tile_size: Tuple[int, int, int] = (96, 96, 96),
#                  overlap: float = 0.25,
#                  transform: Optional[Callable] = None,
#                  target_voxel_spacing: Optional[float] = None,
#                  original_voxel_spacing: Optional[float] = None):
#         """
#         Sliding window inference dataset for 3D volumes with voxel spacing normalization
        
#         Args:
#             tomo_id: ID of the tomogram to process
#             img_files_dir: Directory containing tomogram slices
#             tile_size: Size of each tile (Z, Y, X)
#             overlap: Overlap ratio between adjacent tiles (0.0 to 0.5)
#             transform: Optional transforms to apply
#             target_voxel_spacing: Target voxel spacing for normalization
#             original_voxel_spacing: Original voxel spacing of this tomogram
#         """
#         self.tomo_id = tomo_id
#         self.img_files_dir = img_files_dir
#         self.tile_size = tile_size
#         self.overlap = overlap
#         self.transform = transform
#         self.target_voxel_spacing = target_voxel_spacing
#         self.original_voxel_spacing = original_voxel_spacing
        
#         # Calculate scale factor for voxel spacing normalization
#         if self.original_voxel_spacing is not None and self.target_voxel_spacing is not None:
#             self.scale_factor = self.original_voxel_spacing / self.target_voxel_spacing
#             print(f"Inference voxel spacing: {self.original_voxel_spacing:.2f} -> {self.target_voxel_spacing:.2f} Angstroms")
#             print(f"Scale factor: {self.scale_factor:.3f}")
#         else:
#             self.scale_factor = 1.0
#             print("No voxel spacing normalization applied during inference")
        
#         # Initialize tiler
#         self.tiler = TomogramTiler(base_path=img_files_dir)
        
#         # Get tomogram dimensions (original space)
#         self.original_volume_shape = self._get_original_volume_shape()
        
#         # Calculate adjusted volume shape after resampling
#         self.adjusted_volume_shape = self._get_adjusted_volume_shape()
        
#         # Calculate sliding window positions in adjusted space
#         self.tile_positions = self._calculate_tile_positions()
        
#     def _get_original_volume_shape(self) -> Tuple[int, int, int]:
#         """Get the shape of the original tomogram volume"""
#         slice_mapping = self.tiler._get_slice_mapping(self.tomo_id)
        
#         # Get Z dimension from number of slices
#         z_dim = len(slice_mapping)
        
#         # Get Y, X dimensions from first slice
#         if slice_mapping:
#             first_slice_idx = min(slice_mapping.keys())
#             first_slice_file = slice_mapping[first_slice_idx]
#             tomo_path = os.path.join(self.img_files_dir, self.tomo_id)
#             full_path = os.path.join(tomo_path, first_slice_file)
            
#             img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
#             if img is None:
#                 raise ValueError(f"Could not load image: {full_path}")
            
#             y_dim, x_dim = img.shape
#         else:
#             raise ValueError(f"No slices found for tomogram {self.tomo_id}")
            
#         return (z_dim, y_dim, x_dim)
    
#     def _get_adjusted_volume_shape(self) -> Tuple[int, int, int]:
#         """Get the shape after voxel spacing normalization"""
#         if abs(self.scale_factor - 1.0) < 1e-6:
#             return self.original_volume_shape
        
#         # Calculate new dimensions after resampling
#         adjusted_shape = tuple(int(dim * self.scale_factor) for dim in self.original_volume_shape)
#         return adjusted_shape
    
#     def _calculate_tile_positions(self) -> List[Tuple[int, int, int, int, int, int]]:
#         """Calculate all tile positions with overlap in adjusted coordinate space"""
#         positions = []
#         z_max, y_max, x_max = self.adjusted_volume_shape  # Use adjusted shape
#         tile_z, tile_y, tile_x = self.tile_size
        
#         # Calculate step sizes (accounting for overlap)
#         step_z = max(1, int(tile_z * (1 - self.overlap)))
#         step_y = max(1, int(tile_y * (1 - self.overlap)))
#         step_x = max(1, int(tile_x * (1 - self.overlap)))
        
#         # Generate all tile positions in adjusted space
#         for z_start in range(0, z_max, step_z):
#             for y_start in range(0, y_max, step_y):
#                 for x_start in range(0, x_max, step_x):
#                     # Calculate end positions
#                     z_end = min(z_start + tile_z, z_max)
#                     y_end = min(y_start + tile_y, y_max)
#                     x_end = min(x_start + tile_x, x_max)
                    
#                     # Adjust start positions if tile extends beyond bounds
#                     z_start_adj = max(0, z_end - tile_z)
#                     y_start_adj = max(0, y_end - tile_y)
#                     x_start_adj = max(0, x_end - tile_x)
                    
#                     positions.append((z_start_adj, y_start_adj, x_start_adj, 
#                                     z_end, y_end, x_end))
        
#         return positions
    
#     def _resample_tile(self, tile, scale_factor):
#         """Resample individual tile to target voxel spacing using trilinear interpolation"""
#         if abs(scale_factor - 1.0) < 1e-6:
#             return tile  # No resampling needed
        
#         # Use trilinear interpolation for smooth resampling
#         resampled = F.interpolate(
#             tile.unsqueeze(0),  # Add batch dimension: (1, C, Z, Y, X)
#             scale_factor=[scale_factor] * 3,
#             mode='trilinear',
#             align_corners=False
#         ).squeeze(0)  # Remove batch dimension: (C, Z, Y, X)
        
#         return resampled
    
#     def __len__(self) -> int:
#         return len(self.tile_positions)
    
#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         """Get a tile for inference with voxel spacing normalization"""
#         # Get tile position in adjusted space
#         adjusted_z1, adjusted_y1, adjusted_x1, adjusted_z2, adjusted_y2, adjusted_x2 = self.tile_positions[idx]
        
#         # Convert to original coordinate space for extraction
#         if abs(self.scale_factor - 1.0) > 1e-6:
#             original_z1 = math.floor(adjusted_z1 / self.scale_factor)
#             original_z2 = math.ceil(adjusted_z2 / self.scale_factor)
#             original_y1 = math.floor(adjusted_y1 / self.scale_factor)
#             original_y2 = math.ceil(adjusted_y2 / self.scale_factor)
#             original_x1 = math.floor(adjusted_x1 / self.scale_factor)
#             original_x2 = math.ceil(adjusted_x2 / self.scale_factor)
#         else:
#             # No resampling needed
#             original_z1, original_y1, original_x1 = adjusted_z1, adjusted_y1, adjusted_x1
#             original_z2, original_y2, original_x2 = adjusted_z2, adjusted_y2, adjusted_x2
        
#         # Extract tile from original images
#         tile = self.tiler.extract_tile(
#             tomo_id=self.tomo_id,
#             z1=original_z1, z2=original_z2,
#             y1=original_y1, y2=original_y2,
#             x1=original_x1, x2=original_x2
#         )
        
#         # Resample tile to target voxel spacing
#         if abs(self.scale_factor - 1.0) > 1e-6:
#             tile = self._resample_tile(tile, self.scale_factor)

#         # Apply transforms if specified
#         if self.transform:
#             tile = self.transform(tile)

#         tile = tile[:, :96, :96, :96] #trim if oversized

#         # Pad if undersized
#         pad_z = 96 - tile.shape[1]
#         pad_y = 96 - tile.shape[2]
#         pad_x = 96 - tile.shape[3]
#         if pad_z > 0 or pad_y > 0 or pad_x > 0:
#             tile = F.pad(tile, (0, pad_x, 0, pad_y, 0, pad_z), mode='constant', value=0)

#         if not isinstance(tile, torch.Tensor):
#             tile = torch.tensor(tile, dtype=torch.float32)
        
#         return {
#             "image_tile": tile,
#             "tile_position": torch.tensor([adjusted_z1, adjusted_y1, adjusted_x1, 
#                                          adjusted_z2, adjusted_y2, adjusted_x2], dtype=torch.long),
#             "original_position": torch.tensor([original_z1, original_y1, original_x1,
#                                              original_z2, original_y2, original_x2], dtype=torch.long),
#             "tile_index": torch.tensor(idx, dtype=torch.long),
#             "tomo_id": self.tomo_id,
#             "volume_shape": torch.tensor(self.adjusted_volume_shape, dtype=torch.long),
#             "original_volume_shape": torch.tensor(self.original_volume_shape, dtype=torch.long),
#             "scale_factor": torch.tensor(self.scale_factor, dtype=torch.float32),
#             "voxel_spacing": torch.tensor(self.target_voxel_spacing if self.target_voxel_spacing else self.original_voxel_spacing, dtype=torch.float32)
#         }
    
#     def get_reconstruction_info(self) -> Dict[str, any]:
#         """Get information needed for reconstructing the full volume"""
#         return {
#             "volume_shape": self.adjusted_volume_shape,  # Use adjusted shape for reconstruction
#             "original_volume_shape": self.original_volume_shape,
#             "tile_size": self.tile_size,
#             "overlap": self.overlap,
#             "num_tiles": len(self.tile_positions),
#             "tile_positions": self.tile_positions,
#             "scale_factor": self.scale_factor,
#             "target_voxel_spacing": self.target_voxel_spacing,
#             "original_voxel_spacing": self.original_voxel_spacing
#         }
