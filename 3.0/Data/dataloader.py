import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, Optional, Callable
import cv2
import math


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

    # Consistent coordinate ordering (Z, Y, X)
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
            for coord in row['coordinates']:
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

    def _trim_and_pad_tile(self, tile):
        """Trim tile to tile_size and pad if undersized"""
        tile = tile[:, :self.tile_size[0], :self.tile_size[1], :self.tile_size[2]]
        pad_z = self.tile_size[0] - tile.shape[1]
        pad_y = self.tile_size[1] - tile.shape[2]
        pad_x = self.tile_size[2] - tile.shape[3]
        if pad_z > 0 or pad_y > 0 or pad_x > 0:
            tile = F.pad(tile, (0, pad_x, 0, pad_y, 0, pad_z), mode='constant', value=0)
        return tile

    @staticmethod
    def _adjusted_to_original_bounds(z1, y1, x1, z2, y2, x2, scale_factor):
        """Convert adjusted-space tile bounds back to original-space bounds for extraction."""
        oz1 = math.floor(z1 / scale_factor)
        oz2 = oz1 + round((z2 - z1) / scale_factor)
        oy1 = math.floor(y1 / scale_factor)
        oy2 = oy1 + round((y2 - y1) / scale_factor)
        ox1 = math.floor(x1 / scale_factor)
        ox2 = ox1 + round((x2 - x1) / scale_factor)
        return oz1, oy1, ox1, oz2, oy2, ox2

    def _get_positive_tomogram(self, tomo_row, tomo_id, item_rng):
        if tomo_row['num_coords'] == 0:
            return None

        scale_factor = tomo_row['scale_factor']

        if int(tomo_row['num_coords']) == 1:
            coord_idx = 0
        else:
            coord_idx = item_rng.randint(0, tomo_row['num_coords'] - 1)

        adjusted_label_z, adjusted_label_y, adjusted_label_x = map(int, tomo_row['coordinates'][coord_idx])

        tz, ty, tx = self.tile_size
        bound = 5
        z_range = tomo_row["Array shape (axis 0)"]
        y_range = tomo_row["Array shape (axis 1)"]
        x_range = tomo_row["Array shape (axis 2)"]

        def get_shift(label, size, array_range, bnd):
            half = size // 2
            dist_to_low = label
            dist_to_high = array_range - label
            max_neg_shift = -min(half - bnd, dist_to_low)
            max_pos_shift = min(half - bnd, dist_to_high)
            if max_neg_shift > max_pos_shift:
                return 0
            return item_rng.randint(max_neg_shift, max_pos_shift)

        # Retry loop: up to 3 attempts to place the motor inside the tile
        z1 = y1 = x1 = z2 = y2 = x2 = 0
        motor_inside = False
        for _attempt in range(3):
            shift_z = get_shift(adjusted_label_z, tz, z_range, bound)
            shift_y = get_shift(adjusted_label_y, ty, y_range, bound)
            shift_x = get_shift(adjusted_label_x, tx, x_range, bound)

            center_z = adjusted_label_z + shift_z
            center_y = adjusted_label_y + shift_y
            center_x = adjusted_label_x + shift_x

            z1 = max(0, center_z - tz // 2)
            y1 = max(0, center_y - ty // 2)
            x1 = max(0, center_x - tx // 2)
            z2 = min(z_range, z1 + tz)
            y2 = min(y_range, y1 + ty)
            x2 = min(x_range, x1 + tx)

            if z2 - z1 < tz:
                z1 = max(0, z2 - tz)
            if y2 - y1 < ty:
                y1 = max(0, y2 - ty)
            if x2 - x1 < tx:
                x1 = max(0, x2 - tx)

            if (z1 <= adjusted_label_z < z2 and
                y1 <= adjusted_label_y < y2 and
                x1 <= adjusted_label_x < x2):
                motor_inside = True
                break

        # Fallback: force-center tile on the motor
        if not motor_inside:
            z1 = max(0, min(adjusted_label_z - tz // 2, z_range - tz))
            y1 = max(0, min(adjusted_label_y - ty // 2, y_range - ty))
            x1 = max(0, min(adjusted_label_x - tx // 2, x_range - tx))
            z2 = z1 + tz
            y2 = y1 + ty
            x2 = x1 + tx

        original_z1, original_y1, original_x1, original_z2, original_y2, original_x2 = \
            self._adjusted_to_original_bounds(z1, y1, x1, z2, y2, x2, scale_factor)

        # Extract tile and resample
        tile = self.tiler.extract_tile(
            tomo_id=tomo_id,
            z1=original_z1, z2=original_z2,
            y1=original_y1, y2=original_y2,
            x1=original_x1, x2=original_x2
        )
        tile = self._resample_tile(tile, scale_factor)
        tile = self._trim_and_pad_tile(tile)

        # Calculate local coordinates in the resampled tile space
        local_coords = []
        for (coord_z, coord_y, coord_x) in tomo_row['coordinates']:
            if (z1 <= coord_z < z2 and y1 <= coord_y < y2 and x1 <= coord_x < x2):
                local_coord = (coord_z - z1, coord_y - y1, coord_x - x1)
                local_coords.append(local_coord)

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
                original_z1, original_y1, original_x1, original_z2, original_y2, original_x2 = \
                    self._adjusted_to_original_bounds(z1, y1, x1, z2, y2, x2, scale_factor)

                tile = self.tiler.extract_tile(
                    tomo_id=tomo_id,
                    z1=original_z1, z2=original_z2,
                    y1=original_y1, y2=original_y2,
                    x1=original_x1, x2=original_x2
                )

                tile = self._resample_tile(tile, scale_factor)
                tile = self._trim_and_pad_tile(tile)

                return tile, 0.0, [(-1, -1, -1)], (tile_centre_z, tile_centre_y, tile_centre_x)

        return self._get_random_tomogram(tomo_row, tomo_id, item_rng)

    def _get_random_tomogram(self, tomo_row, tomo_id, item_rng):
        scale_factor = tomo_row['scale_factor']

        tile_centre_z, tile_centre_y, tile_centre_x, z1, y1, x1, z2, y2, x2 = get_rand_coords(
            tomo_row=tomo_row,
            tile_size=self.tile_size,
            item_rng=item_rng
        )

        original_z1, original_y1, original_x1, original_z2, original_y2, original_x2 = \
            self._adjusted_to_original_bounds(z1, y1, x1, z2, y2, x2, scale_factor)
        tile = self.tiler.extract_tile(
            tomo_id=tomo_id,
            z1=original_z1, z2=original_z2,
            y1=original_y1, y2=original_y2,
            x1=original_x1, x2=original_x2
        )

        tile = self._resample_tile(tile, scale_factor)
        tile = self._trim_and_pad_tile(tile)

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

        if int(has_motor) != 1:
            local_coords = [(-1, -1, -1)]

        if self.transform is not None:
            tile = self.transform(tile)

        if not isinstance(tile, torch.Tensor):
            tile = torch.tensor(tile, dtype=torch.float32)
        local_coords = torch.tensor(local_coords, dtype=torch.float32)
        pad = torch.ones((20-local_coords.shape[0], 3)) * -1
        local_coords = torch.concatenate([local_coords, pad], dim=0)

        return {
            "image_tile": tile,
            "has_motor": torch.tensor(has_motor, dtype=torch.float32),
            "local_coords": local_coords,
            "tile_origin": torch.tensor(tile_origin, dtype=torch.float32),
            "tomo_id": tomo_id,
            "voxel_spacing": torch.tensor([self.target_voxel_spacing], dtype=torch.float32),
            "original_shape": [tomo_row[f'Array shape (axis {i})'] for i in range(3)],
            "scale_factor": torch.tensor([tomo_row['scale_factor']], dtype=torch.float32),
        }


def preprocess_dataframe(df):
    """Group coordinates by tomo_id"""
    grouped = df.groupby('tomo_id')
    processed_df = []

    for tomo_id, group in grouped:
        coords = []
        for _, row in group.iterrows():
            coords.append([row["Motor axis 0"], row["Motor axis 1"], row["Motor axis 2"]])

        valid_coords = [c for c in coords if not (c[0] == -1 and c[1] == -1 and c[2] == -1)]
        num_coords = len(valid_coords)
        if num_coords == 0:
            coords = [[-1, -1, -1]]
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
