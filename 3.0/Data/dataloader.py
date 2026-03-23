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
                    continue
            self._slice_mappings[tomo_id] = slice_mapping
        return self._slice_mappings[tomo_id]

    def extract_tile(self, tomo_id, z1, z2, y1, y2, x1, x2):
        """Extract a 3D tile from tomogram slices"""
        slice_mapping = self._get_slice_mapping(tomo_id)
        tomo_path = os.path.join(self.base_path, tomo_id)

        volume = torch.zeros((z2 - z1, y2 - y1, x2 - x1), dtype=torch.float32)

        for i, slice_idx in enumerate(range(z1, z2)):
            if slice_idx not in slice_mapping:
                raise ValueError(f"Slice {slice_idx} missing for tomo_id {tomo_id}")
            full_path = os.path.join(tomo_path, slice_mapping[slice_idx])
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load image: {full_path}")
            volume[i] = torch.from_numpy(img[y1:y2, x1:x2].astype(np.float32) / 255.0)

        return volume.unsqueeze(0)  # (1, Z, Y, X)

    def load_full_tomogram(self, tomo_id, scale_factor=1.0):
        """Load full tomogram as (1, Z, Y, X) tensor, optionally resampled."""
        slice_mapping = self._get_slice_mapping(tomo_id)
        z_dim = len(slice_mapping)
        if z_dim == 0:
            raise ValueError(f"No slices for {tomo_id}")

        first_path = os.path.join(self.base_path, tomo_id, slice_mapping[min(slice_mapping.keys())])
        first_img = cv2.imread(first_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        y_dim, x_dim = first_img.shape

        volume = torch.zeros((1, z_dim, y_dim, x_dim), dtype=torch.float32)
        for i, slice_idx in enumerate(sorted(slice_mapping.keys())):
            path = os.path.join(self.base_path, tomo_id, slice_mapping[slice_idx])
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            volume[0, i] = torch.from_numpy(img)

        if abs(scale_factor - 1.0) >= 0.01:
            volume = F.interpolate(
                volume.unsqueeze(0),
                scale_factor=[scale_factor] * 3,
                mode='trilinear',
                align_corners=False,
            ).squeeze(0)

        return volume  # (1, Z, Y, X)


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

    tile_centre_z = item_rng.randint(tile_depth // 2, z_range - tile_depth // 2)
    tile_centre_y = item_rng.randint(tile_height // 2, y_range - tile_height // 2)
    tile_centre_x = item_rng.randint(tile_width // 2, x_range - tile_width // 2)

    z1 = int(tile_centre_z - tile_depth // 2)
    y1 = int(tile_centre_y - tile_height // 2)
    x1 = int(tile_centre_x - tile_width // 2)
    z2 = int(tile_centre_z + tile_depth // 2)
    y2 = int(tile_centre_y + tile_height // 2)
    x2 = int(tile_centre_x + tile_width // 2)

    return (tile_centre_z, tile_centre_y, tile_centre_x,
            int(z1), int(y1), int(x1), int(z2), int(y2), int(x2))


def create_binary_targets(coordinates, shape, sigma=2.0):
    """
    Create 2-channel (background/foreground) soft Gaussian target volume.
    coordinates: list of (z, y, x) tuples
    shape: (Z, Y, X) spatial dimensions
    Returns: (2, Z, Y, X) tensor
    """
    target = torch.zeros((2,) + shape)
    target[0] = 1.0  # Background channel

    for coord in coordinates:
        z, y, x = coord
        if not (0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]):
            continue
        radius = int(3 * sigma)
        z_min, z_max = int(max(0, z - radius)), int(min(shape[0], z + radius + 1))
        y_min, y_max = int(max(0, y - radius)), int(min(shape[1], y + radius + 1))
        x_min, x_max = int(max(0, x - radius)), int(min(shape[2], x + radius + 1))
        zz, yy, xx = torch.meshgrid(torch.arange(z_min, z_max),
                                    torch.arange(y_min, y_max),
                                    torch.arange(x_min, x_max), indexing='ij')
        dist = torch.sqrt((zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2)
        foreground_prob = torch.exp(-dist ** 2 / (2 * sigma ** 2))
        target[1, z_min:z_max, y_min:y_max, x_min:x_max] = torch.maximum(
            target[1, z_min:z_max, y_min:y_max, x_min:x_max], foreground_prob)

    target[0] = 1.0 - target[1]
    return target


class TomogramSampler:
    """
    Handles positive, hard-negative, and random tile sampling strategies.
    Owns the geometric utilities (coordinate conversion, resampling, padding)
    shared across all sampling modes, keeping CustomDataset focused purely
    on dataset orchestration.
    """
    def __init__(self, tiler: TomogramTiler, tile_size: Tuple[int, int, int]):
        self.tiler = tiler
        self.tile_size = tile_size

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

    def _resample_tile(self, tile, scale_factor):
        if abs(scale_factor - 1.0) < 0.01:
            return tile
        return F.interpolate(
            tile.unsqueeze(0),
            scale_factor=[scale_factor] * 3,
            mode='trilinear',
            align_corners=False
        ).squeeze(0)

    def _trim_and_pad_tile(self, tile):
        """Trim tile to tile_size and zero-pad if undersized."""
        tile = tile[:, :self.tile_size[0], :self.tile_size[1], :self.tile_size[2]]
        pad_z = self.tile_size[0] - tile.shape[1]
        pad_y = self.tile_size[1] - tile.shape[2]
        pad_x = self.tile_size[2] - tile.shape[3]
        if pad_z > 0 or pad_y > 0 or pad_x > 0:
            tile = F.pad(tile, (0, pad_x, 0, pad_y, 0, pad_z), mode='constant', value=0)
        return tile

    def _extract_and_resample(self, tomo_id, z1, y1, x1, z2, y2, x2, scale_factor):
        """Convert to original space, extract, resample, and pad in one call."""
        oz1, oy1, ox1, oz2, oy2, ox2 = self._adjusted_to_original_bounds(
            z1, y1, x1, z2, y2, x2, scale_factor)
        tile = self.tiler.extract_tile(tomo_id=tomo_id,
                                       z1=oz1, z2=oz2, y1=oy1, y2=oy2, x1=ox1, x2=ox2)
        tile = self._resample_tile(tile, scale_factor)
        return self._trim_and_pad_tile(tile)

    def get_positive(self, tomo_row, tomo_id, item_rng):
        """Sample a tile guaranteed to contain at least one motor."""
        if tomo_row['num_coords'] == 0:
            return None

        scale_factor = tomo_row['scale_factor']
        coord_idx = 0 if int(tomo_row['num_coords']) == 1 else item_rng.randint(0, tomo_row['num_coords'] - 1)
        adjusted_label_z, adjusted_label_y, adjusted_label_x = map(int, tomo_row['coordinates'][coord_idx])

        tz, ty, tx = self.tile_size
        bound = 5
        z_range = tomo_row["Array shape (axis 0)"]
        y_range = tomo_row["Array shape (axis 1)"]
        x_range = tomo_row["Array shape (axis 2)"]

        def get_shift(label, size, array_range, bnd):
            half = size // 2
            max_neg = -min(half - bnd, label)
            max_pos = min(half - bnd, array_range - label)
            return 0 if max_neg > max_pos else item_rng.randint(max_neg, max_pos)

        z1 = y1 = x1 = z2 = y2 = x2 = 0
        motor_inside = False
        for _ in range(3):
            center_z = adjusted_label_z + get_shift(adjusted_label_z, tz, z_range, bound)
            center_y = adjusted_label_y + get_shift(adjusted_label_y, ty, y_range, bound)
            center_x = adjusted_label_x + get_shift(adjusted_label_x, tx, x_range, bound)

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

            if z1 <= adjusted_label_z < z2 and y1 <= adjusted_label_y < y2 and x1 <= adjusted_label_x < x2:
                motor_inside = True
                break

        if not motor_inside:
            z1 = max(0, min(adjusted_label_z - tz // 2, z_range - tz))
            y1 = max(0, min(adjusted_label_y - ty // 2, y_range - ty))
            x1 = max(0, min(adjusted_label_x - tx // 2, x_range - tx))
            z2, y2, x2 = z1 + tz, y1 + ty, x1 + tx

        tile = self._extract_and_resample(tomo_id, z1, y1, x1, z2, y2, x2, scale_factor)

        local_coords = [
            (cz - z1, cy - y1, cx - x1)
            for cz, cy, cx in tomo_row['coordinates']
            if z1 <= cz < z2 and y1 <= cy < y2 and x1 <= cx < x2
        ]
        tile_center = ((z1 + z2) // 2, (y1 + y2) // 2, (x1 + x2) // 2)
        return tile, 1.0, local_coords, tile_center

    def get_hard_negative(self, tomo_row, tomo_id, item_rng):
        """Sample a tile that does not overlap any known motor (up to 50 attempts)."""
        scale_factor = tomo_row['scale_factor']
        for _ in range(50):
            tile_centre_z, tile_centre_y, tile_centre_x, z1, y1, x1, z2, y2, x2 = get_rand_coords(
                tomo_row=tomo_row, tile_size=self.tile_size, item_rng=item_rng)
            has_coords = any(
                z1 <= cz < z2 and y1 <= cy < y2 and x1 <= cx < x2
                for cz, cy, cx in tomo_row['coordinates']
            )
            if not has_coords:
                tile = self._extract_and_resample(tomo_id, z1, y1, x1, z2, y2, x2, scale_factor)
                return tile, 0.0, [(-1, -1, -1)], (tile_centre_z, tile_centre_y, tile_centre_x)

        return self.get_random(tomo_row, tomo_id, item_rng)

    def get_random(self, tomo_row, tomo_id, item_rng):
        """Sample a fully random tile; label is set by whether any motor falls inside."""
        scale_factor = tomo_row['scale_factor']
        tile_centre_z, tile_centre_y, tile_centre_x, z1, y1, x1, z2, y2, x2 = get_rand_coords(
            tomo_row=tomo_row, tile_size=self.tile_size, item_rng=item_rng)

        tile = self._extract_and_resample(tomo_id, z1, y1, x1, z2, y2, x2, scale_factor)

        local_coords = [
            (cz - z1, cy - y1, cx - x1)
            for cz, cy, cx in tomo_row['coordinates']
            if z1 <= cz < z2 and y1 <= cy < y2 and x1 <= cx < x2
        ]
        has_motor = 1.0 if local_coords else 0.0
        if not local_coords:
            local_coords = [(-1, -1, -1)]
        return tile, has_motor, local_coords, (tile_centre_z, tile_centre_y, tile_centre_x)


class CustomDataset(Dataset):
    def __init__(self,
                 train_df: pd.DataFrame,
                 img_files_dir: str,
                 tile_size: Union[int, Tuple[int, int, int]] = (96, 96, 96),
                 positive_ratio: float = 0.2,
                 transform: Optional[callable] = None,
                 dataset_size: int = 1000,
                 seed: int = 42,
                 target_voxel_spacing: Optional[float] = None):

        super().__init__()

        if 'Voxel spacing' not in train_df.columns:
            raise ValueError("The DataFrame must contain a 'Voxel spacing' column")

        self.seed = seed
        self.epoch_seed = seed

        if target_voxel_spacing is None:
            self.target_voxel_spacing = np.median(train_df['Voxel spacing'].values)
            print(f"Auto-selected target voxel spacing: {self.target_voxel_spacing:.2f} Angstroms")
        else:
            self.target_voxel_spacing = target_voxel_spacing
            print(f"Using target voxel spacing: {self.target_voxel_spacing:.2f} Angstroms")

        self.df = self._preprocess_dataframe_with_voxel_spacing(train_df.copy())
        self.img_files_dir = img_files_dir
        self.tile_size = tile_size
        self.transform = transform
        self.dataset_size = dataset_size
        self.positive_ratio = positive_ratio

        tiler = TomogramTiler(base_path=img_files_dir)
        self.sampler = TomogramSampler(tiler=tiler, tile_size=tile_size)

    def _preprocess_dataframe_with_voxel_spacing(self, df):
        """
        Pre-calculate adjusted coordinates and shapes for voxel spacing normalisation.
        Avoids repeated coordinate transforms during training.
        """
        adjusted_df = df.copy()
        print("Pre-calculating adjusted coordinates for voxel spacing normalization...")

        for idx, row in adjusted_df.iterrows():
            original_spacing = row['Voxel spacing']
            scale_factor = original_spacing / self.target_voxel_spacing

            adjusted_coords = []
            for coord in row['coordinates']:
                if row['coordinates'][0][0] == -1:
                    adjusted_coords.append([-1, -1, -1])
                else:
                    adjusted_coords.append([
                        math.floor(coord[0] * scale_factor),
                        math.floor(coord[1] * scale_factor),
                        math.floor(coord[2] * scale_factor)
                    ])
            adjusted_df.at[idx, 'coordinates'] = adjusted_coords

            for axis in range(3):
                adjusted_df.at[idx, f'Array shape (axis {axis})'] = math.ceil(
                    row[f'Array shape (axis {axis})'] * scale_factor)

            adjusted_df.at[idx, 'Voxel spacing'] = self.target_voxel_spacing
            adjusted_df.at[idx, 'scale_factor'] = scale_factor

        print(f"Adjusted coordinates for {len(adjusted_df)} tomograms")
        return adjusted_df

    def set_epoch(self, epoch):
        self.epoch_seed = self.seed + epoch

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        item_seed = (self.epoch_seed + idx) % (2 ** 32 - 1)
        item_rng = np.random.RandomState(item_seed)

        if item_rng.random() < self.positive_ratio:
            tomo_row = self.df.loc[self.df['num_coords'] > 0].sample(n=1).iloc[0]
            tomo_id = tomo_row['tomo_id']
            tile, has_motor, local_coords, tile_origin = self.sampler.get_positive(tomo_row, tomo_id, item_rng)
        else:
            tomo_row = self.df.loc[self.df['num_coords'] == 0].sample(n=1).iloc[0]
            tomo_id = tomo_row['tomo_id']
            tile, has_motor, local_coords, tile_origin = self.sampler.get_hard_negative(tomo_row, tomo_id, item_rng)

        if int(has_motor) != 1:
            local_coords = [(-1, -1, -1)]

        if self.transform is not None:
            tile = self.transform(tile)

        if not isinstance(tile, torch.Tensor):
            tile = torch.tensor(tile, dtype=torch.float32)

        local_coords = torch.tensor(local_coords, dtype=torch.float32)
        pad = torch.ones((20 - local_coords.shape[0], 3)) * -1
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
