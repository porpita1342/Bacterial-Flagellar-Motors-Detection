import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Union, Tuple, Optional, Callable
from monai.utils import set_determinism
from torchvision.io import decode_image 
import cv2
# Remove unused imports: albumentations, torchvision, monai.data.Dataset


SEED = 42
set_determinism(seed = SEED)

#reinstalling torch, torchvision with 
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
#solved the issue with decode_image()

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

def preprocess_dataframe(df):
    """Group coordinates by tomo_id"""
    grouped = df.groupby('tomo_id')
    processed_df = []
    
    for tomo_id, group in grouped:
        coords = []
        for _, row in group.iterrows():
            coords.append([row["Motor axis 0"], row["Motor axis 1"], row["Motor axis 2"]])
        
        processed_df.append({
            'tomo_id': tomo_id,
            'coordinates': coords,
            'num_coords': len(coords),
            'Voxel spacing': group['Voxel spacing'].values[0],
            'Array shape (axis 0)': group['Array shape (axis 0)'].values[0],
            'Array shape (axis 1)': group['Array shape (axis 1)'].values[0],
            'Array shape (axis 2)': group['Array shape (axis 2)'].values[0],
        })
    
    return pd.DataFrame(processed_df)


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
    
    # Fixed bounds calculation
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
                tile_size: Union[int, Tuple[int, int, int]] = (32, 128, 128),
                positive_ratio: float = 0.2,
                transform: Optional[Callable] = None,
                augment: bool = True,
                dataset_size: int = 1000,
                seed: int = 42):
        
        super().__init__()  # Fixed: added parentheses
        
        if 'Voxel spacing' not in train_df.columns:
            raise ValueError("The DataFrame must contain a 'Voxel spacing' column")

        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.epoch_seed = seed

        self.df = preprocess_dataframe(train_df)
        self.unprocessed_df = train_df
        self.img_files_dir = img_files_dir
        self.tile_size = tile_size
        self.transform = transform
        self.augment = augment
        self.dataset_size = dataset_size
        self.unique_tomo_ids = train_df['tomo_id'].unique()
        self.positive_ratio = positive_ratio 
        self.tiler = TomogramTiler(base_path=img_files_dir)

    def set_epoch(self, epoch):
        """Call this at the start of each epoch for consistent shuffling"""
        self.epoch_seed = self.seed + epoch
        self.rng = np.random.RandomState(self.epoch_seed)
    
    def __len__(self): 
        return self.dataset_size 


    def _get_hard_negative(self, tomo_row, tomo_id, item_rng):
        """Generate tiles that definitely don't contain coordinates"""
        max_attempts = 50
        
        for attempt in range(max_attempts):
            tile_centre_z, tile_centre_y, tile_centre_x, z1, y1, x1, z2, y2, x2 = get_rand_coords(
                tomo_row=tomo_row, 
                tile_size=self.tile_size,
                item_rng=item_rng
            )
            
            # Check if this tile contains any coordinates
            has_coords = False
            for (label_z, label_y, label_x) in tomo_row['coordinates']:
                if (z1 <= label_z < z2) and (y1 <= label_y < y2) and (x1 <= label_x < x2):
                    has_coords = True
                    break
            
            if not has_coords:
                # Found a true negative tile
                tile = self.tiler.extract_tile(
                    path=self.img_files_dir, tomo_id=tomo_id,
                    z1=z1, z2=z2, y1=y1, y2=y2, x1=x1, x2=x2
                )
                return tile, 0.0, (-1, -1, -1), (tile_centre_z, tile_centre_y, tile_centre_x)
        
        # Fallback if can't find truly negative tile
        return self._get_random_tomogram(tomo_row, tomo_id, item_rng)
    
    def _get_positive_tomogram(self, tomo_row, tomo_id, item_rng): 
        if tomo_row['num_coords'] == 0:
            return None
            
        coord_idx = item_rng.randint(0, tomo_row['num_coords'])  # Fixed: use randint
        label_z, label_y, label_x = map(int, tomo_row['coordinates'][coord_idx])

        
        z, y, x = self.tile_size
        
        # Get tomogram bounds
        z_range = tomo_row["Array shape (axis 0)"]
        y_range = tomo_row["Array shape (axis 1)"]
        x_range = tomo_row["Array shape (axis 2)"]

        # Generate small random shift to avoid always centering (INTEGER shifts)
        max_shift = min(z//4, y//4, x//4, 5)
        shift_z = item_rng.randint(-max_shift, max_shift + 1)
        shift_y = item_rng.randint(-max_shift, max_shift + 1)
        shift_x = item_rng.randint(-max_shift, max_shift + 1)
        
        # Calculate tile bounds with shifts (ensure integers)
        z1 = int(max(0, label_z - z//2 + shift_z))
        y1 = int(max(0, label_y - y//2 + shift_y))
        x1 = int(max(0, label_x - x//2 + shift_x))
        
        z2 = int(min(z_range, z1 + z))
        y2 = int(min(y_range, y1 + y))
        x2 = int(min(x_range, x1 + x))
        
        # Adjust start coordinates if tile extends beyond bounds
        if z2 - z1 < z:
            z1 = max(0, z2 - z)
        if y2 - y1 < y:
            y1 = max(0, y2 - y)
        if x2 - x1 < x:
            x1 = max(0, x2 - x)

        tile_center_z = (z1 + z2) // 2
        tile_center_y = (y1 + y2) // 2
        tile_center_x = (x1 + x2) // 2
        tile = self.tiler.extract_tile(
            path=self.img_files_dir,
            tomo_id=tomo_id,
            z1=z1, z2=z2,
            y1=y1, y2=y2,
            x1=x1, x2=x2
        )
        local_coords = []
        for (label_z, label_y, label_x) in tomo_row['coordinates']:
            if (z1 <= label_z < z2) and (y1 <= label_y < y2) and (x1 <= label_x < x2):
                has_motor = 1.0
                local_coord = (label_z - z1, label_y - y1, label_x - x1)
                local_coords.append(local_coord)
                
        return tile, 1.0, local_coords, (tile_center_z, tile_center_y, tile_center_x)
    def _get_random_tomogram(self, tomo_row, tomo_id, item_rng): 
        """Extract a random tile and check if it contains motor coordinates"""
        tile_centre_z, tile_centre_y, tile_centre_x, z1, y1, x1, z2, y2, x2 = get_rand_coords(
            tomo_row=tomo_row, 
            tile_size=self.tile_size,
            item_rng=item_rng
        )
        
        has_motor = 0.0
        #local_coords = (0, 0, 0)
        tile_origin = (tile_centre_z, tile_centre_y, tile_centre_x)  # Consistent Z,Y,X order
        local_coords = []
        # Check if any coordinates fall within this tile
        for (label_z, label_y, label_x) in tomo_row['coordinates']:
            if (z1 <= label_z < z2) and (y1 <= label_y < y2) and (x1 <= label_x < x2):
                has_motor = 1.0
                local_coord = (label_z - z1, label_y - y1, label_x - x1)
                local_coords.append(local_coord)

        tile = self.tiler.extract_tile(
            path=self.img_files_dir,
            tomo_id=tomo_id,
            z1=z1, z2=z2,
            y1=y1, y2=y2,
            x1=x1, x2=x2
        )
        
        return tile, has_motor, local_coords, tile_origin
    def __getitem__(self, idx):
        item_seed = (self.epoch_seed + idx) % (2**32 - 1)
        item_rng = np.random.RandomState(item_seed)
        
        tomo_id = item_rng.choice(self.unique_tomo_ids)
        tomo_row = self.df[self.df['tomo_id'] == tomo_id].iloc[0]
        
        should_be_positive = item_rng.random() < self.positive_ratio
        
        if should_be_positive and tomo_row['num_coords'] > 0:
            result = self._get_positive_tomogram(tomo_row, tomo_id, item_rng)
            if result is not None:
                tile, has_motor, local_coords, tile_origin = result
            else:
                tile, has_motor, local_coords, tile_origin = self._get_hard_negative(tomo_row, tomo_id, item_rng)
        else:
            # Force negative samples when not supposed to be positive
            tile, has_motor, local_coords, tile_origin = self._get_hard_negative(tomo_row, tomo_id, item_rng)

        if int(has_motor) != 1: 
            local_coords = [(-1, -1, -1)] #to make sure consistent formatting
            
        # Apply transforms
        if self.transform and self.augment:
            tile = self.transform(tile)

        if not isinstance(tile, torch.Tensor):
            tile = torch.tensor(tile, dtype=torch.float32)
        
        return {
            "image_tile": tile,
            "has_motor": torch.tensor(has_motor, dtype=torch.float32),
            "local_coords": torch.tensor(local_coords, dtype=torch.float32),
            "tile_origin": torch.tensor(tile_origin, dtype=torch.float32),
            "tomo_id": tomo_id,
            "voxel_spacing": torch.tensor(tomo_row['Voxel spacing'], dtype=torch.float32),
            "original_shape": [tomo_row[f'Array shape (axis {i})'] for i in range(3)],
        }



class CustomInferenceDataset(Dataset):
    def __init__(self, tile_size, overlap): 





if __name__ == '__main__':
    train_df = pd.read_csv('/mnt/raid0/Kaggle/DS/new_byu-locating-bacterial-flagellar-motors-2025/train_labels.csv')
    img_files_dir = "/mnt/raid0/Kaggle/DS/new_byu-locating-bacterial-flagellar-motors-2025/train/"
    args = tile_size = (96,96,96)
    positive_ratio = 0.2
    transforms = None
    augment = False
    dataset_size = 500
    seed = 42


    args = {
        "tile_size": (96, 96, 96),
        "positive_ratio": 0.2,
        "transform": None,
        "augment": False,
        "dataset_size": 500,
        "seed": 42
    }
    ds = CustomDataset(train_df, img_files_dir,**args)
    for i in range(16): 
        print('-'*50)
        print(i)
        print(ds[i]['image_tile'].shape)
        print(ds[i]['local_coords'])
        print(ds[i]['tile_origin'])
        print(ds[i]['tomo_id'])
    



# def tiling(path, tomo_id, z1, z2, y1, y2, x1, x2): 
#     """Extract a 3D tile from tomogram slices"""

#     #path should to directly to the tomogram file 
#     #We need the two diagonal points to figure out the cube
#     #Instead of working out the tiles in the function before training time, we find out the coordinates and extract them 
#     #live during the training. 

#     tomo_path = os.path.join(path, tomo_id)
#     volume = []
#     all_images = [f for f in os.listdir(tomo_path) if os.path.isfile(os.path.join(tomo_path, f))]
#     slice_to_file = {int(os.path.splitext(img)[0].replace('slice_', '')): img for img in all_images}
    
#     for slice_idx in range(z1, z2):
#         if slice_idx in slice_to_file:
#             image = slice_to_file[slice_idx]
#             full_path = os.path.join(tomo_path, image)
#             img = cv2.imread(full_path)  
#             img = torch.from_numpy(img).float()/255.0      
#             img = img[:, y1:y2, x1:x2]   
#             volume.append(img)
#         else:
#             raise ValueError(f"Slice {slice_idx} missing for tomo_id {tomo_id}.")

#     volume = torch.stack(volume, dim=0).float()  # (Z, 1, Y, X)
#     volume = volume.permute(1, 0, 2, 3)  # (C, Z, Y, X)
#     return volume

# class TilesDataset(Dataset):
#     def __init__(self,
#                  train_df: pd.DataFrame,
#                  img_files_dir: str,
#                  tile_size: Union[int, Tuple[int, int, int]] = (32, 128, 128),
#                  transform: Optional[Callable] = None,
#                  augment: bool = True,
#                  dataset_size: int = 1000):

#         super().__init__()
        
#         # Validate required columns
#         if 'Voxel spacing' not in train_df.columns:
#             raise ValueError("The DataFrame must contain a 'Voxel spacing' column")
        
#         # Store configuration
#         self.df = preprocess_dataframe(train_df)
#         self.unprocessed_df = train_df
#         self.img_files_dir = img_files_dir
#         self.tile_size = tile_size
#         self.transform = transform
#         self.augment = augment
#         self.dataset_size = dataset_size
        
#         # Get unique tomogram IDs
#         self.unique_tomo_ids = train_df['tomo_id'].unique()
    
#     def __len__(self):
#         """Return fixed dataset size since we generate random tiles."""
#         return self.dataset_size
    
#     def __getitem__(self, idx):

#         # Use idx for deterministic randomness (reproducible across epochs)
#         np.random.seed(idx + hash(str(idx)) % 2**32)
#         tomo_id = np.random.choice(self.unique_tomo_ids)
        
#         # Get tomogram metadata
#         tomo_row = self.df[self.df['tomo_id'] == tomo_id].iloc[0]
        
#         # Generate random tile coordinates
#         tile_centre_z, tile_centre_x, tile_centre_y, z1, y1, x1, z2, y2, x2 = get_rand_coords(
#             tomo_row=tomo_row, 
#             tile_size=self.tile_size
#         )
        
#         has_motor = 0.0
#         local_coords = (0, 0, 0)
#         for (label_z, label_y, label_x) in tomo_row['coordinates']:
#             if (z1 < label_z < z2) and (y1 < label_y < y2) and (x1 < label_x < x2):
#                 has_motor = 1.0
#                 local_coords = (label_z - z1, label_y - y1, label_x - x1)
#                 break  # Found a motor, use this tile
        
#         # Load the tile data
#         tile = tiling(
#             path=self.img_files_dir,
#             tomo_id=tomo_id,
#             z1=z1, z2=z2,
#             y1=y1, y2=y2,
#             x1=x1, x2=x2
#         )
        
#         # Apply augmentations/transforms
#         if self.transform and self.augment:
#             tile = self.transform(tile)

#         if not isinstance(tile, torch.Tensor):
#             tile = torch.tensor(tile, dtype=torch.float32)
        
#         return {
#             "image_tile": tile,
#             "has_motor": torch.tensor(has_motor, dtype=torch.float32),
#             "local_coords": torch.tensor(local_coords, dtype=torch.float32) if has_motor == 1.0 else torch.empty((0, 3), dtype=torch.float32),
#             "tile_origin": torch.tensor([tile_centre_z, tile_centre_x, tile_centre_y], dtype=torch.float32),
#             "tomo_id": tomo_id,
#             "voxel_spacing": torch.tensor(tomo_row['Voxel spacing'], dtype=torch.float32),
#             "original_shape": [tomo_row[f'Array shape (axis {i})'] for i in range(3)]
#         }



# class TilesDataset(Dataset):
#     def __init__(self,
#                  train_df: pd.DataFrame,
#                  img_files_dir: str,
#                  tile: bool = True,
#                  transform=None,
#                  undersampling_rate:float = 0.2,
#                  tile_size: Union[int, Tuple[int, int, int]] = (32, 128, 128),
#                  random_tiles = True):
        
#         super().__init__(data=train_df, transform=transform)
#         self.undersampling_rate = int(1/undersampling_rate)
#         self.df = preprocess_dataframe(train_df)
#         self.random = random_tiles
#         self.tile_size = tile_size
#         self.img_files_dir = img_files_dir
#         self.transform = transform      
#         self.tile = tile
#         self.unique_tomo_ids = train_df['tomo_id'].unique()
#         self.unprocessed_df = train_df
        
#         self.prev_outputs = [int(0)] * self.undersampling_rate 
#         # Ensure the DataFrame has the 'Voxel spacing' column
#         if 'Voxel spacing' not in train_df.columns:
#             raise ValueError("The DataFrame must contain a 'Voxel spacing' column")
        
#     def __len__(self):
#         # return len(self.unique_tomo_ids)
#         return 1000 #realistically i can have however many i want because i am generating random tiles each time anyway
    
#     def __getitem__(self, idx):
#         tomo_id = np.random.choice(self.unique_tomo_ids)
#         tomo_row = self.df[self.df['tomo_id'] == tomo_id].iloc[0]
#         tile_centre_z, tile_centre_x, tile_centre_y,z1,y1,x1,z2,y2,x2 = get_rand_coords(tomo_row=tomo_row, 
#                                                                        tile_size = self.tile_size)

#         #Since after the preprocess_df function, numerous rows for each coords are condensed into one
#         #with all coordinates all condensed into a list of coordinates
#         #for now just pick a random coord out of the list bruh 
#         label_z, label_y, label_x = tomo_row['coordinates'][np.random.randint(len(tomo_row['coordinates']))]

#         has_motor = 0.0
#         max_attempts = 100
#         attempts = 0 #implementing the sampling logic in the Dataset class is not a good move. You are going to have issues
#         #during multithreading.
#         while (self.prev_outputs[-self.undersampling_rate:] == [int(0)]*self.undersampling_rate 
#             and int(has_motor) == int(0)
#             and attempts<max_attempts
#             ):  
#             (tile_centre_z, tile_centre_x, tile_centre_y,
#              z1, y1, x1, z2, y2, x2) = get_rand_coords(tomo_row=tomo_row, tile_size=self.tile_size)
#             for (label_z,label_y, label_x) in tomo_row['coordinates']:
#                 if (z1< label_z < z2) and (y1<label_y<y2) and (x1<label_x<x2):

#                     has_motor = 1.0
#                     local_coords = (label_z - z1, label_y - y1, label_x - x1)
#             attempts+=1

        

#         self.prev_outputs.append(int(has_motor))
#         self.prev_outputs = self.prev_outputs[1:] #drops the oldest record
#         voxel_spacing = tomo_row['Voxel spacing']


#         tile = tiling(path=self.img_files_dir,
#                             tomo_id=tomo_id,
#                             z1=z1,z2=z2,y1=y1,y2=y2,x1=x1,x2=x2)
        
#         return {
#              "image_tile": tile,
#              "local_coords": torch.tensor(local_coords, dtype=torch.float32) if int(has_motor)==int(1) else torch.empty((0,3)),
#              #mind that coords are local
#              "tile_origin": torch.tensor([tile_centre_z, tile_centre_x, tile_centre_y]),
#              "tomo_id": tomo_id,
#              "has_motor": torch.tensor(has_motor),
#              "voxel_spacing": torch.tensor(voxel_spacing),
#              "OGSHAPE": [tomo_row[f'Array shape (axis {i})'] for i in range(3) ]
#         }







# if __name__ == "__main__":
#     train_df = pd.read_csv('/mnt/raid0/Kaggle/DS/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv')
#     train_img_paths = "/mnt/raid0/Kaggle/DS/byu-locating-bacterial-flagellar-motors-2025/train/"
#     ds = TilesDataset(train_df, train_img_paths)
#     print("Dataset length:", len(ds))
#     sample = ds[0]
#     print("Sample data")
#     print("Tile image shape:", sample['image_tile'].shape)
#     print("new shape after squeezing", sample['image_tile'].squeeze().shape)
#     print("Tile origin:", sample['tile_origin'])
#     print("Has motor:", sample['has_motor'])
#     print("local coords shape:", sample['local_coords'].shape)
#     print(f"local coords:{sample['local_coords']}")
#     print("original tomogram shape:",sample['OGSHAPE'] )
         
# class TomogramDataset(Dataset): #This dataset returns a dictionary of tiles 
#     def __init__(self,
#                  train_df: pd.DataFrame,
#                  img_files_dir: str,
#                  tile: bool = True,
#                  transform=None,
#     ):
#         super().__init__(data=train_df, transform=transform)
#         self.df = preprocess_dataframe(train_df)
#         self.img_files_dir = img_files_dir
#         self.transform = transform      
#         self.tile = tile
#         self.unique_tomo_ids = trtile_detph, tile_height, tile_width,
#     def __len__(self):
#         return len(self.unique_tomo_ids)
        
#     def __getitem__(self, idx):
#         tomo_id = self.unique_tomo_ids[idx]
        
#         # Load the volume
#         # volume = []
#         # img_dir = os.path.join(self.img_files_dir, f"{tomo_id}")
#         # for filename in os.listdir(img_dir):
#         #     path = os.path.join(img_dir, filename)
#         #     img = decode_image(path)
#         #     volume.append(img)
#         # volume = torch.stack(volume, dim=0).float()
            

#         # if self.transform:
#         #     volume = self.transform(volume)
        
#         # Get the row for this tomogram
#         # tomo_row = self.df[self.df['tomo_id'] == tomo_id].iloc[0]
#         tomo_row = self.df[self.df['tomo_id'] == tomo_id]

#         # Get voxel spacing
#         voxel_spacing = tomo_row['Voxel spacing'].iloc[0]
        
#         # Get coordinates list directly from the 'coordinates' field
#         coords_list = tomo_row['coordinates'].iloc[0]
        
#         has_motor = 1.0  # Default: has motor
#         if len(coords_list) == 0 or (len(coords_list) == 1 and coords_list[0] == [-1, -1, -1]):
#             has_motor = 0.0
#             coords_tensor = torch.tensor([[-1, -1, -1]], dtype=torch.float32)
#         else:
#             coords_tensor = torch.tensor(coords_list, dtype=torch.float32)
        
#         if self.tile:
#             tile_dicts = []
#             tiles, positions = tiling(path=self.img_files_dir,train_df=self.unprocessed_df,
#                                       tomo_id=tomo_id,tile_size=(32,384,384), 
#                                       overlap_percent=0.25, 
#                                       return_positions=True)
#             for tile, (zb, yb, xb) in zip(tiles, positions): 
#                 local_coords = []
#                 for coord in coords_list:
#                     z, y, x = coord
#                     # Check if the coordinate is inside this tile
#                     if (zb <= z < zb + 32) and (yb <= y < yb + 384) and (xb <= x < xb + 384):
#                         local_coords.append((z - zb, y - yb, x - xb)) 
#                         #This is to adjust the global 3D coordinates to local coordinates.
#                 has_motor = len(local_coords) > 0
#                 tile_dicts.append({
#                     "image_tile": tile,
#                     "coords": torch.tensor(local_coords, dtype=torch.float32) if local_coords else torch.empty((0,3)),
#                     "tile_origin": (zb, yb, xb),
#                     "tomo_id": tomo_id,
#                     "has_motor": has_motor,
#                     "voxel_spacing": voxel_spacing,
#                 })

#             return tile_dicts


        # return {
        #     'image': volume,
        #     'coords': coords_tensor, 
        #     'has_motor': torch.tensor([has_motor], dtype=torch.float32),
        #     'tomo_id': tomo_id,
        #     'voxel_spacing': torch.tensor([voxel_spacing], dtype=torch.float32)
        # }

# class FlattenedTileDataset(Dataset):
#     def __init__(self, tomogram_dataset):
#         self.tomogram_dataset = tomogram_dataset
#         self.tile_map = []  # Maps flat index to (tomo_idx, tile_idx)
        
#         # Build a mapping from flat index to (tomogram_index, tile_index)
#         for tomo_idx in range(len(tomogram_dataset)):
#            
         
# class TomogramDataset(Dataset): #This dataset returns a dictionary of tiles 
#     def __init__(self,
#                  train_df: pd.DataFrame,
#                  img_files_dir: str,
#                  tile: bool = True,
#                  transform=None,
#     ):
#         super().__init__(data=train_df, transform=transform)
#         self.df = preprocess_dataframe(train_df)
#         self.img_files_dir = img_files_dir
#         self.transform = transform      
#         self.tile = tile
#         self.unique_tomo_ids = train_df['tomo_id'].unique()


#         self.unprocessed_df = train_df
        
#         # Ensure the DataFrame has the 'Voxel spacing' column
#         if 'Voxel spacing' not in train_df.columns:
#             raise ValueError("The DataFrame must contain a 'Voxel spacing' column")

#     def __len__(self):
#         return len(self.unique_tomo_ids)
        
#     def __getitem__(self, idx):
#         tomo_id = self.unique_tomo_ids[idx]
        
#         # Load the volume
#         # volume = []
#         # img_dir = os.path.join(self.img_files_dir, f"{tomo_id}")
#         # for filename in os.listdir(img_dir):
#         #     path = os.path.join(img_dir, filename)
#         #     img = decode_image(path)
#         #     volume.append(img)
#         # volume = torch.stack(volume, dim=0).float()
            

#         # if self.transform:
#         #     volume = self.transform(volume)
        
#         # Get the row for this tomogram
#         # tomo_row = self.df[self.df['tomo_id'] == tomo_id].iloc[0]
#         tomo_row = self.df[self.df['tomo_id'] == tomo_id]

#         # Get voxel spacing
#         voxel_spacing = tomo_row['Voxel spacing'].iloc[0]
        
#         # Get coordinates list directly from the 'coordinates' field
#         coords_list = tomo_row['coordinates'].iloc[0]
        
#         has_motor = 1.0  # Default: has motor
#         if len(coords_list) == 0 or (len(coords_list) == 1 and coords_list[0] == [-1, -1, -1]):
#             has_motor = 0.0
#             coords_tensor = torch.tensor([[-1, -1, -1]], dtype=torch.float32)
#         else:
#             coords_tensor = torch.tensor(coords_list, dtype=torch.float32)
        
#         if self.tile:
#             tile_dicts = []
#             tiles, positions = tiling(path=self.img_files_dir,train_df=self.unprocessed_df,
#                                       tomo_id=tomo_id,tile_size=(32,384,384), 
#                                       overlap_percent=0.25, 
#                                       return_positions=True)
#             for tile, (zb, yb, xb) in zip(tiles, positions): 
#                 local_coords = []
#                 for coord in coords_list:
#                     z, y, x = coord
#                     # Check if the coordinate is inside this tile
#                     if (zb <= z < zb + 32) and (yb <= y < yb + 384) and (xb <= x < xb + 384):
#                         local_coords.append((z - zb, y - yb, x - xb)) 
#                         #This is to adjust the global 3D coordinates to local coordinates.
#                 has_motor = len(local_coords) > 0
#                 tile_dicts.append({
#                     "image_tile": tile,
#                     "coords": torch.tensor(local_coords, dtype=torch.float32) if local_coords else torch.empty((0,3)),
#                     "tile_origin": (zb, yb, xb),
#                     "tomo_id": tomo_id,
#                     "has_motor": has_motor,
#                     "voxel_spacing": voxel_spacing,
#                 })

#             return tile_dicts


        # return {
        #     'image': volume,
        #     tiles = tomogram_dataset[tomo_idx]  # List of tile dicts
#             for tile_idx in range(len(tiles)):
#                 self.tile_map.append((tomo_idx, tile_idx))
    
#     def __len__(self):
#         return len(self.tile_map)
    
#     def __getitem__(self, idx):
#         tomo_idx, tile_idx = self.tile_map[idx]
#         tiles = self.tomogram_dataset[tomo_idx]
#         return tiles[tile_idx]


# class FlattenedTileDataset(Dataset):
#     def __init__(self, tomogram_dataset: Dataset):
#         self.tomogram_dataset = tomogram_dataset

#         # Build a flat index mapping: (tomo_index, tile_index)
#         self.flat_index = []
#         for tomo_idx in range(len(tomogram_dataset)):
#             tile_dicts = tomogram_dataset[tomo_idx]  # Only load metadata
#             self.flat_index.extend([(tomo_idx, tile_idx) for tile_idx in range(len(tile_dicts))])

#     def __len__(self):
#         return len(self.flat_index)

#     def __getitem__(self, idx):
#         tomo_idx, tile_idx = self.flat_index[idx]
#         tile_dicts = self.tomogram_dataset[tomo_idx]  # Only loads one tomogram
#         return tile_dicts[tile_idx]



###OUTPUT
# Dataset length: 648
# Sample data
# Sample has 80 tiles

# Tile 0:
# Tile image shape: torch.Size([32, 1, 384, 956])
# new shape after squeezing torch.Size([32, 384, 956])
# Tile origin: (0, 0, 0)
# Has motor: False
# Coords shape: torch.Size([0, 3])
# Coords:
# tensor([], size=(0, 3))

# Tile 1:
# Tile image shape: torch.Size([32, 1, 384, 956])
# new shape after squeezing torch.Size([32, 384, 956])
# Tile origin: (0, 0, 288)
# Has motor: False
# Coords shape: torch.Size([0, 3])
# Coords:
# tensor([], size=(0, 3))

# Tile 2:
# Tile image shape: torch.Size([32, 0, 384, 956])
# new shape after squeezing torch.Size([32, 0, 384, 956])
# Tile origin: (0, 288, 0)
# Has motor: False
# Coords shape: torch.Size([0, 3])
# Coords:
# tensor([], size=(0, 3))

# Tile 3:
# Tile image shape: torch.Size([32, 0, 384, 956])
# new shape after squeezing torch.Size([32, 0, 384, 956])
# Tile origin: (0, 288, 288)
# Has motor: False
# Coords shape: torch.Size([0, 3])
# Coords:
# tensor([], size=(0, 3))

# Tile 4:
# Tile image shape: torch.Size([32, 1, 384, 956])
# new shape after squeezing torch.Size([32, 384, 956])
# Tile origin: (24, 0, 0)
# Has motor: False
# Coords shape: torch.Size([0, 3])
# Coords:
# tensor([], size=(0, 3))


# def tiling(volume:torch.Tensor, 
#            tile_size: Union[int, Tuple[int,int,int]] = [32,384,384], #can either be an int or a tuple with size [int,int,int]
#            overlap_percent: float = 0.25,
#            return_positions: bool = True):
    
#     if isinstance(tile_size, int):
#         tile_size = (tile_size, tile_size, tile_size)
#     #volume dimension is Array size 0 (Z, number of slices), Array size 1 (height), Arrays Size 2 (width)  
#     volume = volume.squeeze()
#     D, H, W = volume.shape
#     d, h, w = tile_size
#     stride_d = int(d * (1 - overlap_percent))
#     stride_h = int(h * (1 - overlap_percent))
#     stride_w = int(w * (1 - overlap_percent))
#     tiles = []
#     positions = []
#     for z in range(0, D - d + 1, stride_d):
#         for y in range(0, H - h + 1, stride_h):
#             for x in range(0, W - w + 1, stride_w):
#                 tile = volume[z:z+d, y:y+h, x:x+w]
#                 tiles.append(tile)
#                 if return_positions:
#                     positions.append((z, y, x))
#     if return_positions:
#         return tiles, positions
#     return tiles



# def tiling(path: str,
#            train_df: pd.DataFrame,
#            tomo_id: str,
#            tile_size: Union[int, Tuple[int, int, int]] = (32, 384, 384),
#            overlap_percent: float = 0.25,
#            return_positions: bool = True):
    
#     if isinstance(tile_size, int):
#         tile_size = (tile_size, tile_size, tile_size)
    
#     row = train_df[train_df['tomo_id'] == tomo_id].iloc[0]
#     D, H, W = row['Array shape (axis 0)'], row['Array shape (axis 1)'], row['Array shape (axis 2)']
#     d, h, w = tile_size
#     stride_d = int(d * (1 - overlap_percent))
#     stride_h = int(h * (1 - overlap_percent))
#     stride_w = int(w * (1 - overlap_percent))
    
#     tiles = []
#     positions = []
    
#     tomo_path = os.path.jopath
#     # Create mapping from slice number to filename
#     slice_to_file = {int(os.path.splitext(img)[0].replace('slice_', '')): img for img in all_images}
    
#     for z in range(0, D - d + 1, stride_d):
#         volume = []
#         # Load slices for current tile
#         for slice_idx in range(z, z + d):
#             if slice_idx in slice_to_file:
#                 image = slice_to_file[slice_idx]
#                 full_path = os.path.join(tomo_path, image)
#                 img = decode_image(full_path)               
#                 volume.append(img)
#             else:
#                 raise ValueError(f"Slice {slice_idx} missing for tomo_id {tomo_id}.")
        
#         volume = torch.stack(volume, dim=0).float()  # (D, 1, H, W)
#         volume = volume.permute(1, 0, 2, 3)  #make is shape (C, D, H, W)
#         #volume = volume.squeeze(1) 
#         #I think this is where the issue lies. Squeezing the channel dimension 
#         for y in range(0, H - h + 1, stride_h):
#             for x in range(0, W - w + 1, stride_w):
#                 tile = volume[:, :, y:y+h, x:x+w]
#                 tiles.append(tile)
#                 if return_positions:
#                     positions.append((z, y, x))
    
#     if return_positions:
#         return tiles, positions
#     return tiles

### OUTPUT
# Dataset length: 648
# Sample data
# Image shape: torch.Size([500, 1, 924, 956])
# Coordinates shape: torch.Size([6, 3])
# The coordinates are: tensor([[235., 403., 137.],
#         [243., 363., 153.],
#         [222., 379., 144.],
#         [225., 262., 628.],
#         [225., 241., 643.],
#         [231., 289., 632.]])
# Has motor: 1.0
# Tomogram ID: tomo_00e463
# Voxel spacing: 19.700000762939453


#everything seems to be working fine