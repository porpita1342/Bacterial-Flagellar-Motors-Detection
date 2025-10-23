import os
import pandas as pd
import torch
from monai.data import Dataset
from monai.utils import set_determinism
import torchvision
from torchvision.io import decode_image 
from typing import Tuple, List, Dict, Union
import numpy as np 
SEED = 42
set_determinism(seed = SEED)

#reinstalling torch, torchvision with 
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
#solved the issue with decode_image()



def tiling(path,tomo_id, z1, z2, y1, y2, x1, x2): 
#path should to directly to the tomogram file 
#We need the two diagonal points to figure out the cube
#Instead of working out the tiles in the function before training time, we find out hte coordinates and extract them 
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
                img = img[:, y1:y2, x1:x2]   
                volume.append(img)
            else:
                raise ValueError(f"Slice {slice_idx} missing for tomo_id {tomo_id}.")

    volume = torch.stack(volume, dim=0).float()# (D, 1, H, W)
    volume = volume.permute(1, 0, 2, 3)  #make this shape (C, D, H, W)

    return volume


def preprocess_dataframe(df):
    # Group by tomo_id
    grouped = df.groupby('tomo_id')
    
    # Create a new DataFrame with one row per tomo_id
    processed_df = []
    
    for tomo_id, group in grouped:
        # Extract all coordinates
        coords = []
        for _, row in group.iterrows():
            coords.append([row["Motor axis 0"], row["Motor axis 1"], row["Motor axis 2"]])
        
        # Create a single row with all coordinates
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
                    tile_size: Union[int, Tuple[int, int, int]] = (32, 128, 128),): 
    
        z_range = tomo_row["Array shape (axis 0)"]
        y_range = tomo_row["Array shape (axis 1)"]
        x_range = tomo_row["Array shape (axis 2)"]

        tile_detph, tile_height, tile_width = tile_size[:3]
        tile_centre_z = np.random.randint(0 + tile_detph, z_range - tile_detph + 1)
        tile_centre_y = np.random.randint(0 + tile_height, y_range - tile_height + 1)
        tile_centre_x = np.random.randint(0 + tile_width, x_range - tile_width + 1)


        z1 = int(tile_centre_z - tile_detph // 2)
        y1 = int(tile_centre_y - tile_height // 2)
        x1 = int(tile_centre_x - tile_width // 2)
        #visualise a cude, (z1,y1,x1) is the bottom front left corner 

        z2 = int(tile_centre_z + tile_detph // 2)
        y2 = int(tile_centre_y + tile_height // 2)
        x2 = int(tile_centre_x + tile_width // 2)
        #(z2,y2,x2) is the top front right corner
        return ( tile_centre_z, tile_centre_x, tile_centre_y,
            int(z1), int(y1), int(x1), int(z2), int(y2), int(x2))

class TilesDataset(Dataset):
    def __init__(self,
                 train_df: pd.DataFrame,
                 img_files_dir: str,
                 tile: bool = True,
                 transform=None,
                 undersampling_rate:float = 0.2,
                 tile_size: Union[int, Tuple[int, int, int]] = (32, 128, 128),
                 random_tiles = True):
        
        super().__init__(data=train_df, transform=transform)
        self.undersampling_rate = int(1/undersampling_rate)
        self.df = preprocess_dataframe(train_df)
        self.random = random_tiles
        self.tile_size = tile_size
        self.img_files_dir = img_files_dir
        self.transform = transform      
        self.tile = tile
        self.unique_tomo_ids = train_df['tomo_id'].unique()
        self.unprocessed_df = train_df
        
        self.prev_outputs = [int(0)] * self.undersampling_rate 
        # Ensure the DataFrame has the 'Voxel spacing' column
        if 'Voxel spacing' not in train_df.columns:
            raise ValueError("The DataFrame must contain a 'Voxel spacing' column")
    def __len__(self):
        # return len(self.unique_tomo_ids)
        return 1000 #realistically i can have however many i want because i am geneerating random tiles each time anyway
    
    def __getitem__(self, idx):
        tomo_id = np.random.choice(self.unique_tomo_ids)
        tomo_row = self.df[self.df['tomo_id'] == tomo_id].iloc[0]
        tile_centre_z, tile_centre_x, tile_centre_y,z1,y1,x1,z2,y2,x2 = get_rand_coords(tomo_row=tomo_row, 
                                                                       tile_size = self.tile_size)

        #Since after the preprocess_df function, numerous rows for each coords are condensed into one
        #with all coordinates all condensed into a list of coordinates
        #for now just pick a random coord out of the list bruh 
        label_z, label_y, label_x = tomo_row['coordinates'][np.random.randint(len(tomo_row['coordinates']))]

        has_motor = 0.0
        while (self.prev_outputs[-self.undersampling_rate:] == [int(0)]*self.undersampling_rate
            and int(has_motor) == int(0)
            ):  
            (
                    tile_centre_z, tile_centre_x, tile_centre_y,
                    z1, y1, x1, z2, y2, x2
                ) = get_rand_coords(tomo_row=tomo_row, tile_size=self.tile_size)
            for (label_z,label_y, label_x) in tomo_row['coordinates']:
                if (z1< label_z < z2) and (y1<label_y<y2) and (x1<label_x<x2):

                    has_motor = 1.0
                    local_coords = (label_z - z1, label_y - y1, label_x - x1)


        


        self.prev_outputs.append(int(has_motor))
        self.prev_outputs = self.prev_outputs[1:] #drops the oldest record
        voxel_spacing = tomo_row['Voxel spacing']


        tile = tiling(path=self.img_files_dir,
                            tomo_id=tomo_id,
                            z1=z1,z2=z2,y1=y1,y2=y2,x1=x1,x2=x2)
        
        return {
             "image_tile": tile,
             "local_coords": torch.tensor(local_coords, dtype=torch.float32) if int(has_motor)==int(1) else torch.empty((0,3)),
             #mind that coords are local
             "tile_origin": torch.tensor([tile_centre_z, tile_centre_x, tile_centre_y]),
             "tomo_id": tomo_id,
             "has_motor": torch.tensor(has_motor),
             "voxel_spacing": torch.tensor(voxel_spacing),
             "OGSHAPE": [tomo_row[f'Array shape (axis {i})'] for i in range(3) ]
        }


if __name__ == "__main__":
    train_df = pd.read_csv('/mnt/raid0/Kaggle/DS/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv')
    train_img_paths = "/mnt/raid0/Kaggle/DS/byu-locating-bacterial-flagellar-motors-2025/train/"
    ds = TilesDataset(train_df, train_img_paths)
    print("Dataset length:", len(ds))
    sample = ds[0]
    print("Sample data")
    print("Tile image shape:", sample['image_tile'].shape)
    print("new shape after squeezing", sample['image_tile'].squeeze().shape)
    print("Tile origin:", sample['tile_origin'])
    print("Has motor:", sample['has_motor'])
    print("local coords shape:", sample['local_coords'].shape)
    print(f"local coords:{sample['local_coords']}")
    print("original tomogram shape:",sample['OGSHAPE'] )
         
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