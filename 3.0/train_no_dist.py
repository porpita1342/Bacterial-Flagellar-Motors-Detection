from Model.SegResNet import *
from Data.dataloader import (TomogramTiler,
                              CustomDataset, 
                              preprocess_dataframe,
                              )
import os
import time
import torch
import pandas as pd 
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Utils.metrics import comp_score
from Utils.utils import (set_seed, clear_gpu_memory ,ComprehensiveLogger)
from Utils.args import get_config
from Config.model_configs import (model_configs, train_df_cfg, train_loader_cfg)
from tqdm import tqdm
from monai.inferers import sliding_window_inference
import cv2
import numpy as np 

parser = get_config()
cfg = parser.parse_args()
SEED = cfg.seed
set_seed(SEED)
    
def create_validation_set():
    """Create validation set from training data"""
    global train_df
    
    valid_set = train_df[train_df['num_coords'] == 1].sample(frac=0.2, random_state=SEED)
    train_df = train_df.drop(valid_set.index)
    
    print(f"Training tomograms: {len(train_df)}")
    print(f"Validation tomograms: {len(valid_set)}")
    
    return valid_set
