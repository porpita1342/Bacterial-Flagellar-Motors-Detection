from types import SimpleNamespace
from Utils.args import get_config
from monai.transforms import (
    Compose, RandGaussianNoise, RandAdjustContrast, RandGaussianSmooth,
    Rand3DElastic, RandAffine, RandFlip, RandSpatialCrop
)

train_transforms = Compose([
    RandGaussianNoise(prob=0.2),
    RandAdjustContrast(prob=0.2, gamma=(0.9, 1.1)),
    RandGaussianSmooth(prob=0.2),

])

parser = get_config()
cfg = parser.parse_args()


model_configs = SimpleNamespace()

model_configs.segresnet_backbone = {
    'spatial_dims': 3,
    'in_channels': 1,
    'out_channels': 2,
    'init_filters': 8,
    'blocks_down': (1, 2, 2, 4),
    'blocks_up': (1, 1, 1),
    'dropout_prob': 0.2,
   # 'head_dropout_prob': 0.1,
}


train_df_cfg = {
    "tile_size": tuple(cfg.input_dimensions),
    "positive_ratio": cfg.positive_ratio,
    "transform": train_transforms,
    "dataset_size": cfg.dataset_size,
    "seed": 42,
    "target_voxel_spacing": cfg.target_voxel_spacing
}

train_loader_cfg = { 
    "batch_size": cfg.batch_size,
    "shuffle": cfg.shuffle,
    "num_workers": cfg.num_workers,
    "pin_memory": cfg.pin_memory,
    "prefetch_factor": 1
}