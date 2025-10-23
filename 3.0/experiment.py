from monai.inferers import sliding_window_inference
from Model.SegResNet import *
from torch.utils.data import DataLoader
from torch.optim import Adam
from Config.model_configs import (model_configs, train_df_cfg, train_loader_cfg)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Utils.args import get_config
from Utils.utils import (set_seed, clear_gpu_memory ,ComprehensiveLogger)
from torchsummary import summary

if __name__ == "__main__":
    exampler = torch.randn(4,1,760,760,280)
    parser = get_config()
    cfg = parser.parse_args()
    SEED = cfg.seed
    DATA_PATH = cfg.data_dir

    set_seed(SEED)
    
    model = SegResNetBackbone(**model_configs.segresnet_backbone)
    model.eval()
    # Create optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    # summary(model, (8,1, 96, 96, 96))  # Adjust to your input shape

    output = sliding_window_inference(inputs=exampler,roi_size=cfg.input_dimensions,predictor=model,overlap=0.25,sw_batch_size=1)
    print("Everything ran successfully")

    
    