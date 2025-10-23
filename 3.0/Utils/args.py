import configargparse

def get_parser():
    p = configargparse.ArgParser(
        default_config_files=[],
        description="Training and inference config"
    )
    # Config file path (YAML/INI/JSON)
    p.add('-c', '--config', is_config_file=True, help='Config file path')

    # Data and checkpoint dirs
    p.add('--data_dir', type=str, required=True, help='Path to dataset')
    p.add('--checkpoint_dir', type=str, required=True, help='Directory to save checkpoints')
    p.add('--inference_data_dir', type=str, default=None, help='Path to inference dataset')

    # Modes
    p.add('--train', action='store_true', help='Run training stage (default: False)')
    p.add('--val', action='store_true', help='Run validation stage (default: False)')
    p.add('--test', action='store_true', help='Run test stage (default: False)')
    p.add('--train_val', action='store_true', help='Run combined train+validation stage (default: False)')

    # Dataset & preprocessing
    p.add('--positive_ratio', type=float, default=0.2, help='Positive sample ratio')
    p.add('--dataset_size', type=int, default=8000, help='Number of samples in dataset')
    p.add('--target_voxel_spacing', type=float, default=10.0, help='Target voxel spacing')
    p.add('--input_dimensions', type=int, nargs=3, default=[96,96,96], help='Input dimension [z,y,x]')
    p.add('--gaussian_blob_sigma', type=float, default=2.0, help='Sigma for Gaussian blob in target generation')

    # Mixup options
    p.add('--mixup', action='store_true', help='Enable Mixup for training')
    p.add('--mixup_beta', type=float, default=0.2, help='Beta value for mixup')

    # Training hyperparams
    p.add('--lr', type=float, default=0.0001, help='Learning rate')  
    p.add('--epochs', type=int, default=100, help='Number of epochs')  
    p.add('--batch_size', type=int, default=16, help='Batch size')
    p.add('--shuffle', action='store_true', help='Shuffle dataset during training')
    p.add('--pin_memory', action='store_true', help='Pin memory during DataLoader')
    p.add('--num_workers', type=int, default=4, help='Number of DataLoader workers') 
    p.add('--early_stopping_patience', type=int, default=10, help='Early stopping patience')

    # Class weights and device
    p.add('--class_weights', type=float, nargs='+', default=[0.1, 0.9], help='Class weights [background, foreground]') 
    p.add('--device', type=str, default='cuda', help='Device to run on (cuda or cpu)')

    # Inference options
    p.add('--inference_batch_size', type=int, default=4, help='Batch size during inference')
    p.add('--inference_overlap', type=float, default=0.25, help='Sliding window overlap during inference')
    p.add('--inference_num_workers', type=int, default=4, help='Number of workers during inference')
    p.add('--inference_tile_size', type=int, nargs=3, default=[96,96,96], help='Tile size during inference')

    # Logging and checkpoint saving
    p.add('--model_save_path', type=str, default='./checkpoints/', help='Path to save model checkpoints')
    p.add('--log_dir', type=str, default='./logs/', help='Directory to save logs')
    p.add('--experiment_name', type=str, default='coordinate_localization', help='Experiment name for logging')

    # Seed
    p.add('--seed', type=int, default=42, help='Random seed')

    return p

def get_config(): 
    return get_parser()

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print(args)
