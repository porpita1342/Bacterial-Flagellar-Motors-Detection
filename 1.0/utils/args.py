
import configargparse


def get_config():
    p = configargparse.ArgParser(
        default_config_files=['configs/train.conf'])


    p.add('-c', '--config', is_config_file=True, help='Config file path')
    p.add('--data_dir', type=str, required=True, help='Path to dataset')
    p.add('mixup', type=bool, default=True, help="Enable Mixup for training")
    p.add('input_dimensions', type=list, default=[96,96,96], help="Define the input dimensions in terms of [z,y,x]")
    p.add('--train', action='store_true', help='Whether to run training stage (default: False)')
    p.add('--val', action='store_true', help='Whether to run validation stage (default: False)')
    p.add('--test', action='store_true', help='Whether to run test stage (default: False)')
    p.add('--train_val', action='store_true', help='Whether to run combined train+validation stage (default: False)')
    p.add('--train_porportions', type=str,default=0.8, help='The proportion of the dataset used for training. The rest is used for validation.')
    p.add('--seed',type=int, default=42, help='The seed used.')


    p.add('--batch_size_val', type=int, default=None, help='Batch size for validation (default: None, uses training batch size)')
    p.add('--use_custom_batch_sampler', action='store_true', help='Use a custom batch sampler for the DataLoader (default: False)')
    p.add('--val_df', type=str, default=None, help='Path to validation dataframe CSV file (default: None)')
    p.add('--test_df', type=str, default=None, help='Path to test dataframe CSV file (default: None)')
    p.add('--val_data_folder', type=str, default=None, help='Path to validation data folder (default: None)')
    p.add('--train_aug', type=str, default=None, help='Name of training augmentation pipeline (default: None)')
    p.add('--val_aug', type=str, default=None, help='Name of validation augmentation pipeline (default: None)')



    p.add('--lr', type=float, default=0.01, help='Learning rate')
    p.add('--epochs', type=int, default=10, help='Number of epochs')
    p.add('--batch_size', type=int, default=16, help='Batch size')
    p.add('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'], help='Optimizer to use (default: adam)')
    p.add('--seed', type=int, default=-1, help='Seed for the training')
    p.add('--distirbuted', type=bool, default=False, help='Enable distributed training')

    return p




# def sync_across_gpus(t, world_size):
#     torch.distributed.barrier()
#     gather_t_tensor = [torch.ones_like(t) for _ in range(world_size)]
#     torch.distributed.all_gather(gather_t_tensor, t)
#     return torch.cat(gather_t_tensor)

