import pandas as pd 
from monai.utils import set_determinism
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import pandas as pd
import configargparse

import sys
import os
sys.path.append("Data")
sys.path.append("ModelBuilding")
sys.path.append("Utils")

from Utils import args 
from args import get_config, set_seed, sync_across_gpus
from ModelBuilding import (
    SegResNet_detection,
    detection_head,
)
from Data import (
    data_loading, 
    transforms
)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"



parser = get_config()
cfg = parser.parse_args() #cfg is Namespace, acts just like a dictionary except that you can access its variables like an attribute

















-











#im too dumb for distributed pytorch 
# def train(): 
#     global cfg
#     if cfg.seed < 0: 
#         cfg.seed = np.random.randint(1_000_100)
#     print("seed",cfg.seed)
#     set_seed(cfg.seed)

#     if cfg.distributed:

#         cfg.local_rank = int(os.environ["LOCAL_RANK"])
#         #When you are running on Multi-GPU, the LOCAL RANK stores the indexes for each GPU
#         #Say if you are running on 4 GPUs, cfg.local_rank would now be ONE OF THE intergers {0,1,2,3}
#         device = "cuda:%d" % cfg.local_rank
#         #Local_rank = 3 --> "cuda:3"
#         cfg.device = device

#         torch.cuda.set_device(cfg.local_rank)
#         #tells cuda that this process should be run on the one specified in local_rank.

#         torch.distributed.init_process_group(backend="nccl", init_method="env://")
#         #tells to use nccl backend which is super fast
#         cfg.world_size = torch.distributed.get_world_size()
#         #tells you the number of GPUs running, for our example, cfg.world_size = 4
#         cfg.rank = torch.distributed.get_rank()
#         #This is the global rank. Local rank gives the index for each GPU on a SINGLE machine. global rank scans over all machines and assign each GPU a unique number. 
#         #if we are using one machine only, cfg.rank = cfg.local_rank


#         print(f"Process {cfg.rank}, total {cfg.world_size}, local rank {cfg.local_rank}.")
#         cfg.group = torch.distributed.new_group(np.arange(cfg.world_size))

#         cfg.seed = int(
#             sync_across_gpus(torch.Tensor([cfg.seed]).to(device), cfg.world_size)
#             .detach()
#             .cpu()
#             .numpy()[0]
#         )

#         print(f"LOCAL_RANK {cfg.local_rank}, device {device}, seed {cfg.seed}")




# def eval_model(): 




if __name__ == "__main__":
    print(torch.__version__)






