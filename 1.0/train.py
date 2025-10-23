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
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from accelerate import Accelerator
from Utils import args 
from args import get_config, set_seed, sync_across_gpus
from models.Segresnet import SegResNet_Detection_Model,SegResNet_Detection_Config
from utils.metrics import comp_score
from utils.utils import (set_seed, 
                         memory_check, 
                         clear_gpu_memory, 
                         save_checkpoint)
from utils.args import get_config
from data.dataloader import TilesDataset

# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    accelerator,  # Add accelerator parameter
    num_epochs=50,
    checkpoint_dir="checkpoints",
    scheduler=None,
    early_stopping_patience=10,
    min_radius=1000.0,
    beta=2.0,
    save_checkpoints=False 
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_epoch = -1
    patience_counter = 0

    history = {
        'train_loss': [],
        'train_cls_loss': [],
        'train_coord_loss': [],
        'val_loss': [],
        'val_cls_loss': [],
        'val_coord_loss': [],
        'val_fbeta': []
    }

    print(f"Starting training on device: {accelerator.device}")
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        memory_check()
        model.train()
        train_loss = 0.0
        train_cls_loss = 0.0
        train_coord_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
        
        for batch_data in train_pbar:
            # Remove .to(device) calls - accelerator handles this
            volume = batch_data['image_tile']
            has_motor = batch_data['has_motor']
            coords = batch_data['local_coords']
            voxel_spacing = batch_data['voxel_spacing']
            
            optimizer.zero_grad()
            outputs = model(
                volume,
                labels={"has_motor": has_motor, "local_coords": coords}
            )
            loss = outputs["loss"]
            loss_dict = outputs["loss_dict"]
            
            # Use accelerator.backward instead of accelerate.backward
            accelerator.backward(loss)
            optimizer.step()
            # Remove scheduler.step() from here
            
            batch_loss = loss.item()
            batch_cls_loss = loss_dict["cls_loss"].item()
            batch_coord_loss = loss_dict["coord_loss"].item()
            train_loss += batch_loss
            train_cls_loss += batch_cls_loss
            train_coord_loss += batch_coord_loss
            train_pbar.set_postfix({
                'loss': f"{batch_loss:.4f}",
                'coord_loss': f"{batch_coord_loss:.4f}"
            })
            clear_gpu_memory()
            
        train_loss /= len(train_loader)
        train_cls_loss /= len(train_loader)
        train_coord_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_cls_loss = 0.0
        val_coord_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
            for batch_data in val_pbar:
                # Remove .to(device) calls
                volume = batch_data['image_tile']
                has_motor = batch_data['has_motor']
                coords = batch_data['local_coords']
                voxel_spacing = batch_data['voxel_spacing']
                tomo_ids = batch_data['tomo_id']
                
                outputs = model(
                    volume,
                    labels={"has_motor": has_motor, "local_coords": coords}
                )
                loss = outputs["loss"]
                loss_dict = outputs["loss_dict"]
                batch_loss = loss.item()
                batch_cls_loss = loss_dict["cls_loss"].item()
                batch_coord_loss = loss_dict["coord_loss"].item()
                val_loss += batch_loss
                val_cls_loss += batch_cls_loss
                val_coord_loss += batch_coord_loss
                
                # Gather predictions for distributed training
                pred_has_motor = accelerator.gather(outputs["has_motor"]).cpu().numpy()
                pred_coords = accelerator.gather(outputs["coordinates"]).cpu().numpy()
                true_has_motor = accelerator.gather(has_motor).cpu().numpy()
                true_coords = accelerator.gather(coords).cpu().numpy()
                voxel_spacing_np = accelerator.gather(voxel_spacing).cpu().numpy()
                
                for i in range(len(tomo_ids)):
                    val_predictions.append({
                        'tomo_id': tomo_ids[i],
                        'Has motor': 1 if pred_has_motor[i] > 0.5 else 0,
                        'Motor axis 0': pred_coords[i, 0],
                        'Motor axis 1': pred_coords[i, 1],
                        'Motor axis 2': pred_coords[i, 2],
                    })
                    val_targets.append({
                        'tomo_id': tomo_ids[i],
                        'Has motor': true_has_motor[i, 0],
                        'Motor axis 0': true_coords[i, 0],
                        'Motor axis 1': true_coords[i, 1],
                        'Motor axis 2': true_coords[i, 2],
                        'Voxel spacing': voxel_spacing_np[i, 0]
                    })
                val_pbar.set_postfix({
                    'loss': f"{batch_loss:.4f}",
                    'cls_loss': f"{batch_cls_loss:.4f}",
                    'coord_loss': f"{batch_coord_loss:.4f}"
                })
                
        val_loss /= len(val_loader)
        val_cls_loss /= len(val_loader)
        val_coord_loss /= len(val_loader)
        val_pred_df = pd.DataFrame(val_predictions)
        val_target_df = pd.DataFrame(val_targets)
        val_fbeta = comp_score(
            val_target_df,
            val_pred_df,
            min_radius=min_radius,
            beta=beta
        )
        
        if scheduler is not None:
            scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, CLS: {train_cls_loss:.4f}, COORD: {train_coord_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, CLS: {val_cls_loss:.4f}, COORD: {val_coord_loss:.4f}, F-beta: {val_fbeta:.4f}")

        history['train_loss'].append(train_loss)
        history['train_cls_loss'].append(train_cls_loss)
        history['train_coord_loss'].append(train_coord_loss)
        history['val_loss'].append(val_loss)
        history['val_cls_loss'].append(val_cls_loss)
        history['val_coord_loss'].append(val_coord_loss)
        history['val_fbeta'].append(val_fbeta)

        # Early stopping and best model logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            if save_checkpoints:
                checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
                save_checkpoint(model, optimizer, epoch, val_loss, val_fbeta, history, checkpoint_path)
        else:
            patience_counter += 1
            print(f"Validation loss didn't improve. Patience: {patience_counter}/{early_stopping_patience}")

        if save_checkpoints:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
            save_checkpoint(model, optimizer, epoch, val_loss, val_fbeta, history, checkpoint_path)

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    time_elapsed = time.time() - start_time
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    
    if save_checkpoints and os.path.exists(os.path.join(checkpoint_dir, "best_model.pt")):
        checkpoint = torch.load(os.path.join(checkpoint_dir, "best_model.pt"))
        accelerator.unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])

    clear_gpu_memory()
    return model, history


def set_everything_up():
    parser = get_config()
    cfg = parser.parse_args()
    SEED = cfg.seed
    set_seed(SEED)
    
    accelerator = Accelerator()
    DATA_PATH = cfg.data_dir
    train_df = pd.read_csv(os.path.join(DATA_PATH,'train_labels.csv'))
    train_img_paths = os.path.join(DATA_PATH,'train')

    full_dataset = TilesDataset(train_df, train_img_paths) 
    train_size = int(cfg.train_proportions * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
    config = SegResNet_Detection_Config()
    model = SegResNet_Detection_Model(config)
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Prepare all components with accelerator
    train_loader, val_loader, model, optimizer, scheduler = accelerator.prepare(
        train_loader, val_loader, model, optimizer, scheduler
    )
    
    return train_loader, val_loader, model, optimizer, scheduler, accelerator

if __name__ == '__main__': 
    train_loader, val_loader, model, optimizer, scheduler, accelerator = set_everything_up()
    
    # Remove manual device placement when using accelerator
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=accelerator.device,  # Use accelerator's device
        scheduler=scheduler,
        accelerator=accelerator,
    )


class CustomInferenceDataset(Dataset):
    def __init__():
        return None 



inference_dataset = CustomInferenceDataset(cfg.input_dimensions,cfg.overlap)


















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






