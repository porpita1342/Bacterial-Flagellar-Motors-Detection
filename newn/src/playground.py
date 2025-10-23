# %%
from Data.data_loading import TomogramDataset, FlattenedTileDataset
from components.Eval_metrics import distance_metric
import pandas as pd 
from monai.utils import set_determinism
from components.Eval_metrics import score
from ModelBuilding.SegResNet_detection import SegResNet_Detection_Model, SegResNet_Detection_Config
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import pandas as pd
import torchvision
print(torch.__version__)
print(torchvision.__version__)
print(torch.version.cuda)

SEED = 42
set_determinism(seed = SEED)




# %%
# def custom_collate(batch):
#     """
#     Custom collate function that handles variable-sized 3D volumes.
#     """
#     # Extract elements from the batch
#     images = [item['image'] for item in batch]  # List of tensors with different shapes
#     coords = [item['coords'] for item in batch]  # List of tensors
#     has_motor = torch.stack([item['has_motor'] for item in batch])  # Can be stacked as same shape
#     tomo_ids = [item['tomo_id'] for item in batch]  # List of strings/identifiers
#     voxel_spacings = torch.stack([item['voxel_spacing'] for item in batch])  # Can be stacked as same shape
    
#     # Return a dictionary with the batch data
#     return {
#         'image': images,  # List of tensors with different shapes
#         'coords': coords,  # List of tensors
#         'has_motor': has_motor,  # Tensor of shape [batch_size, 1]
#         'tomo_id': tomo_ids,  # List of strings/identifiers
#         'voxel_spacing': voxel_spacings  # Tensor of shape [batch_size, 1]
#     }   

# %%
train_df = pd.read_csv('/mnt/raid0/Kaggle/DS/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv')
train_img_paths = "/mnt/raid0/Kaggle/DS/byu-locating-bacterial-flagellar-motors-2025/train/"
# train_df = train_df[train_df['Number of motors'] <=1]  
#right now let us just focus on the cases with 1 or 0 motor. The Dataset class already supports multi label 
#we just need to create a multilabel loss, but not right now.

# train_df = train_df.head(5)


full_dataset = FlattenedTileDataset(TomogramDataset(train_df, train_img_paths))
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(
    full_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    #collate_fn= custom_collate 
    #Not needed since i am setting batch_size to 1
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    #collate_fn= custom_collate
)

config = SegResNet_Detection_Config(
    spatial_dims=3,
    in_channels=1,
    init_filters=32,
    dropout_prob=0.2,
    head_dropout_prob=0.1
)


# %%
# import pprint
# sample = train_ds[2]  # Get the 3rd item

# print("Image shape:", sample['image'].shape)
# print("Coordinates:", sample['coords'])
# print("Has motor:", sample['has_motor'])
# print("Tomogram ID:", sample['tomo_id'])
# print("Voxel spacing:", sample['voxel_spacing'])
# pp = pprint.PrettyPrinter(indent=4)
# if isinstance(sample, dict):
# 	pp.pprint(sample.keys())  
# else:
# 	pp.pprint(dir(sample)) 



# %%
import torch
import os
import time
import gc
import pandas as pd
from tqdm import tqdm

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

def save_checkpoint(model, optimizer, epoch, val_loss, val_fbeta, history, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_fbeta': val_fbeta,
        'history': history
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
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

    print(f"Starting training on device: {device}")
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        train_loss = 0.0
        train_cls_loss = 0.0
        train_coord_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
        for batch_data in train_pbar:
            volume = batch_data['image'].to(device)
            has_motor = batch_data['has_motor'].to(device)
            coords = batch_data['coords'].to(device)
            voxel_spacing = batch_data['voxel_spacing'].to(device)
            optimizer.zero_grad()
            outputs = model(
                volume,
                labels={"has_motor": has_motor, "coords": coords}
            )
            loss = outputs["loss"]
            loss_dict = outputs["loss_dict"]
            loss.backward()
            optimizer.step()
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
                volume = batch_data['image'].to(device)
                has_motor = batch_data['has_motor'].to(device)
                coords = batch_data['coords'].to(device)
                voxel_spacing = batch_data['voxel_spacing'].to(device)
                tomo_ids = batch_data['tomo_id']
                outputs = model(
                    volume,
                    labels={"has_motor": has_motor, "coords": coords}
                )
                loss = outputs["loss"]
                loss_dict = outputs["loss_dict"]
                batch_loss = loss.item()
                batch_cls_loss = loss_dict["cls_loss"].item()
                batch_coord_loss = loss_dict["coord_loss"].item()
                val_loss += batch_loss
                val_cls_loss += batch_cls_loss
                val_coord_loss += batch_coord_loss
                pred_has_motor = outputs["has_motor"].cpu().numpy()
                pred_coords = outputs["coordinates"].cpu().numpy()
                true_has_motor = has_motor.cpu().numpy()
                true_coords = coords.cpu().numpy()
                voxel_spacing_np = voxel_spacing.cpu().numpy()
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
        val_fbeta = score(
            val_target_df,
            val_pred_df,
            min_radius=min_radius,
            beta=beta
        )
        if scheduler is not None:
            if hasattr(scheduler, 'step'):
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
            # Optionally save best model checkpoint
            if save_checkpoints:
                checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
                save_checkpoint(model, optimizer, epoch, val_loss, val_fbeta, history, checkpoint_path)
        else:
            patience_counter += 1
            print(f"Validation loss didn't improve. Patience: {patience_counter}/{early_stopping_patience}")

        # Optionally save checkpoint every epoch
        if save_checkpoints:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
            save_checkpoint(model, optimizer, epoch, val_loss, val_fbeta, history, checkpoint_path)

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    time_elapsed = time.time() - start_time
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")

    # Optionally load best model weights
    if save_checkpoints and os.path.exists(os.path.join(checkpoint_dir, "best_model.pt")):
        checkpoint = torch.load(os.path.join(checkpoint_dir, "best_model.pt"))
        model.load_state_dict(checkpoint['model_state_dict'])

    clear_gpu_memory()
    return model, history



model = SegResNet_Detection_Model(config)

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Create optimizer and scheduler
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=5, 
    verbose=True
)

# Train the model
trained_model, history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=50,
    checkpoint_dir="checkpoints",
    scheduler=scheduler,
    early_stopping_patience=10,
    min_radius=1000.0,
    beta=2.0
)


# %%


# %%


# %%


# %%


# %%



