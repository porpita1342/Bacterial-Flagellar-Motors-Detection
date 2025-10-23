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
from accelerate import Accelerator
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Utils.metrics import comp_score
from Utils.utils import (set_seed, clear_gpu_memory ,ComprehensiveLogger)
from Utils.args import get_config
from Config.model_configs import (model_configs, train_df_cfg, train_loader_cfg)
from tqdm import tqdm
from monai.inferers import sliding_window_inference
import cv2
import numpy as np 

# ================================
# GLOBAL VARIABLES
# ================================

accelerator = None
train_df = None
valid_df = None
model = None
trainer = None
optimizer = None
scheduler = None
train_loader = None
valid_loader = None
logger = None

# ================================
# SETUP AND TRAINING FUNCTIONS
# ================================

def create_validation_set():
    """Create validation set from training data"""
    global train_df
    
    valid_set = train_df.sample(n=130, random_state=42)
    train_df = train_df.drop(valid_set.index)
    
    print(f"Training tomograms: {len(train_df)}")
    print(f"Validation tomograms: {len(valid_set)}")
    
    return valid_set

def set_everything_up(): 
    global accelerator, train_df, valid_df, model, trainer, optimizer, scheduler
    global train_loader, valid_loader, logger
    
    parser = get_config()
    cfg = parser.parse_args()
    SEED = cfg.seed
    DATA_PATH = cfg.data_dir

    set_seed(SEED)
    
    # Initialize logger
    logger = ComprehensiveLogger(log_dir=cfg.log_dir, experiment_name="BYU")
    logger.main_logger.info("Starting coordinate localization training...")
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Load and preprocess data
    train_df = pd.read_csv(os.path.join(DATA_PATH, 'train_labels.csv'))
    train_img_paths = os.path.join(DATA_PATH, 'train')
    train_df = preprocess_dataframe(train_df)
    
    # Create validation set
    valid_df = create_validation_set()
    
    # Create datasets
    train_ds = CustomDataset(train_df, img_files_dir=train_img_paths, **train_df_cfg)
    valid_ds = CustomDataset(valid_df, img_files_dir=train_img_paths, **train_df_cfg)
    
    # Create dataloaders
    train_loader = DataLoader(train_ds, **train_loader_cfg)
    valid_loader = DataLoader(valid_ds, **train_loader_cfg)
    
    # Create model
    model = SegResNetBackbone(**model_configs.segresnet_backbone)
    
    # Create optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Prepare everything with Accelerate
    model, optimizer, scheduler, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, valid_loader
    )
    
    # Create trainer AFTER preparing model
    trainer = AccelerateCoordinateLocalizationTrainer(
        model=model, 
        accelerator=accelerator,
        weight=torch.tensor(cfg.class_weights),
        mixup_alpha=cfg.mixup_beta
    )
    
    logger.main_logger.info("Setup completed successfully!")
    return cfg

def train_epoch(epoch, logger):
    """Enhanced training epoch with memory monitoring"""
    global model, trainer, optimizer, train_loader, accelerator
    
    model.train()
    total_loss = 0
    num_batches = 0
    epoch_start_time = time.time()
    
    # Memory check at start of epoch
    logger.memory_check_and_log(f"Epoch {epoch} Start")
    train_bar = tqdm(train_loader, desc=f"[TRAINING] Epoch {epoch + 1}/{cfg.epochs}", unit="batch")

    for batch_idx, batch in enumerate(train_bar):
        batch_start_time = time.time()
        
        optimizer.zero_grad()
        
        # Extract batch data
        batch_x = batch['image_tile']
        coordinates_list = [batch['local_coords'][i].tolist() for i in range(len(batch['local_coords']))]
        
        # Forward pass
        results = trainer.train_step(batch_x, coordinates_list)
        
        # Backward pass
        accelerator.backward(results['total_loss'])
        optimizer.step()
        
        total_loss += results['total_loss'].item()
        num_batches += 1
        
        batch_time = time.time() - batch_start_time
        
        # Log batch results with periodic memory checks
        if accelerator.is_main_process:
            logger.log_batch(epoch, batch_idx, results, batch_time, memory_check_interval=25)
    
    avg_loss = total_loss / num_batches
    epoch_time = time.time() - epoch_start_time
    
    return avg_loss, epoch_time


def validate_epoch():
    """Validate for one epoch"""
    global model, trainer, valid_loader, accelerator
    
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in valid_loader:
            batch_x = batch['image_tile']
            coordinates_list = [batch['local_coords'][i].tolist() for i in range(len(batch['local_coords']))]
            
            # Forward pass without MixUp
            targets_full = trainer.create_batch_targets(coordinates_list, batch_x.shape[2:])
            # targets_penult = trainer.downsample_targets(targets_full, scale_factor=0.5)
            
            pred_final, pred_penultimate = model(batch_x)
            
            loss_final, _ = trainer.criterion(pred_final, targets_full)
            # loss_penult, _ = trainer.criterion(pred_penultimate, targets_penult)
            
            # total_loss += (loss_final + 0.5 * loss_penult).item()
            total_loss += (loss_final).item()

            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss

def train():
    """Main training function with simplified checkpointing and deletion"""
    global cfg
    cfg = set_everything_up()
    best_val_loss = float('inf')
    patience_counter = 0
    last_best_path = None  # Track the previous best model path

    for epoch in range(cfg.epochs):
        # Set epoch for dataset shuffling
        if hasattr(train_loader.dataset, 'set_epoch'):
            train_loader.dataset.set_epoch(epoch)
        
        # Train and validate
        train_loss, epoch_time = train_epoch(epoch, logger)
        val_loss = validate_epoch()
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log epoch results
        if accelerator.is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            logger.log_epoch(epoch, train_loss, val_loss, current_lr, epoch_time)
        
        # Check and save if this is the new best model (with deletion)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            if accelerator.is_main_process:
                # Delete old best model if it exists
                if last_best_path is not None and os.path.exists(last_best_path):
                    os.remove(last_best_path)
                    logger.main_logger.info(f"Deleted old best model: {last_best_path}")
                
                # Save new best model
                checkpoint_path = os.path.join(cfg.checkpoint_dir, f'best_model_epoch_{epoch}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'config': vars(cfg),
                    'timestamp': time.time()
                }, checkpoint_path)
                
                logger.main_logger.info(f"Saved new best model with val_loss: {val_loss:.4f}")
                
                # Update tracker
                last_best_path = checkpoint_path
        else:
            patience_counter += 1
        
        if epoch > 10 and epoch % 5 == 0:
            if accelerator.is_main_process:
                checkpoint_path = os.path.join(cfg.checkpoint_dir, f'model_epoch_{epoch}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'config': vars(cfg),
                    'timestamp': time.time()
                }, checkpoint_path)
                
                logger.main_logger.info(f"Interval model saved with val_loss: {val_loss:.4f}")
        
        # Early stopping check
        if patience_counter >= cfg.early_stopping_patience:
            if accelerator.is_main_process:
                logger.main_logger.info(f"Early stopping triggered after {patience_counter} epochs")
            break
        
        clear_gpu_memory()
    
    # Save final metrics
    if accelerator.is_main_process:
        logger.save_all_metrics()

def _load_full_tomogram(tomo_id, img_files_dir, tiler):
    """Load full tomogram as tensor (1, 1, Z, Y, X)"""
    slice_mapping = tiler._get_slice_mapping(tomo_id)
    z_dim = len(slice_mapping)
    if z_dim == 0:
        raise ValueError(f"No slices for {tomo_id}")
    
    # Load first slice for Y/X dims
    first_path = os.path.join(img_files_dir, tomo_id, slice_mapping[min(slice_mapping.keys())])
    first_img = cv2.imread(first_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    y_dim, x_dim = first_img.shape
    
    volume = torch.zeros((1, z_dim, y_dim, x_dim), dtype=torch.float32)  # (C=1, Z, Y, X)
    for i, slice_idx in enumerate(sorted(slice_mapping.keys())):
        path = os.path.join(img_files_dir, tomo_id, slice_mapping[slice_idx])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        volume[0, i] = torch.from_numpy(img)
    
    return volume

def inference(valid_df, img_dir):
    """Run inference using MONAI's sliding window inferer with timing and logging"""
    global model, accelerator, logger
    
    model.eval()
    results = []
    logger.memory_check_and_log("Inference Start")

    # Initialize tiler (assuming it's defined elsewhere)
    tiler = TomogramTiler(base_path=img_dir)

    for idx, row in valid_df.iterrows():
        tomo_id = row['tomo_id']

        try:
            total_start_time = time.time()

            # Load full tomogram
            full_tomogram = _load_full_tomogram(tomo_id, img_dir, tiler).to(accelerator.device)

            inference_start_time = time.time()
            pred_final = sliding_window_inference(
                inputs=full_tomogram,
                roi_size=(96, 96, 96),
                sw_batch_size=4,
                predictor=model,
                overlap=0.25,
                mode="gaussian", 
                sw_device=accelerator.device
            )
            pred_probs = torch.sigmoid(pred_final)  # Apply sigmoid for probabilities

            inference_time = time.time() - inference_start_time
            total_time = time.time() - total_start_time

            # Compute score (assuming comp_score expects the probability tensor)
            score = comp_score(pred_probs, row) if 'comp_score' in globals() else None

            # Log results (adapted; num_tiles is approximate based on volume and roi_size)
            num_tiles = (full_tomogram.shape[2] // 72) * (full_tomogram.shape[3] // 72) * (full_tomogram.shape[4] // 72)  # Rough estimate (step ~96*0.75)
            if accelerator.is_main_process:
                logger.log_inference(
                    tomo_id=tomo_id,
                    num_tiles=num_tiles,
                    inference_time=inference_time,
                    reconstruction_time=0.0,  # No separate reconstruction needed
                    total_time=total_time,
                    volume_shape=pred_probs.shape,
                    score=score
                )
            
            results.append({
                'tomo_id': tomo_id,
                'score': score,
                'num_tiles': num_tiles,
                'inference_time': inference_time,
                'reconstruction_time': 0.0,
                'total_time': total_time,
                'tiles_per_second': num_tiles / inference_time if inference_time > 0 else 0,
                'volume_shape': pred_probs.shape
            })
                
        except Exception as e:
            if accelerator.is_main_process:
                logger.main_logger.error(f"Error processing {tomo_id}: {e}")
            continue

    logger.memory_check_and_log("Inference Complete")
    return results


if __name__ == "__main__":
    # Train the model
    train()
    
    # Run inference with detailed logging
    cfg = get_config().parse_args()
    img_dir = os.path.join(cfg.data_dir, 'test')



    inference_results = inference(valid_df, img_dir)
    
    if accelerator.is_main_process:
        logger.main_logger.info("Training and inference completed!")
        
        # Summary statistics
        avg_score = sum(r['score'] for r in inference_results if r['score']) / len(inference_results)
        avg_tiles = sum(r['num_tiles'] for r in inference_results) / len(inference_results)
        avg_speed = sum(r['tiles_per_second'] for r in inference_results) / len(inference_results)
        
        logger.main_logger.info(f"Summary Statistics:")
        logger.main_logger.info(f"  Average Score: {avg_score:.4f}")
        logger.main_logger.info(f"  Average Tiles per Tomogram: {avg_tiles:.1f}")
        logger.main_logger.info(f"  Average Processing Speed: {avg_speed:.1f} tiles/s")
        
        # Save final metrics
        logger.save_all_metrics()





# def inference(valid_df, img_dir):
#     """Run inference with comprehensive timing and tile counting"""
#     global model, accelerator, logger
    
#     model.eval()
#     results = []
#     logger.memory_check_and_log("Inference Start")

#     for idx, row in valid_df.iterrows():
#         tomo_id = row['tomo_id']

#         try:
#             total_start_time = time.time()

#             # Create inference dataset
#             inference_dataset = CustomInferenceDataset(
#                 tomo_id=tomo_id,
#                 img_files_dir=img_dir,
#                 train_df=valid_df,
#                 tile_size=(96, 96, 96),
#                 overlap=0.25
#             )
            
#             num_tiles = len(inference_dataset)
            
#             # Create dataloader
#             inference_loader = DataLoader(
#                 inference_dataset,
#                 batch_size=4,
#                 shuffle=False,
#                 num_workers=2
#             )
            
#             # Get reconstruction info
#             recon_info = inference_dataset.get_reconstruction_info()
            
#             # Initialize reconstructor
#             reconstruction_start_time = time.time()
#             reconstructor = VolumeReconstructor(
#                 volume_shape=recon_info["volume_shape"],
#                 tile_size=(96, 96, 96),
#                 overlap=0.25,
#                 blend_mode="gaussian"
#             )
            
#             # Run inference
#             inference_start_time = time.time()
#             tiles_processed = 0
            
#             with torch.no_grad():
#                 for batch in inference_loader:
#                     tiles = batch["image_tile"].to(accelerator.device)
#                     positions = batch["tile_position"]
                    
#                     # Forward pass
#                     pred_final, _ = model(tiles)
#                     pred_probs = torch.sigmoid(pred_final)
                    
#                     # Add tiles to reconstructor
#                     for i in range(tiles.shape[0]):
#                         tile_pred = pred_probs[i].cpu()
#                         tile_pos = tuple(positions[i].tolist())
#                         reconstructor.add_tile(tile_pred, tile_pos)
#                         tiles_processed += 1
            
#             inference_time = time.time() - inference_start_time
            
#             # Get final volume
#             final_volume = reconstructor.get_final_volume()
#             reconstruction_time = time.time() - reconstruction_start_time - inference_time
#             total_time = time.time() - total_start_time
            
#             # Compute score
#             score = comp_score(final_volume, row) if 'comp_score' in globals() else None
            
#             # Log detailed results
#             if accelerator.is_main_process:
#                 logger.log_inference(
#                     tomo_id=tomo_id,
#                     num_tiles=num_tiles,
#                     inference_time=inference_time,
#                     reconstruction_time=reconstruction_time,
#                     total_time=total_time,
#                     volume_shape=final_volume.shape,
#                     score=score
#                 )
            
#             results.append({
#                 'tomo_id': tomo_id,
#                 'score': score,
#                 'num_tiles': num_tiles,
#                 'inference_time': inference_time,
#                 'reconstruction_time': reconstruction_time,
#                 'total_time': total_time,
#                 'tiles_per_second': num_tiles / inference_time,
#                 'volume_shape': final_volume.shape
#             })
                
#         except Exception as e:
#             if accelerator.is_main_process:
#                 logger.main_logger.error(f"Error processing {tomo_id}: {e}")
#             continue
#     logger.memory_check_and_log("Inference Complete")

    
#     return results
