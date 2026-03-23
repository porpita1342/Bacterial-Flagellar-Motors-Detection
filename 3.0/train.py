from Model.SegResNet import (
    SegResNetBackbone,
    AccelerateCoordinateLocalizationTrainer,
    DenseBCE,
    create_binary_targets,
    count_parameters,
    human_format,
)
from Data.dataloader import (TomogramTiler,
                              CustomDataset,
                              preprocess_dataframe,
                              )
import os
import time
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import Adam
from accelerate import Accelerator
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Utils.metrics import comp_score, detect_peaks, nms_coords, fbeta_score_coords, compute_fbeta
from collections import defaultdict
from Utils.utils import (set_seed, clear_gpu_memory ,ComprehensiveLogger)
from Utils.args import get_config
from Config.model_configs import (model_configs, get_train_df_cfg, get_train_loader_cfg, get_valid_loader_cfg)
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

    # Create a timestamped session directory to isolate logs and checkpoints
    session_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = os.path.join("sessions", session_name)
    cfg.log_dir = os.path.join(session_dir, "logs")
    cfg.checkpoint_dir = os.path.join(session_dir, "checkpoints")
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

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
    train_df_cfg = get_train_df_cfg(cfg)
    train_ds = CustomDataset(train_df, img_files_dir=train_img_paths, **train_df_cfg)
    valid_ds = CustomDataset(valid_df, img_files_dir=train_img_paths, **train_df_cfg)

    # Create dataloaders
    train_loader_cfg = get_train_loader_cfg(cfg)
    valid_loader_cfg = get_valid_loader_cfg(cfg)
    train_loader = DataLoader(train_ds, **train_loader_cfg)
    valid_loader = DataLoader(valid_ds, **valid_loader_cfg)

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

def train_epoch(epoch, logger, cfg):
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
    """Validate for one epoch, computing loss and F_beta detection score."""
    global model, trainer, valid_loader, accelerator, cfg

    model.eval()
    total_loss = 0
    num_batches = 0

    # Distance threshold in adjusted voxel space (1000 Å / target_voxel_spacing)
    threshold_voxels = 1000.0 / cfg.target_voxel_spacing

    # Per-tomo accumulators
    tomo_pred_coords = defaultdict(list)  # tomo_id -> [(z, y, x, conf), ...]
    tomo_gt_coords = defaultdict(set)     # tomo_id -> {(z, y, x), ...}
    tile_sz = None

    with torch.no_grad():
        for batch in valid_loader:
            batch_x = batch['image_tile']
            coordinates_list = [batch['local_coords'][i].tolist() for i in range(len(batch['local_coords']))]
            tile_origins = batch['tile_origin']   # (B, 3) tensor — tile centres in adjusted space
            tomo_ids = batch['tomo_id']

            if tile_sz is None:
                tile_sz = batch_x.shape[2:]  # (Z, Y, X)

            # Forward pass without augmentation
            targets_full = trainer.create_batch_targets(coordinates_list, batch_x.shape[2:])
            pred_final, pred_penultimate = model(batch_x)

            # Loss (match training: final + 0.5 * penultimate)
            targets_penult = trainer.downsample_targets(targets_full, scale_factor=0.5)
            loss_final, _ = trainer.criterion(pred_final, targets_full)
            loss_penult, _ = trainer.criterion(pred_penultimate, targets_penult)
            total_loss += (loss_final + 0.5 * loss_penult).item()
            num_batches += 1

            # Peak detection on the foreground channel
            fg_probs = torch.sigmoid(pred_final[:, 1])  # (B, Z, Y, X)

            for b in range(len(tomo_ids)):
                tomo_id = tomo_ids[b]
                origin = tile_origins[b].cpu()
                z1_off = int(origin[0].item()) - tile_sz[0] // 2
                y1_off = int(origin[1].item()) - tile_sz[1] // 2
                x1_off = int(origin[2].item()) - tile_sz[2] // 2

                # Ground truth: convert local tile coords to global adjusted-space coords
                for lz, ly, lx in coordinates_list[b]:
                    if lz < 0:
                        continue
                    tomo_gt_coords[tomo_id].add((
                        int(lz) + z1_off,
                        int(ly) + y1_off,
                        int(lx) + x1_off,
                    ))

                # Predictions: detect peaks and convert to global coords
                peaks, confidences = detect_peaks(fg_probs[b])
                for k in range(len(peaks)):
                    lz, ly, lx = peaks[k].tolist()
                    conf = confidences[k].item()
                    tomo_pred_coords[tomo_id].append((
                        lz + z1_off,
                        ly + y1_off,
                        lx + x1_off,
                        conf,
                    ))

    # Aggregate F_beta across all tomograms seen in this epoch
    all_tp = all_fp = all_fn = 0
    for tomo_id in set(tomo_gt_coords) | set(tomo_pred_coords):
        gt_list = list(tomo_gt_coords.get(tomo_id, set()))
        pred_list = nms_coords(tomo_pred_coords.get(tomo_id, []), threshold_voxels)
        tp, fp, fn = fbeta_score_coords(pred_list, gt_list, threshold_voxels, beta=2.0)
        all_tp += tp
        all_fp += fp
        all_fn += fn

    fbeta = compute_fbeta(all_tp, all_fp, all_fn, beta=2.0)
    avg_loss = total_loss / num_batches
    return avg_loss, fbeta

def train():
    """Main training function with simplified checkpointing and deletion"""
    global cfg
    cfg = set_everything_up()
    best_val_loss = float('inf')
    patience_counter = 0
    last_best_path = None  # Track the previous best model path

    # Scan checkpoint directory for existing best model files from previous runs
    existing_best = [
        os.path.join(cfg.checkpoint_dir, f)
        for f in os.listdir(cfg.checkpoint_dir)
        if f.startswith('best_model_epoch_') and f.endswith('.pt')
    ] if os.path.isdir(cfg.checkpoint_dir) else []
    if existing_best:
        # Find the one with the lowest val_loss
        for path in existing_best:
            try:
                ckpt = torch.load(path, map_location='cpu', weights_only=False)
                val = ckpt.get('val_loss', float('inf'))
                if val < best_val_loss:
                    best_val_loss = val
                    last_best_path = path
            except Exception:
                continue
        # Delete all other best model files except the true best
        for path in existing_best:
            if path != last_best_path:
                os.remove(path)
                if accelerator.is_main_process:
                    logger.main_logger.info(f"Cleaned up stale best model: {path}")
        if accelerator.is_main_process and last_best_path is not None:
            logger.main_logger.info(f"Resuming with best val_loss: {best_val_loss:.4f} from {last_best_path}")

    # Set validation loader epoch once for deterministic validation
    if hasattr(valid_loader.dataset, 'set_epoch'):
        valid_loader.dataset.set_epoch(0)

    for epoch in range(cfg.epochs):
        # Set epoch for dataset shuffling
        if hasattr(train_loader.dataset, 'set_epoch'):
            train_loader.dataset.set_epoch(epoch)

        # Train and validate
        train_loss, epoch_time = train_epoch(epoch, logger, cfg)
        val_loss, val_fbeta = validate_epoch()

        # Update scheduler
        scheduler.step(val_loss)

        # Log epoch results
        if accelerator.is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            logger.log_epoch(epoch, train_loss, val_loss, current_lr, epoch_time)
            logger.main_logger.info(f"Epoch {epoch:3d} | Val F2 (beta=2): {val_fbeta:.4f}")

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

def _load_full_tomogram(tomo_id, img_files_dir, tiler, scale_factor=1.0):
    """Load full tomogram as tensor (1, 1, Z, Y, X), optionally resampled."""
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

    # Resample if scale_factor != 1.0
    if abs(scale_factor - 1.0) >= 0.01:
        volume = F.interpolate(
            volume.unsqueeze(0),  # (1, C, Z, Y, X)
            scale_factor=[scale_factor] * 3,
            mode='trilinear',
            align_corners=False,
        ).squeeze(0)  # (C, Z, Y, X)

    return volume

def inference(valid_df, img_dir, target_voxel_spacing=10.0):
    """Run inference using MONAI's sliding window inferer with timing and logging"""
    global model, accelerator, logger

    model.eval()
    results = []
    logger.memory_check_and_log("Inference Start")

    tiler = TomogramTiler(base_path=img_dir)

    for idx, row in valid_df.iterrows():
        tomo_id = row['tomo_id']

        try:
            total_start_time = time.time()

            # Compute per-tomogram scale factor
            original_spacing = row['Voxel spacing']
            scale_factor = original_spacing / target_voxel_spacing

            # Load full tomogram (resampled to target voxel spacing)
            full_tomogram = _load_full_tomogram(tomo_id, img_dir, tiler, scale_factor=scale_factor).to(accelerator.device)

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
            pred_probs = torch.sigmoid(pred_final)

            inference_time = time.time() - inference_start_time
            total_time = time.time() - total_start_time

            # Competition metric requires post-processing (peak detection -> coordinates)
            # which doesn't exist yet, so score is None
            score = None

            num_tiles = (full_tomogram.shape[2] // 72) * (full_tomogram.shape[3] // 72) * (full_tomogram.shape[4] // 72)
            if accelerator.is_main_process:
                logger.log_inference(
                    tomo_id=tomo_id,
                    num_tiles=num_tiles,
                    inference_time=inference_time,
                    reconstruction_time=0.0,
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
    parser = get_config()
    cfg = parser.parse_args()

    if cfg.train or cfg.train_val:
        train()

    if cfg.val or cfg.test or cfg.train_val:
        # Re-parse config if train() wasn't called (cfg may not be set)
        if not (cfg.train or cfg.train_val):
            cfg = set_everything_up()

        img_dir = os.path.join(cfg.data_dir, 'test')
        inference_results = inference(valid_df, img_dir, target_voxel_spacing=cfg.target_voxel_spacing)

        if accelerator.is_main_process:
            logger.main_logger.info("Inference completed!")

            scored_results = [r for r in inference_results if r['score'] is not None]
            if scored_results:
                avg_score = sum(r['score'] for r in scored_results) / len(scored_results)
                logger.main_logger.info(f"  Average Score: {avg_score:.4f}")

            if inference_results:
                avg_tiles = sum(r['num_tiles'] for r in inference_results) / len(inference_results)
                avg_speed = sum(r['tiles_per_second'] for r in inference_results) / len(inference_results)
                logger.main_logger.info(f"Summary Statistics:")
                logger.main_logger.info(f"  Average Tiles per Tomogram: {avg_tiles:.1f}")
                logger.main_logger.info(f"  Average Processing Speed: {avg_speed:.1f} tiles/s")

            # Save final metrics
            logger.save_all_metrics()
