from dataclasses import dataclass
from typing import Any
from collections import defaultdict

import os
import time
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import Adam
from accelerate import Accelerator
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from monai.inferers import sliding_window_inference

from Model.SegResNet import SegResNetBackbone, human_format
from Data.dataloader import TomogramTiler, CustomDataset, preprocess_dataframe, create_binary_targets
from Utils.aug import rotate, flip_3d, Mixup
from Utils.metrics import detect_peaks, nms_coords, fbeta_score_coords, compute_fbeta, DenseBCE
from Utils.utils import set_seed, clear_gpu_memory, ComprehensiveLogger
from Utils.args import get_config
from Config.model_configs import (model_configs, get_train_df_cfg, get_valid_df_cfg,
                                   get_train_loader_cfg, get_valid_loader_cfg)


# ================================
# SESSION STATE
# ================================

@dataclass
class TrainingSession:
    """Holds all stateful objects for a training run, eliminating module-level globals."""
    cfg: Any
    accelerator: Any
    model: Any
    trainer: Any
    optimizer: Any
    scheduler: Any
    train_loader: Any
    valid_loader: Any
    logger: Any
    train_df: Any
    valid_df: Any


# ================================
# TRAINER
# ================================

class AccelerateCoordinateLocalizationTrainer:
    """Wraps the model with training utilities for HuggingFace Accelerate."""

    def __init__(self, model, accelerator, weight, mixup_alpha=0.2, gaussian_blob_sigma=2.0):
        self.model = model
        self.accelerator = accelerator
        self.mixup = Mixup(mix_beta=mixup_alpha, mixadd=False)
        self.gaussian_blob_sigma = gaussian_blob_sigma
        self.criterion = DenseBCE(class_weights=weight)

        total_params = sum(p.numel() for p in self.model.parameters())
        if self.accelerator.is_main_process:
            print(f'Net parameters: {human_format(total_params)}')

    def create_batch_targets(self, coordinates_list, shape):
        """Create 2-channel Gaussian targets for an entire batch."""
        batch_targets = torch.zeros((len(coordinates_list), 2) + shape, device=self.accelerator.device)
        for i, coords in enumerate(coordinates_list):
            if isinstance(coords, torch.Tensor):
                coords = coords.tolist()
            batch_targets[i] = create_binary_targets(
                coordinates=coords, shape=shape, sigma=self.gaussian_blob_sigma
            ).to(self.accelerator.device)
        return batch_targets

    def downsample_targets(self, targets, scale_factor):
        """Downsample targets for penultimate-head deep supervision."""
        return F.interpolate(targets, scale_factor=scale_factor, mode='trilinear', align_corners=False)

    def train_step(self, batch_x, coordinates_list):
        """
        Full augmented forward pass:
          1. Build full-res targets
          2. Rotate + flip input and targets together
          3. Mixup input and targets together
          4. Downsample targets for penultimate supervision
          5. Forward pass → (final, penultimate) logits
          6. Loss = final_loss + 0.5 * penultimate_loss
        """
        targets_full = self.create_batch_targets(coordinates_list, batch_x.shape[2:])
        batch_x, targets_full = rotate(batch_x, targets_full)
        batch_x, targets_full = flip_3d(batch_x, targets_full)
        batch_x, targets_full = self.mixup(batch_x, targets_full)
        targets_penult = self.downsample_targets(targets_full, scale_factor=0.5)

        pred_final, pred_penultimate = self.model(batch_x)
        loss_final, class_losses_final = self.criterion(pred_final, targets_full)
        loss_penult, class_losses_penult = self.criterion(pred_penultimate, targets_penult)

        return {
            'total_loss': loss_final + 0.5 * loss_penult,
            'final_bg_loss': class_losses_final[0],
            'final_fg_loss': class_losses_final[1],
            'penult_bg_loss': class_losses_penult[0],
            'penult_fg_loss': class_losses_penult[1],
        }


# ================================
# SETUP
# ================================

def create_validation_set(train_df):
    """Split out 130 tomograms as the validation set."""
    valid_set = train_df.sample(n=130, random_state=42)
    train_df = train_df.drop(valid_set.index)
    print(f"Training tomograms: {len(train_df)}")
    print(f"Validation tomograms: {len(valid_set)}")
    return train_df, valid_set


def set_everything_up() -> TrainingSession:
    """Initialise and return a fully configured TrainingSession."""
    parser = get_config()
    cfg = parser.parse_args()
    set_seed(cfg.seed)

    session_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = os.path.join("sessions", session_name)
    cfg.log_dir = os.path.join(session_dir, "logs")
    cfg.checkpoint_dir = os.path.join(session_dir, "checkpoints")
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    logger = ComprehensiveLogger(log_dir=cfg.log_dir, experiment_name="BYU")
    logger.main_logger.info("Starting coordinate localization training...")

    accelerator = Accelerator()

    train_df = pd.read_csv(os.path.join(cfg.data_dir, 'train_labels.csv'))
    train_img_paths = os.path.join(cfg.data_dir, 'train')
    train_df = preprocess_dataframe(train_df)
    train_df, valid_df = create_validation_set(train_df)

    train_ds = CustomDataset(train_df, img_files_dir=train_img_paths, **get_train_df_cfg(cfg))
    valid_ds = CustomDataset(valid_df, img_files_dir=train_img_paths, **get_valid_df_cfg(cfg))

    train_loader = DataLoader(train_ds, **get_train_loader_cfg(cfg))
    valid_loader = DataLoader(valid_ds, **get_valid_loader_cfg(cfg))

    model = SegResNetBackbone(**model_configs.segresnet_backbone)
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    model, optimizer, scheduler, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, valid_loader
    )

    trainer = AccelerateCoordinateLocalizationTrainer(
        model=model,
        accelerator=accelerator,
        weight=torch.tensor(cfg.class_weights),
        mixup_alpha=cfg.mixup_beta,
    )

    logger.main_logger.info("Setup completed successfully!")
    return TrainingSession(
        cfg=cfg,
        accelerator=accelerator,
        model=model,
        trainer=trainer,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
        logger=logger,
        train_df=train_df,
        valid_df=valid_df,
    )


# ================================
# TRAINING AND VALIDATION
# ================================

def train_epoch(session: TrainingSession, epoch: int):
    """Run one training epoch. Returns (avg_loss, epoch_time_seconds)."""
    session.model.train()
    total_loss = 0
    num_batches = 0
    epoch_start_time = time.time()

    session.logger.memory_check_and_log(f"Epoch {epoch} Start")
    train_bar = tqdm(session.train_loader,
                     desc=f"[TRAINING] Epoch {epoch + 1}/{session.cfg.epochs}", unit="batch")

    for batch_idx, batch in enumerate(train_bar):
        batch_start_time = time.time()
        session.optimizer.zero_grad()

        batch_x = batch['image_tile']
        coordinates_list = [batch['local_coords'][i].tolist() for i in range(len(batch['local_coords']))]

        results = session.trainer.train_step(batch_x, coordinates_list)
        session.accelerator.backward(results['total_loss'])
        session.optimizer.step()

        total_loss += results['total_loss'].item()
        num_batches += 1

        if session.accelerator.is_main_process:
            session.logger.log_batch(epoch, batch_idx, results,
                                     time.time() - batch_start_time, memory_check_interval=25)

    return total_loss / num_batches, time.time() - epoch_start_time


def validate_epoch(session: TrainingSession):
    """Run one validation epoch. Returns (avg_loss, f2_score)."""
    session.model.eval()
    total_loss = 0
    num_batches = 0
    threshold_voxels = 1000.0 / session.cfg.target_voxel_spacing

    tomo_pred_coords = defaultdict(list)
    tomo_gt_coords = defaultdict(set)
    tile_sz = None

    with torch.no_grad():
        for batch in session.valid_loader:
            batch_x = batch['image_tile']
            coordinates_list = [batch['local_coords'][i].tolist() for i in range(len(batch['local_coords']))]
            tile_origins = batch['tile_origin']
            tomo_ids = batch['tomo_id']

            if tile_sz is None:
                tile_sz = batch_x.shape[2:]

            targets_full = session.trainer.create_batch_targets(coordinates_list, batch_x.shape[2:])
            pred_final, pred_penultimate = session.model(batch_x)

            targets_penult = session.trainer.downsample_targets(targets_full, scale_factor=0.5)
            loss_final, _ = session.trainer.criterion(pred_final, targets_full)
            loss_penult, _ = session.trainer.criterion(pred_penultimate, targets_penult)
            total_loss += (loss_final + 0.5 * loss_penult).item()
            num_batches += 1

            fg_probs = torch.sigmoid(pred_final[:, 1])

            for b in range(len(tomo_ids)):
                tomo_id = tomo_ids[b]
                origin = tile_origins[b].cpu()
                z1_off = int(origin[0].item()) - tile_sz[0] // 2
                y1_off = int(origin[1].item()) - tile_sz[1] // 2
                x1_off = int(origin[2].item()) - tile_sz[2] // 2

                for lz, ly, lx in coordinates_list[b]:
                    if lz < 0:
                        continue
                    tomo_gt_coords[tomo_id].add((int(lz) + z1_off, int(ly) + y1_off, int(lx) + x1_off))

                peaks, confidences = detect_peaks(fg_probs[b])
                for k in range(len(peaks)):
                    lz, ly, lx = peaks[k].tolist()
                    tomo_pred_coords[tomo_id].append(
                        (lz + z1_off, ly + y1_off, lx + x1_off, confidences[k].item()))

    all_tp = all_fp = all_fn = 0
    for tomo_id in set(tomo_gt_coords) | set(tomo_pred_coords):
        gt_list = list(tomo_gt_coords.get(tomo_id, set()))
        pred_list = nms_coords(tomo_pred_coords.get(tomo_id, []), threshold_voxels)
        tp, fp, fn = fbeta_score_coords(pred_list, gt_list, threshold_voxels, beta=2.0)
        all_tp += tp
        all_fp += fp
        all_fn += fn

    return total_loss / num_batches, compute_fbeta(all_tp, all_fp, all_fn, beta=2.0)


# ================================
# MAIN TRAINING LOOP
# ================================

def train() -> TrainingSession:
    """Full training loop with checkpointing and early stopping. Returns the session."""
    session = set_everything_up()
    best_val_loss = float('inf')
    patience_counter = 0
    last_best_path = None

    # Resume from any existing best checkpoint
    existing_best = [
        os.path.join(session.cfg.checkpoint_dir, f)
        for f in os.listdir(session.cfg.checkpoint_dir)
        if f.startswith('best_model_epoch_') and f.endswith('.pt')
    ] if os.path.isdir(session.cfg.checkpoint_dir) else []

    if existing_best:
        for path in existing_best:
            try:
                ckpt = torch.load(path, map_location='cpu', weights_only=False)
                val = ckpt.get('val_loss', float('inf'))
                if val < best_val_loss:
                    best_val_loss = val
                    last_best_path = path
            except Exception:
                continue
        for path in existing_best:
            if path != last_best_path:
                os.remove(path)
                if session.accelerator.is_main_process:
                    session.logger.main_logger.info(f"Cleaned up stale best model: {path}")
        if session.accelerator.is_main_process and last_best_path:
            session.logger.main_logger.info(
                f"Resuming with best val_loss: {best_val_loss:.4f} from {last_best_path}")

    if hasattr(session.valid_loader.dataset, 'set_epoch'):
        session.valid_loader.dataset.set_epoch(0)

    for epoch in range(session.cfg.epochs):
        if hasattr(session.train_loader.dataset, 'set_epoch'):
            session.train_loader.dataset.set_epoch(epoch)

        train_loss, epoch_time = train_epoch(session, epoch)
        val_loss, val_fbeta = validate_epoch(session)
        session.scheduler.step(val_loss)

        if session.accelerator.is_main_process:
            current_lr = session.optimizer.param_groups[0]['lr']
            session.logger.log_epoch(epoch, train_loss, val_loss, current_lr, epoch_time)
            session.logger.main_logger.info(f"Epoch {epoch:3d} | Val F2 (beta=2): {val_fbeta:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if session.accelerator.is_main_process:
                if last_best_path and os.path.exists(last_best_path):
                    os.remove(last_best_path)
                    session.logger.main_logger.info(f"Deleted old best model: {last_best_path}")
                checkpoint_path = os.path.join(session.cfg.checkpoint_dir, f'best_model_epoch_{epoch}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': session.accelerator.unwrap_model(session.model).state_dict(),
                    'optimizer_state_dict': session.optimizer.state_dict(),
                    'scheduler_state_dict': session.scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'config': vars(session.cfg),
                    'timestamp': time.time(),
                }, checkpoint_path)
                session.logger.main_logger.info(f"Saved new best model with val_loss: {val_loss:.4f}")
                last_best_path = checkpoint_path
        else:
            patience_counter += 1

        if epoch > 10 and epoch % 5 == 0 and session.accelerator.is_main_process:
            checkpoint_path = os.path.join(session.cfg.checkpoint_dir, f'model_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': session.accelerator.unwrap_model(session.model).state_dict(),
                'optimizer_state_dict': session.optimizer.state_dict(),
                'scheduler_state_dict': session.scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'config': vars(session.cfg),
                'timestamp': time.time(),
            }, checkpoint_path)
            session.logger.main_logger.info(f"Interval model saved with val_loss: {val_loss:.4f}")

        if patience_counter >= session.cfg.early_stopping_patience:
            if session.accelerator.is_main_process:
                session.logger.main_logger.info(
                    f"Early stopping triggered after {patience_counter} epochs")
            break

        clear_gpu_memory()

    if session.accelerator.is_main_process:
        session.logger.save_all_metrics()

    return session


# ================================
# INFERENCE
# ================================

def inference(session: TrainingSession, img_dir: str):
    """
    Sliding-window inference over all tomograms in session.valid_df.
    Uses cfg values for tile size, batch size, and overlap.
    """
    session.model.eval()
    results = []
    session.logger.memory_check_and_log("Inference Start")
    tiler = TomogramTiler(base_path=img_dir)

    for _, row in session.valid_df.iterrows():
        tomo_id = row['tomo_id']
        try:
            total_start_time = time.time()
            scale_factor = row['Voxel spacing'] / session.cfg.target_voxel_spacing
            full_tomogram = tiler.load_full_tomogram(
                tomo_id, scale_factor=scale_factor).to(session.accelerator.device)

            inference_start_time = time.time()
            pred_final = sliding_window_inference(
                inputs=full_tomogram,
                roi_size=tuple(session.cfg.inference_tile_size),
                sw_batch_size=session.cfg.inference_batch_size,
                predictor=session.model,
                overlap=session.cfg.inference_overlap,
                mode="gaussian",
                sw_device=session.accelerator.device,
            )
            pred_probs = torch.sigmoid(pred_final)
            inference_time = time.time() - inference_start_time
            total_time = time.time() - total_start_time

            num_tiles = (full_tomogram.shape[2] // 72) * \
                        (full_tomogram.shape[3] // 72) * \
                        (full_tomogram.shape[4] // 72)

            if session.accelerator.is_main_process:
                session.logger.log_inference(
                    tomo_id=tomo_id, num_tiles=num_tiles,
                    inference_time=inference_time, reconstruction_time=0.0,
                    total_time=total_time, volume_shape=pred_probs.shape, score=None)

            results.append({
                'tomo_id': tomo_id,
                'score': None,
                'num_tiles': num_tiles,
                'inference_time': inference_time,
                'reconstruction_time': 0.0,
                'total_time': total_time,
                'tiles_per_second': num_tiles / inference_time if inference_time > 0 else 0,
                'volume_shape': pred_probs.shape,
            })

        except Exception as e:
            if session.accelerator.is_main_process:
                session.logger.main_logger.error(f"Error processing {tomo_id}: {e}")
            continue

    session.logger.memory_check_and_log("Inference Complete")
    return results


# ================================
# ENTRY POINT
# ================================

if __name__ == "__main__":
    parser = get_config()
    cfg = parser.parse_args()

    session = None
    if cfg.train or cfg.train_val:
        session = train()

    if cfg.val or cfg.test or cfg.train_val:
        if session is None:
            session = set_everything_up()

        img_dir = os.path.join(session.cfg.data_dir, 'test')
        inference_results = inference(session, img_dir)

        if session.accelerator.is_main_process:
            session.logger.main_logger.info("Inference completed!")

            scored_results = [r for r in inference_results if r['score'] is not None]
            if scored_results:
                avg_score = sum(r['score'] for r in scored_results) / len(scored_results)
                session.logger.main_logger.info(f"  Average Score: {avg_score:.4f}")

            if inference_results:
                avg_tiles = sum(r['num_tiles'] for r in inference_results) / len(inference_results)
                avg_speed = sum(r['tiles_per_second'] for r in inference_results) / len(inference_results)
                session.logger.main_logger.info(f"Summary Statistics:")
                session.logger.main_logger.info(f"  Average Tiles per Tomogram: {avg_tiles:.1f}")
                session.logger.main_logger.info(f"  Average Processing Speed: {avg_speed:.1f} tiles/s")

            session.logger.save_all_metrics()
