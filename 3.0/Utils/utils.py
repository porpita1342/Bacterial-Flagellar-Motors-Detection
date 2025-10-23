
import os
import random
import numpy as np
import torch
import gc 
import json
import logging
import time
from typing import Tuple, Dict
import csv
# import vtk
# from vtk.util import numpy_support


def set_seed(seed: int = 42):
    """Set the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def memory_check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)

    print(f"Total memory: {total_memory / 1e9:.2f} GB")
    print(f"Allocated memory: {allocated / 1e9:.2f} GB")
    print(f"Reserved memory: {reserved / 1e9:.2f} GB")

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

class ComprehensiveLogger:
    def __init__(self, log_dir: str = "logs", experiment_name: str = "coordinate_localization"):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup multiple loggers
        self._setup_loggers()
        
        # Metrics storage
        self.training_metrics = []
        self.validation_metrics = []
        self.inference_metrics = []
        self.memory_metrics = []
        self.system_metrics = []
    
    def _setup_loggers(self):
        """Setup separate loggers for different purposes"""
        
        # 1. Main training logger
        self.main_logger = self._create_logger(
            "main_training", 
            os.path.join(self.log_dir, f"{self.experiment_name}_main.log")
        )
        
        # 2. Memory monitoring logger
        self.memory_logger = self._create_logger(
            "memory_monitor", 
            os.path.join(self.log_dir, f"{self.experiment_name}_memory.log")
        )
        
        # 3. Inference logger
        self.inference_logger = self._create_logger(
            "inference", 
            os.path.join(self.log_dir, f"{self.experiment_name}_inference.log")
        )
        
        # 4. Error logger
        self.error_logger = self._create_logger(
            "errors", 
            os.path.join(self.log_dir, f"{self.experiment_name}_errors.log"),
            level=logging.ERROR
        )
    
    def _create_logger(self, name: str, log_file: str, level=logging.INFO):
        """Create a logger with file and console handlers"""
        logger = logging.getLogger(f"{self.experiment_name}_{name}")
        logger.setLevel(level)
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        # Console handler (only for main logger)
        if name == "main_training":
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            logger.addHandler(console_handler)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        return logger
    
    def memory_check_and_log(self, context: str = ""):
        """Enhanced memory check with logging"""
        if not torch.cuda.is_available():
            return None
            
        device = torch.device("cuda")
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        free_memory = total_memory - reserved
        
        memory_info = {
            'timestamp': time.time(),
            'context': context,
            'total_memory_gb': total_memory / 1e9,
            'allocated_memory_gb': allocated / 1e9,
            'reserved_memory_gb': reserved / 1e9,
            'free_memory_gb': free_memory / 1e9,
            'memory_utilization_percent': (allocated / total_memory) * 100
        }
        
        # Store metrics
        self.memory_metrics.append(memory_info)
        
        # Log to memory logger
        self.memory_logger.info(f"Memory Check - {context} | "
                               f"Total: {memory_info['total_memory_gb']:.2f}GB | "
                               f"Allocated: {memory_info['allocated_memory_gb']:.2f}GB | "
                               f"Reserved: {memory_info['reserved_memory_gb']:.2f}GB | "
                               f"Free: {memory_info['free_memory_gb']:.2f}GB | "
                               f"Utilization: {memory_info['memory_utilization_percent']:.1f}%")
        
        # Console output for main checks
        if context in ["Training Start", "Training End", "Epoch End"]:
            print(f"GPU Memory - {context}:")
            print(f"  Total: {memory_info['total_memory_gb']:.2f} GB")
            print(f"  Allocated: {memory_info['allocated_memory_gb']:.2f} GB")
            print(f"  Free: {memory_info['free_memory_gb']:.2f} GB")
            print(f"  Utilization: {memory_info['memory_utilization_percent']:.1f}%")
        
        return memory_info
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                  lr: float, epoch_time: float):
        """Log epoch results with memory check"""
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': lr,
            'epoch_time': epoch_time,
            'timestamp': time.time()
        }
        self.training_metrics.append(metrics)
        
        self.main_logger.info(f"Epoch {epoch:3d} | "
                             f"Train Loss: {train_loss:.4f} | "
                             f"Val Loss: {val_loss:.4f} | "
                             f"LR: {lr:.6f} | "
                             f"Time: {epoch_time:.2f}s")
        
        # Memory check at end of epoch
        self.memory_check_and_log(f"Epoch {epoch} End")
    
    def log_batch(self, epoch: int, batch_idx: int, batch_results: Dict, 
                  batch_time: float, memory_check_interval: int = 50):
        """Log batch results with periodic memory checks"""
        if batch_idx % 10 == 0:
            self.main_logger.info(f"Epoch {epoch} Batch {batch_idx:3d} | "
                                 f"Total: {batch_results['total_loss']:.4f} | "
                                 f"FG: {batch_results['final_fg_loss']:.4f} | "
                                 f"BG: {batch_results['final_bg_loss']:.4f} | "
                                 f"Time: {batch_time:.3f}s")
        
        # Memory check every N batches
        if batch_idx % memory_check_interval == 0:
            self.memory_check_and_log(f"Epoch {epoch} Batch {batch_idx}")
    
    def log_inference(self, tomo_id: str, num_tiles: int, inference_time: float, 
                     reconstruction_time: float, total_time: float, 
                     volume_shape: Tuple[int, int, int], score: float = None):
        """Log inference results with memory monitoring"""
        metrics = {
            'tomo_id': tomo_id,
            'num_tiles': num_tiles,
            'inference_time': inference_time,
            'reconstruction_time': reconstruction_time,
            'total_time': total_time,
            'volume_shape': volume_shape,
            'tiles_per_second': num_tiles / inference_time if inference_time > 0 else 0,
            'score': score,
            'timestamp': time.time()
        }
        self.inference_metrics.append(metrics)
        
        self.inference_logger.info(f"Inference {tomo_id} | "
                                  f"Tiles: {num_tiles} | "
                                  f"Inference: {inference_time:.2f}s | "
                                  f"Reconstruction: {reconstruction_time:.2f}s | "
                                  f"Total: {total_time:.2f}s | "
                                  f"Speed: {num_tiles/inference_time:.1f} tiles/s | "
                                  f"Shape: {volume_shape}" +
                                  (f" | Score: {score:.4f}" if score else ""))
        
        # Memory check after each tomogram
        self.memory_check_and_log(f"Inference {tomo_id} Complete")
    
    def log_error(self, error_msg: str, context: str = ""):
        """Log errors to separate error file"""
        self.error_logger.error(f"{context} | {error_msg}")
        
        # Also log memory state during errors
        memory_info = self.memory_check_and_log(f"Error - {context}")
        if memory_info and memory_info['memory_utilization_percent'] > 90:
            self.error_logger.warning(f"High memory usage detected: {memory_info['memory_utilization_percent']:.1f}%")
    
    def save_all_metrics(self):
        """Save all metrics to separate JSON files"""
        
        # 1. Training metrics
        training_file = os.path.join(self.log_dir, f"{self.experiment_name}_training_metrics.json")
        with open(training_file, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
        
        # 2. Inference metrics
        inference_file = os.path.join(self.log_dir, f"{self.experiment_name}_inference_metrics.json")
        with open(inference_file, 'w') as f:
            json.dump(self.inference_metrics, f, indent=2)
        
        # 3. Memory metrics
        memory_file = os.path.join(self.log_dir, f"{self.experiment_name}_memory_metrics.json")
        with open(memory_file, 'w') as f:
            json.dump(self.memory_metrics, f, indent=2)
        
        # 4. Combined metrics (for easy analysis)
        combined_file = os.path.join(self.log_dir, f"{self.experiment_name}_all_metrics.json")
        all_metrics = {
            'training': self.training_metrics,
            'validation': self.validation_metrics,
            'inference': self.inference_metrics,
            'memory': self.memory_metrics,
            'experiment_info': {
                'experiment_name': self.experiment_name,
                'total_epochs': len(self.training_metrics),
                'total_inferences': len(self.inference_metrics),
                'total_memory_checks': len(self.memory_metrics)
            }
        }
        
        with open(combined_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        self.main_logger.info(f"All metrics saved to {self.log_dir}")


# def visualise_tensors(tensor,coordinate=None):
#     point = coordinate.numpy()
#     volume_np = tensor.numpy()
#     volume_np = (volume_np - volume_np.min()) / (volume_np.max() - volume_np.min()) * 255
#     volume_np = volume_np.astype(np.uint8)

#     # Convert NumPy to VTK image data
#     _,depth, height, width = volume_np.shape
#     flat_data = volume_np.flatten(order='F')  # Column-major for VTK
#     vtk_data_array = numpy_support.numpy_to_vtk(flat_data, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)

#     # Create VTK image data object
#     image_data = vtk.vtkImageData()
#     image_data.SetDimensions(width, height, depth)
#     image_data.GetPointData().SetScalars(vtk_data_array)

#     # Set spacing (optional, for real-world units)
#     image_data.SetSpacing(1.0, 1.0, 1.0)

#     # Volume mapper
#     volume_mapper = vtk.vtkSmartVolumeMapper()
#     volume_mapper.SetInputData(image_data)

#     # Volume color and opacity transfer functions
#     volume_color = vtk.vtkColorTransferFunction()
#     volume_color.AddRGBPoint(0, 0.0, 0.0, 0.0)
#     volume_color.AddRGBPoint(255, 1.0, 1.0, 1.0)

#     volume_scalar_opacity = vtk.vtkPiecewiseFunction()
#     volume_scalar_opacity.AddPoint(0, 0.0)
#     volume_scalar_opacity.AddPoint(255, 1.0)

#     volume_property = vtk.vtkVolumeProperty()
#     volume_property.SetColor(volume_color)
#     volume_property.SetScalarOpacity(volume_scalar_opacity)
#     volume_property.ShadeOn()
#     volume_property.SetInterpolationTypeToLinear()

#     # Create volume actor
#     volume = vtk.vtkVolume()
#     volume.SetMapper(volume_mapper)
#     volume.SetProperty(volume_property)

#     # Renderer, render window, interactor
#     renderer = vtk.vtkRenderer()
#     render_window = vtk.vtkRenderWindow()
#     render_window.AddRenderer(renderer)
#     render_window_interactor = vtk.vtkRenderWindowInteractor()
#     render_window_interactor.SetRenderWindow(render_window)
#     points = vtk.vtkPoints()
#     points.InsertNextPoint(point)

#     # Add volume to renderer
#     renderer.AddVolume(volume)
#     renderer.SetBackground(0.1, 0.1, 0.2)
#     render_window.SetSize(600, 600)
#     if coordinate is not None: 

#     # Step 2: Create a polydata object to hold the point
#         polydata = vtk.vtkPolyData()
#         polydata.SetPoints(points)

#         # Step 3: Add a glyph (e.g. sphere or simple vertex) to make the point visible
#         # --- Option A: Simple point (faster, simpler) ---
#         vertex_filter = vtk.vtkVertexGlyphFilter()
#         vertex_filter.SetInputData(polydata)
#         vertex_filter.Update()
#         point_mapper = vtk.vtkPolyDataMapper()
#         point_mapper.SetInputConnection(vertex_filter.GetOutputPort())

#         point_actor = vtk.vtkActor()
#         point_actor.SetMapper(point_mapper)
#         point_actor.GetProperty().SetPointSize(10)
#         point_actor.GetProperty().SetColor(1.0, 0.0, 0.0) 
#         renderer.AddActor(point_actor)
#     # Start rendering
#     render_window.Render()
#     render_window_interactor.Start()

