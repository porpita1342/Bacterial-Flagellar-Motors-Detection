
import os
import random
import numpy as np
import torch
import gc 
import vtk
from vtk.util import numpy_support
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


def visualise_tensors(tensor,coordinate=None):
    point = coordinate.numpy()
    volume_np = tensor.numpy()
    volume_np = (volume_np - volume_np.min()) / (volume_np.max() - volume_np.min()) * 255
    volume_np = volume_np.astype(np.uint8)

    # Convert NumPy to VTK image data
    _,depth, height, width = volume_np.shape
    flat_data = volume_np.flatten(order='F')  # Column-major for VTK
    vtk_data_array = numpy_support.numpy_to_vtk(flat_data, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)

    # Create VTK image data object
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(width, height, depth)
    image_data.GetPointData().SetScalars(vtk_data_array)

    # Set spacing (optional, for real-world units)
    image_data.SetSpacing(1.0, 1.0, 1.0)

    # Volume mapper
    volume_mapper = vtk.vtkSmartVolumeMapper()
    volume_mapper.SetInputData(image_data)

    # Volume color and opacity transfer functions
    volume_color = vtk.vtkColorTransferFunction()
    volume_color.AddRGBPoint(0, 0.0, 0.0, 0.0)
    volume_color.AddRGBPoint(255, 1.0, 1.0, 1.0)

    volume_scalar_opacity = vtk.vtkPiecewiseFunction()
    volume_scalar_opacity.AddPoint(0, 0.0)
    volume_scalar_opacity.AddPoint(255, 1.0)

    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(volume_color)
    volume_property.SetScalarOpacity(volume_scalar_opacity)
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()

    # Create volume actor
    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    # Renderer, render window, interactor
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)
    points = vtk.vtkPoints()
    points.InsertNextPoint(point)

    # Add volume to renderer
    renderer.AddVolume(volume)
    renderer.SetBackground(0.1, 0.1, 0.2)
    render_window.SetSize(600, 600)
    if coordinate is not None: 

    # Step 2: Create a polydata object to hold the point
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)

        # Step 3: Add a glyph (e.g. sphere or simple vertex) to make the point visible
        # --- Option A: Simple point (faster, simpler) ---
        vertex_filter = vtk.vtkVertexGlyphFilter()
        vertex_filter.SetInputData(polydata)
        vertex_filter.Update()
        point_mapper = vtk.vtkPolyDataMapper()
        point_mapper.SetInputConnection(vertex_filter.GetOutputPort())

        point_actor = vtk.vtkActor()
        point_actor.SetMapper(point_mapper)
        point_actor.GetProperty().SetPointSize(10)
        point_actor.GetProperty().SetColor(1.0, 0.0, 0.0) 
        renderer.AddActor(point_actor)
    # Start rendering
    render_window.Render()
    render_window_interactor.Start()

