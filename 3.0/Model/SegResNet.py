import torch.nn as nn
import torch
from typing import List, Tuple
from torch import Tensor
import torch.nn.functional as F
from monai.networks.blocks import ResBlock
from monai.networks.blocks.segresnet_block import get_conv_layer, get_upsample_layer
from monai.networks.layers import get_norm_layer, get_act_layer, Dropout
from monai.utils import UpsampleMode
from Utils.aug import rotate, flip_3d, Mixup


class AccelerateCoordinateLocalizationTrainer:
    """Modified trainer that works with Accelerate"""
    
    def __init__(self, model, accelerator, weight, mixup_alpha=0.2, gaussian_blob_sigma=2.0):
        self.model = model  # Already prepared by Accelerate
        self.accelerator = accelerator
        self.mixup = Mixup(mix_beta=mixup_alpha, mixadd=False)
        self.gaussian_blob_sigma = gaussian_blob_sigma
        self.criterion = DenseBCE(class_weights=weight)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        if self.accelerator.is_main_process:
            print(f'Net parameters: {human_format(total_params)}')
    
    def create_batch_targets(self, coordinates_list, shape):
        """Create targets for entire batch"""
        batch_size = len(coordinates_list)
        batch_targets = torch.zeros((batch_size, 2) + shape, device=self.accelerator.device)
        
        for i, coords in enumerate(coordinates_list):
            if isinstance(coords, torch.Tensor):
                coords = coords.tolist()
            batch_targets[i] = create_binary_targets(
                coordinates=coords, shape=shape, sigma=self.gaussian_blob_sigma
            ).to(self.accelerator.device)
            
        return batch_targets
    
    def downsample_targets(self, targets, scale_factor):
        """Downsample targets for penultimate supervision"""
        return F.interpolate(targets, scale_factor=scale_factor, mode='trilinear', align_corners=False)
    
    def train_step(self, batch_x, coordinates_list):
        """Training step with Accelerate support"""
        targets_full = self.create_batch_targets(coordinates_list, batch_x.shape[2:])
        targets_penult = self.downsample_targets(targets_full, scale_factor=0.5)
        
        # Apply MixUp
        targets_full, targets_penult = rotate(targets_full,targets_penult)
        targets_full, targets_penult = flip_3d(targets_full,targets_penult)

        mixed_x, mixed_targets_full = self.mixup(batch_x, targets_full)
        mixed_x, mixed_targets_penult = self.mixup(mixed_x, targets_penult)
        
        pred_final, pred_penultimate = self.model(mixed_x)
        
        # Compute losses
        loss_final, class_losses_final = self.criterion(pred_final, mixed_targets_full)
        loss_penult, class_losses_penult = self.criterion(pred_penultimate, mixed_targets_penult)
        
        total_loss = loss_final + 0.5 * loss_penult
        
        return {
            'total_loss': total_loss,
            'final_bg_loss': class_losses_final[0],
            'final_fg_loss': class_losses_final[1],
            'penult_bg_loss': class_losses_penult[0],
            'penult_fg_loss': class_losses_penult[1]
        }



class SegResNetBackbone(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: float | None = None,
        act: tuple | str = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE, #tells you it can either be Enum UpsampleMode or a string. Default is NONTRAINABLE
    ): 
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act  # input options
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final
        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters)
        self.down_layers = self._make_down_layers()
        self.up_layers, self.up_samples = self._make_up_layers()
        self.conv_final = self._make_final_conv(out_channels, in_channels=self.init_filters)
        self.conv_penultimate = self._make_final_conv(out_channels,in_channels=self.init_filters*2)

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_down_layers(self):
        down_layers = nn.ModuleList() #this is to create an empty list specifically designed to hold pytorch modules (layers)
        #You can index it just like a normal python list

        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        # blocks_down: tuple = (1, 2, 2, 4),

        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2**i #Doubles the number of channels per layer. 
            #layer 1: 8 *2 
            #layer 2: 8 *2*2 
            #layer 3: 8 *2*2*2 
            #layer 4: 8 *2*2*2*2 
            pre_conv = (
                get_conv_layer(spatial_dims = spatial_dims, in_channels=layer_in_channels // 2, out_channels=layer_in_channels, stride=2, bias=False) if i > 0 else nn.Identity()
            )   #You can see that out_channel is double the in_channel
            #nn.Identity() when i == 0 is because you do not perform a convolution on the input image. We pass that through the Resblock first
            down_layer = nn.Sequential(
                pre_conv, *[ResBlock(spatial_dims=spatial_dims, in_channels=layer_in_channels, norm=norm, act=self.act) for _ in range(item)]# Remember that * is the unpacking operator
                          #These operators unpacks the list of ResBlocks and nn.Sequential treat each of the element (each Resblock) as a separate parameter
            ) #this defines the number of ResBlocks per layer. Because we defined 1,2,2,4
            down_layers.append(down_layer)
        return down_layers


    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList() #Initialise them to contain the layers
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        #blocks_up: tuple = (1, 1, 1)
        n_up = len(blocks_up)
        for i in range(n_up): #iterates from 0 to 2 
            sample_in_channels = filters * 2 ** (n_up - i) 
            #Remember that init_filters = 8 
            # When i=0: 8 * 2 ** 3 = 64
            # When i=1: 8 * 2 ** 2 = 32
            # When i=2: 8 * 2 ** 1 = 16
            up_layers.append(
                nn.Sequential(
                    *[
                        ResBlock(spatial_dims=spatial_dims, in_channels=sample_in_channels // 2, norm=norm, act=self.act)
                        for _ in range(blocks_up[i]) #This depends on how we defined blocks_up, but since we did (1,1,1) 
                        #There will only be one Resblock per upsample layer
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims=spatial_dims, in_channels = sample_in_channels, out_channels = sample_in_channels // 2, kernel_size=1, bias=False),
                        get_upsample_layer(spatial_dims=spatial_dims, in_channels=sample_in_channels // 2, upsample_mode=upsample_mode, scale_factor=2),
                    ]#get_upsample_layer is simply another wrapper for the Upsample. This function ensures that in_channels = out_channels
                    #scale_factor = 2 is the default value but i will write it out anyway
                )
            )
        return up_layers, up_samples

    def _make_final_conv(self, out_channels: int, in_channels:int):
        return nn.Sequential(
            # get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            # self.act_mod,
            # should i even use this for the final output?
            get_conv_layer(spatial_dims= self.spatial_dims, in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True),
        )
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)

        down_x = []

        for down in self.down_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x
    
    def decode(self, x: Tensor, down_x: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        feature_maps = []

        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)
            feature_maps.append(x)

        if self.use_conv_final:
            # Use separate conv layers for the last two feature maps
            processed_feature_maps = [
                self.conv_penultimate(feature_maps[-2]),  # Penultimate map
                self.conv_final(feature_maps[-1])         # Final map
            ]
        else:
            processed_feature_maps = feature_maps[-2:]

        return feature_maps, processed_feature_maps

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x, down_x = self.encode(x)
        down_x.reverse()

        feature_maps, processed_feature_maps = self.decode(x, down_x)
        
        # Return the final output (last processed map) and the penultimate map for deep supervision
        final_output = processed_feature_maps[-1]
        penultimate_output = processed_feature_maps[-2]
        if self.training:
            return final_output, penultimate_output
        else:
            return final_output



class DenseBCE(nn.Module):
    def __init__(self, class_weights=None, pos_weight=None, reduction='mean'):
        super(DenseBCE, self).__init__()
        self.class_weights = class_weights
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(x)
        
        # Compute BCE loss manually for more control
        eps = torch.tensor(1e-8).to(target.device) # For numerical stability
        try: 
            bce_loss = -(target * torch.log(probs + eps) + (1 - target) * torch.log(1 - probs + eps))
        except RuntimeError:
            breakpoint()
        
        # Apply positive weight if specified (similar to BCEWithLogitsLoss)
        if self.pos_weight is not None:
            bce_loss = target * self.pos_weight.to(bce_loss.device) * bce_loss + (1 - target) * bce_loss
        
        # Compute per-class losses
        class_losses = bce_loss.mean((0, 2, 3, 4))  # Average over batch and spatial dims
        
        # Apply class weights
        if self.class_weights is not None:
            weighted_loss = (class_losses * self.class_weights.to(class_losses.device)).sum()
        else:
            weighted_loss = class_losses.sum()
        
        return weighted_loss, class_losses



    
def create_binary_targets(coordinates, shape, sigma=2.0):
    """
    Create binary targets for coordinate localization
    coordinates: List of (z, y, x) coordinate tuples
    shape: (H, W, D) of the target volume
    Returns: (2, H, W, D) tensor with background/foreground probabilities
    """
    target = torch.zeros((2,) + shape)
    target[0] = 1.0  # Background class
    
    for coord in coordinates:
        z, y, x = coord
        if not (0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]):
            continue
            
        # Create local region around coordinate instead of full meshgrid
        radius = int(3 * sigma)  # 3-sigma rule
        z_min, z_max = int(max(0, z-radius)), int(min(shape[0], z+radius+1))
        y_min, y_max = int(max(0, y-radius)), int(min(shape[1], y+radius+1))
        x_min, x_max = int(max(0, x-radius)), int(min(shape[2], x+radius+1))
        zz, yy, xx = torch.meshgrid(torch.arange(z_min, z_max),
                                    torch.arange(y_min, y_max),
                                    torch.arange(x_min, x_max), indexing='ij')
        dist = torch.sqrt((zz-z)**2 + (yy-y)**2 + (xx-x)**2)
        foreground_prob = torch.exp(-dist**2 / (2*sigma**2))
        target[1, z_min:z_max, y_min:y_max, x_min:x_max] = torch.maximum(
            target[1, z_min:z_max, y_min:y_max, x_min:x_max], foreground_prob)
        
    target[0] = 1.0 - target[1]  # Update background
    return target


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


