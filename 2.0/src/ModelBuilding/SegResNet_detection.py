import torch
from torch import nn, Tensor
from transformers import PretrainedConfig

from .detection_head import ObjectDetectionHead, ObjectDetectionOutput
from monai.networks.nets import SegResNet
import torch.jit
from monai.networks.blocks import ResBlock
from monai.networks.blocks.segresnet_block import get_conv_layer, get_upsample_layer
from monai.networks.layers import get_norm_layer, get_act_layer, Dropout
from monai.utils import UpsampleMode
from typing import Optional, Dict, List, Union
from components.Eval_metrics import score
import torch.nn.functional as F
class SegResNet_Detection_Config(PretrainedConfig): #Do i really need this? I save as pt or onnx anyway 
    def __init__(
        self,
        spatial_dims=3,
        in_channels=1,
        init_filters=32,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.2,
        head_dropout_prob=0.1,
        class_weighting=[1.0,300.0],
        confidence_threshold=0.50,
        **kwargs,

    ):
        super().__init__(**kwargs)
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.init_filters = init_filters
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.head_dropout_prob = head_dropout_prob
        self.class_weighting = class_weighting
        self.confidence_threshold =  confidence_threshold


class SegResNetBackbone(nn.Module):
    """
    SegResNet based on `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.
    The module does not include the variational autoencoder (VAE).
    The model supports 2D or 3D inputs.

    Args:
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        init_filters: number of output channels for initial convolution layer. Defaults to 8.
        in_channels: number of input channels for the network. Defaults to 1.
        out_channels: number of output channels for the network. Defaults to 2.
        dropout_prob: probability of an element to be zero-ed. Defaults to ``None``.
        act: activation type and arguments. Defaults to ``RELU``.
        norm: feature normalization type and arguments. Defaults to ``GROUP``.
        norm_name: deprecating option for feature normalization type.
        num_groups: deprecating option for group norm. parameters.
        use_conv_final: if add a final convolution block to output. Defaults to ``True``.
        blocks_down: number of down sample blocks in each layer. Defaults to ``[1,2,2,4]``.
        blocks_up: number of up sample blocks in each layer. Defaults to ``[1,1,1]``.
        upsample_mode: [``"deconv"``, ``"nontrainable"``, ``"pixelshuffle"``]
            The mode of upsampling manipulations.
            Using the ``nontrainable`` modes cannot guarantee the model's reproducibility. Defaults to``nontrainable``.

            - ``deconv``, uses transposed convolution layers.
            - ``nontrainable``, uses non-trainable `linear` interpolation.
            - ``pixelshuffle``, uses :py:class:`monai.networks.blocks.SubpixelUpsample`.

    """

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
        self.conv_final = self._make_final_conv(out_channels)

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
                          #This operators unpacks the list of Resblock and nn.Sequential treat each of the element (each Resblock) as a separate parameter
            ) #this defines the number of Resblocks per layer. Because we defined 1,2,2,4
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

    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)
            ic(self.dropout(x).shape)

        down_x = []

        # for down in self.down_layers:
        #     ic(x.shape)
        #     x = down(x)
        #     down_x.append(x)

        # return x, down_x
    

        ic("Shape before down_layers")
        for layer, down in enumerate(self.down_layers):
            
            x = down(x)
            ic(f"downlayer {layer} output shape:", x.shape)
            down_x.append(x)

        return x, down_x

    def decode(self, x: Tensor, down_x: List[Tensor]) -> Tensor:
        feature_maps = []

        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)
            feature_maps.append(x)

        if self.use_conv_final:
            x = self.conv_final(x)

        return x, feature_maps

    def forward(self, x: Tensor) -> Tensor:
        print("Input shape:", x.shape)
        x, down_x = self.encode(x)
        down_x.reverse()

        x = self.decode(x, down_x)
        return x

class SegResNet_Detection_Model(nn.Module):


    def __init__(self, config: SegResNet_Detection_Config):
        super().__init__()
        self.config = config
        self.confidence_threshold = self.config.confidence_threshold
        self.class_weighting = self.config.class_weighting
        # Use the SegResNet backbone for feature extraction
        self.backbone = SegResNetBackbone(
            spatial_dims=self.config.spatial_dims,
            in_channels=self.config.in_channels,
            init_filters=self.config.init_filters,
            blocks_down=self.config.blocks_down,
            blocks_up=self.config.blocks_up,
            dropout_prob=self.config.dropout_prob,
        )


    def get_feature_size(self):
        # Calculate the feature size based on the backbone configuration
        # This is a placeholder - you'll need to calculate the actual size
        return self.config.init_filters * (2 ** (len(self.config.blocks_down) - 1))
    def forward(self, volume, labels=None):
        _, feature_maps = self.backbone(volume)
        
        if self.training and labels is not None:
            # Training: calculate loss
            target_shape = feature_maps[-1].shape[2:]
            target_block = torch.zeros(feature_maps[-1].shape[0], *target_shape, 
                                    dtype=torch.long, device=volume.device)
            
            for batch_idx, coord in enumerate(labels['coords']):  
                if coord is not None and coord[0] != -1:
                    z, y, x = coord
                    target_block[batch_idx, z, y, x] = 1
             
            criterion = nn.CrossEntropyLoss(weight=self.class_weighting.to(volume.device)) 
            #NEED A CUSTOM DENSE CE LOSS. SINCE WE ARE USING MIXUP WE NEED IT TO ACCEPT SOFT LABELS
            losses = []
            
            for i, feature_map in enumerate(feature_maps[-3:-1]):
                upsampled = F.interpolate(feature_map, size=target_shape, 
                                        mode='trilinear', align_corners=False)
                losses.append(criterion(upsampled, target_block) * (0.5 ** i))
            
            return {"loss": sum(losses)}

        else:
            # Inference: get predictions and coordinates
            final_predictions = feature_maps[-1]
            probabilities = F.softmax(final_predictions, dim=1)
            
            # Extract coordinates from motor class (class 1)
            motor_probs = probabilities[:, 1]  # Get motor class probabilities
            
            coordinates = []
            for batch_idx in range(motor_probs.shape[0]):
                max_prob = motor_probs[batch_idx].max()
                if max_prob > self.confidence_threshold:  # Confidence threshold
                    # Get 3D coordinates of maximum
                    max_idx = motor_probs[batch_idx].argmax() #max_idx is a scalar for the position of the coordinate in a flattened map
                    d, h, w = motor_probs[batch_idx].shape
                    z = max_idx // (h * w)
                    y = (max_idx % (h * w)) // w
                    x = max_idx % w
                    coordinates.append((z.item(), y.item(), x.item()))
                else:
                    coordinates.append((-1,-1,-1))  # No motor detected
            
            return {"probabilities": probabilities, "coordinates": coordinates}



















    
    # def forward(self, volume, labels=None):
    #     # Extract features using the backbone
    #     _, feature_maps = self.backbone(volume)
        
    #     # Get the final feature map
    #     features = feature_maps[-1]
        
    #     # Apply pooling and dropout
    #     pooled = self.pool(features)
    #     pooled = self.dropout(pooled)
    #     flattened = self.flatten(pooled)
        
    #     # Predict if a motor is present (0 to 1)
    #     ic(self.get_feature_size())
    #     ic(flattened.shape)
    #     has_motor = self.classifier(flattened)
    #     has_motor = has_motor.squeeze(1)
    #     ic(has_motor)
    #     # Predict coordinates
    #     coordinates = self.regressor(flattened)
        
    #     # During inference (not training), apply threshold logic
    #     if not self.training:
    #         # If has_motor < threshold (e.g., 0.5), set coordinates to [-1, -1, -1]
    #         threshold = 0.5
    #         no_motor_mask = (has_motor < threshold).view(-1, 1)
    #         no_motor_coords = torch.tensor(
    #             [-1, -1, -1], 
    #             device=coordinates.device
    #         ).expand_as(coordinates)
            
    #         coordinates = torch.where(
    #             no_motor_mask, 
    #             no_motor_coords,
    #             coordinates
    #         )
        
    #     # Calculate loss if labels are provided (during training)
    #     loss = None
    #     loss_dict = None
        
    #     if labels is not None:
    #         cls_loss = nn.BCELoss()(has_motor, labels["has_motor"])
            
    #         # Mean squared error for coordinate prediction
    #         # Only calculate for samples that have motors
    #         motor_mask = labels["has_motor"] > 0.5
            
    #         if motor_mask.sum() > 0:
    #             # MSE loss for samples with motors
    #             coord_loss = nn.MSELoss()(
    #                 coordinates[motor_mask], 
    #                 labels["coords"][motor_mask]
    #             )
    #         else:
    #             # No motors in this batch
    #             coord_loss = torch.tensor(0.0, device=coordinates.device)
            
    #         # Total loss is a combination of both
    #         loss = cls_loss + coord_loss
    #         loss_dict = {
    #             "cls_loss": cls_loss, 
    #             "coord_loss": coord_loss
    #         }
        
    #     # Return predictions and loss
    #     return {
    #         "has_motor": has_motor,
    #         "coordinates": coordinates,
    #         "loss": loss,
    #         "loss_dict": loss_dict
    #     }