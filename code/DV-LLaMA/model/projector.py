import os
import re

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.regnet import RegStage
from timm.models.layers import LayerNorm, LayerNorm2d
from transformers import TRANSFORMERS_CACHE


def parse_snapshot_folder(repo_id, cache_dir=None, repo_type="model"):
    revision = "main"
    # 1. parse the downloaded cache folder
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    else:
        cache_dir = cache_dir
    object_id = repo_id.replace("/", "--")
    repo_cache = os.path.join(cache_dir, f"{repo_type}s--{object_id}")
    # 2. resolve refs (for instance to convert main to the associated commit sha)
    refs_dir = os.path.join(repo_cache, "refs")
    if os.path.isdir(refs_dir):
        revision_file = os.path.join(refs_dir, revision)
        if os.path.isfile(revision_file):
            with open(revision_file) as f:
                revision = f.read()
    # 3. acquire the snapshot folder
    folder = os.path.join(repo_cache, "snapshots", revision)

    return folder


def load_mm_projector(model_path, cache_dir=None, token=None):
    if os.path.exists(os.path.join(model_path, 'dv_projector.bin')):
        is_local = True
        folder = model_path
    else:
        is_local = False
        folder = parse_snapshot_folder(model_path, cache_dir=cache_dir, repo_type="model")
        if not os.path.exists(os.path.join(folder, 'dv_projector.bin')):  
            # downloading from remote repo
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_path, cache_dir=cache_dir, token=token)

    projector_weights = torch.load(os.path.join(folder, 'dv_projector.bin'), map_location='cpu')  
    projector_weights = {k: v.to(torch.float16) for k, v in projector_weights.items()}
    return projector_weights


class IdentityMap(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class DynamicAdapterBlock(nn.Module):
    """Dynamic adapter block with configurable expansion ratio and activation."""
    
    def __init__(self, channels, expansion_ratio=4, activation=nn.GELU):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)
        self.adapter_dim = int(channels // expansion_ratio)
        
        self.adapter = nn.Sequential(
            nn.Linear(channels, self.adapter_dim),
            activation(),
            nn.Linear(self.adapter_dim, channels)
        )
        
    def forward(self, x):
        residual = x
        x = self.pre_norm(x)
        x = residual + self.adapter(x)
        return x


def build_vision_projector(config, delay_load=False, **kwargs):
    """Builds the vision-language projector with DVLLaMA-specific enhancements."""
    
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    
    elif projector_type == "dynamic_adapter": 
        return nn.Sequential(
            nn.Linear(config.mm_hidden_size, config.hidden_size),
            DynamicAdapterBlock(
                config.hidden_size, 
                expansion_ratio=config.adapter_expansion_ratio,
                activation=getattr(nn, config.adapter_activation)
            )
        )
    
    elif projector_type == "stc_connector":
        return STCConnector(config)
    
    elif projector_type == "stp_connector":
        return STPConnector(config)
    
    elif projector_type == "stc_connector_v35":
        return STCConnectorV35(config)
    
    elif projector_type == "spatial_conv":
        return SpatialConv(config)
    
    elif projector_type == "spatial_pool":
        return SpatialPool(config)
    
    elif projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


class STCConnector(nn.Module):
    """Spatio-Temporal Convolutional Connector with enhancements for DVLLaMA."""

    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        """Temporal Convolutional Vision-Language Connector.
        
        Args:
            config: config object.
            downsample: (temporal, height, width) downsample rate.
            depth: depth of the spatial interaction blocks.
            mlp_depth: depth of the vision-language projector layers.
        """
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size = config.mm_hidden_size
        self.hidden_size = hidden_size = config.hidden_size
        self.output_hidden_size = output_hidden_size = config.hidden_size
        self.depth = depth
        self.mlp_depth = mlp_depth
        self.downsample = downsample
        
        # 支持动态适配器配置
        if depth != 0:
            if config.dynamic_adapter_type == "parallel":
                self.s1 = RegStage(
                    depth=depth,
                    in_chs=encoder_hidden_size,
                    out_chs=hidden_size,
                    stride=1,
                    dilation=1,
                    act_layer=nn.SiLU,
                    norm_layer=LayerNorm2d,
                )
            else:
                self.s1 = nn.Sequential(
                    LayerNorm2d(encoder_hidden_size),
                    nn.Conv2d(encoder_hidden_size, hidden_size, kernel_size=1),
                    nn.SiLU()
                )
        else:
            self.s1 = nn.Identity()
            
        # 3D采样器支持动态配置
        self.sampler = nn.Sequential(
            nn.Conv3d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=downsample,
                stride=downsample,
                padding=tuple([s//2 for s in downsample]),  # 调整padding以保持兼容性
                bias=True
            ),
            nn.SiLU()
        )
        
        if depth != 0:
            self.s2 = RegStage(
                depth=depth,
                in_chs=hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
        else:
            self.s2 = nn.Identity()
            
        # 支持动态MLP配置
        self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)

    def forward(self, x):
        """Aggregate tokens on the temporal and spatial dimensions.
        Args:
            x: input tokens [b, t, h, w, d] / [b, t, l, d]
        Returns:
            aggregated tokens [b, l, d]
        """
        t = x.size(1)
        if x.ndim == 4:
            hw = int(x.size(2) ** 0.5)
            x = einops.rearrange(x, "b t (h w) d -> b d t h w", h=hw, w=hw)
        elif x.ndim == 5:
            x = einops.rearrange(x, "b t h w d -> b d t h w")

        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        # 1. the first stage of the adapter
        x = self.s1(x)
        x = einops.rearrange(x, "(b t) d h w -> b d t h w", t=t)
        # 2. downsampler
        x = self.sampler(x)
        new_t = x.size(2)
        # 3. the second stage of the adapter
        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        x = self.s2(x)
        x = einops.rearrange(x, "(b t) d h w -> b (t h w) d", t=new_t)
        x = self.readout(x)
        return x


class STPConnector(STCConnector):
    """Spatio-Temporal Pooling Connector with enhancements for DVLLaMA."""

    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        super().__init__(config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth)
        self.sampler = nn.Sequential(nn.AvgPool3d(downsample), nn.SiLU())


class STCConnectorV35(STCConnector):
    """Enhanced Spatio-Temporal Convolutional Connector for DVLLaMA v3.5."""

    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        super().__init__(config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth)
        self.sampler = nn.Sequential(
            nn.Conv3d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=downsample,
                stride=downsample,
                padding=0,
                bias=True
            ),
            nn.SiLU()
        )


class SpatialConv(STCConnector):
    """Spatial Convolution-based Connector for DVLLaMA."""

    def __init__(self, config, downsample=(1, 2, 2), depth=0, mlp_depth=2):
        super().__init__(config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth)


class SpatialPool(STPConnector):
    """Spatial Pooling-based Connector for DVLLaMA."""

    def __init__(self, config, downsample=(1, 2, 2), depth=0, mlp_depth=2):
        super().__init__(config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth)


class DynamicTemporalAdapter(nn.Module):
    """Dynamic Temporal Adapter for handling video sequences in DVLLaMA."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.temporal_kernel_size = config.temporal_kernel_size
        
        self.temporal_conv = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=self.temporal_kernel_size,
            stride=1,
            padding=self.temporal_kernel_size // 2,
            groups=self.hidden_size if config.depthwise_temporal_conv else 1
        )
        
        self.norm = nn.LayerNorm(self.hidden_size)
        self.activation = nn.GELU()
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
        Returns:
            [batch_size, seq_len, hidden_size]
        """
        # 交换维度以适应Conv1d
        x_transposed = x.transpose(1, 2)  # [batch_size, hidden_size, seq_len]
        
        # 应用时间卷积
        temporal_features = self.temporal_conv(x_transposed)
        temporal_features = temporal_features.transpose(1, 2)  # [batch_size, seq_len, hidden_size]
        
        # 残差连接和归一化
        output = self.norm(x + temporal_features)
        output = self.activation(output)
        
        return output