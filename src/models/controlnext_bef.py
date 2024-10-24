# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import math
from torch import nn
from torch.nn import functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.resnet import Downsample2D, ResnetBlock2D
from einops import rearrange



logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class ControlNeXtOutput(BaseOutput):
    """
    The output of [`ControlNetModel`].

    Args:
        down_block_res_samples (`tuple[torch.Tensor]`):
            A tuple of downsample activations at different resolutions for each downsampling block. Each tensor should
            be of shape `(batch_size, channel * resolution, height //resolution, width // resolution)`. Output can be
            used to condition the original UNet's downsampling activations.
        mid_down_block_re_sample (`torch.Tensor`):
            The activation of the midde block (the lowest sample resolution). Each tensor should be of shape
            `(batch_size, channel * lowest_resolution, height // lowest_resolution, width // lowest_resolution)`.
            Output can be used to condition the original UNet's middle block activation.
    """

    down_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor


class Block2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
    ) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        output_states = ()

        for resnet in zip(self.resnets):
            hidden_states = resnet(hidden_states, temb)
            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states
    

class IdentityModule(nn.Module):
    def __init__(self):
        super(IdentityModule, self).__init__()

    def forward(self, *args):
        if len(args) > 0:
            return args[0]
        else:
            return None


class BasicBlock(nn.Module):
    def __init__(self, 
                 in_channels: int,
                out_channels: Optional[int] = None,
                stride=1,
                conv_shortcut: bool = False,
                dropout: float = 0.0,
                temb_channels: int = 512,
                groups: int = 32,
                groups_out: Optional[int] = None,
                pre_norm: bool = True,
                eps: float = 1e-6,
                non_linearity: str = "swish",
                skip_time_act: bool = False,
                time_embedding_norm: str = "default",  # default, scale_shift, ada_group, spatial
                kernel: Optional[torch.FloatTensor] = None,
                output_scale_factor: float = 1.0,
                use_in_shortcut: Optional[bool] = None,
                up: bool = False,
                down: bool = False,
                conv_shortcut_bias: bool = True,
                conv_2d_out_channels: Optional[int] = None,):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, 
                          out_channels, 
                          kernel_size=3 if stride != 1 else 1, 
                          stride=stride, 
                          padding=1 if stride != 1 else 0, 
                          bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, *args):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class Block2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            # in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                # ResnetBlock2D(
                #     in_channels=in_channels,
                #     out_channels=out_channels,
                #     temb_channels=temb_channels,
                #     eps=resnet_eps,
                #     groups=resnet_groups,
                #     dropout=dropout,
                #     time_embedding_norm=resnet_time_scale_shift,
                #     non_linearity=resnet_act_fn,
                #     output_scale_factor=output_scale_factor,
                #     pre_norm=resnet_pre_norm,
                BasicBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                ) if i == num_layers - 1 else \
                IdentityModule()
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    # Downsample2D(
                    #     out_channels,
                    #     use_conv=True,
                    #     out_channels=out_channels,
                    #     padding=downsample_padding,
                    #     name="op",
                    # )
                    BasicBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        temb_channels=temb_channels,
                        stride=2,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=resnet_pre_norm,
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
    ) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        output_states = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states
    



class ControlNeXtModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        time_embed_dim = 256,
        in_channels = [128, 128],
        out_channels = [128, 256],
        groups = [4, 8],
        controlnext_scale=1.
    ):
        super().__init__()

        self.time_proj = Timesteps(128, True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(128, time_embed_dim)
        self.embedding = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(2, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(2, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(2, 128),
            nn.ReLU(),
        )

        self.down_res = nn.ModuleList()
        self.down_sample = nn.ModuleList()
        for i in range(len(in_channels)):
            self.down_res.append(
                ResnetBlock2D(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    temb_channels=time_embed_dim,
                    groups=groups[i]
                ),
            )
            self.down_sample.append(
                Downsample2D(
                    out_channels[i],
                    use_conv=True,
                    out_channels=out_channels[i],
                    padding=1,
                    name="op",
                )
            )
        
        self.mid_convs = nn.ModuleList()
        self.mid_convs.append(nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels[-1],
                out_channels=out_channels[-1],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.GroupNorm(8, out_channels[-1]),
            nn.Conv2d(
                in_channels=out_channels[-1],
                out_channels=out_channels[-1],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.GroupNorm(8, out_channels[-1]),
        ))
        self.mid_convs.append(
            nn.Conv2d(
            in_channels=out_channels[-1],
            out_channels=320,
            kernel_size=1,
            stride=1,
        ))

        self.scale = controlnext_scale

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
    ):
        
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size = sample.shape[0]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        sample = self.embedding(sample)

        for res, downsample in zip(self.down_res, self.down_sample):
            sample = res(sample, emb)
            sample = downsample(sample, emb)
        
        sample = self.mid_convs[0](sample) + sample
        sample = self.mid_convs[1](sample)
        
        return {
            'output': sample,
            'scale': self.scale,
        }
    
