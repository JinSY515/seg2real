from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from diffusers.models.modeling_utils import ModelMixin

from src.models.motion_module import zero_module
from src.models.resnet import InflatedConv3d
import torch

class PoseGuider(ModelMixin):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        condition_num : int = 1,
        block_out_channels: Tuple[int] = (16, 32, 64, 128),
    ):
        super().__init__()
        self.condition_num = condition_num
        self.channel_in = nn.Conv3d(
            in_channels=condition_num * 3, out_channels=3, kernel_size=3, padding=1, # kernel_size=(2, 1, 1), stride=(2,1,1), padding=(0,0,0)
        )
        self.conv_in = InflatedConv3d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.blocks.append(
                InflatedConv3d(
                    channel_in, channel_out, kernel_size=3, padding=1, stride=2
                )
            )

        self.conv_out = zero_module(
            InflatedConv3d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )

    def forward(self, conditioning):
        # embedding = self.conv_in(conditioning)
        if self.condition_num != 1:
            embedding = self.channel_in(conditioning)
            embedding = self.conv_in(embedding)
        else: 
            embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding
    
class RefGuider(ModelMixin):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 64, 128),
    ):
        super().__init__()
        self.conv_in = InflatedConv3d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                nn.Sequential(
                    InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1),
                    nn.BatchNorm3d(channel_in),  # BatchNorm 추가
                    nn.SiLU(),  # Activation 추가
                )
            )
            self.blocks.append(
                nn.Sequential(
                    InflatedConv3d(
                        channel_in, channel_out, kernel_size=3, padding=1, stride=2
                    ),
                    nn.BatchNorm3d(channel_out),  # BatchNorm 추가
                    nn.SiLU(),  # Activation 추가
                )
            )

        self.conv_out = (
            InflatedConv3d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )

    def forward(self, conditioning):
        
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            # embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

         # KL divergence 적용
        mu = embedding.mean(dim=(2, 3, 4))
        log_var = embedding.var(dim=(2, 3, 4), unbiased=False).log()
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return embedding, kl_loss  # KL Loss 반환
    

