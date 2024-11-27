import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv4d import Interpolate4d, Encoder4D

class OurModel(nn.Module):
    def __init__(
        self,
        feat_dim=[9,8],
        time_embed_dim=9,
    ):
        super().__init__()
        self.encoders = nn.ModuleList([])
        self.encoders.append(
            Encoder4D( # Encoder for conv_5 채널도 늘리고 레이어 더 달기 (무겁게 가도될듯)
                        corr_levels=(feat_dim[0], 16,  32, 16, feat_dim[1]),
                        kernel_size=(
                            (1, 1, 3, 3), # (1, 1, 3, 3)으로도 돌려보기? 
                            (1, 1, 3, 3),
                            (1, 1, 3, 3),
                            (1, 1, 3, 3),
                            # (1, 1, 3, 3), 
                        ),
                        stride=(
                            (1, 1, 1, 1),
                            (1, 1, 1, 1),
                            (1, 1, 1, 1),
                            (1, 1, 1, 1),
                            # (1, 1, 1, 1),
                        ),
                        padding=(
                            (0, 0, 1, 1),
                            (0, 0, 1, 1),
                            (0, 0, 1, 1),
                            (0, 0, 1, 1),
                            # (0, 0, 1, 1),
                        ),
                        group=(1,1,1,1),
                        residual=False
                    )
        )
        # self.time_embedding = nn.Linear(1, time_embed_dim)
        # self.time_embed_dim = time_embed_dim
        # self.encoders.append(
        #     Encoder4D( # Encoder for conv_5
        #                 corr_levels=(feat_dim[1], 10, 10, feat_dim[1]),
        #                 kernel_size=(
        #                     (3, 3, 3, 3),
        #                     (3, 3, 3, 3),
        #                     (3, 3, 3, 3),
        #                 ),
        #                 stride=(
        #                     (1, 1, 1, 1),
        #                     (1, 1, 1, 1),
        #                     (1, 1, 1, 1),
        #                 ),
        #                 padding=(
        #                     (1, 1, 1, 1),
        #                     (1, 1, 1, 1),
        #                     (1, 1, 1, 1),
        #                 ),
        #                 group=(1,1,1),
        #                 residual=False
        #             )
        # )
        # self.encoders = Encoder4D( # Encoder for conv_5
        #                 corr_levels=(feat_dim[0], 10, 10, feat_dim[1]),
        #                 kernel_size=(
        #                     (3, 3, 3, 3),
        #                     (3, 3, 3, 3),
        #                     (3, 3, 3, 3),
        #                 ),
        #                 stride=(
        #                     (1, 1, 1, 1),
        #                     (1, 1, 1, 1),
        #                     (1, 1, 1, 1),
        #                 ),
        #                 padding=(
        #                     (1, 1, 1, 1),
        #                     (1, 1, 1, 1),
        #                     (1, 1, 1, 1),
        #                 ),
        #                 group=(1,1,1),
        #                 residual=False
        #             )

    def forward(self, corr, t=None):
        # t = t.view(-1, 1).float()
        # time_embed = self.time_embedding(t)
        # time_embed = time_embed.view(1, self.time_embed_dim, 1, 1, 1, 1)
        # corr = corr + time_embed
        corr = self.encoders[0](corr)[0] 
        # print("corr", corr)
        # corr = self.encoders[1](corr)[0]
        return corr
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.dtype = next(self.parameters()).dtype
        return self