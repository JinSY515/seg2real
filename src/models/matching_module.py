import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv4d import Interpolate4d, Encoder4D

class OurModel(nn.Module):
    def __init__(
        self,
        feat_dim=[2,1]
    ):
        super().__init__()
        self.encoders = nn.ModuleList([])
        self.encoders.append(
            Encoder4D( # Encoder for conv_5
                        corr_levels=(feat_dim[0], 10, 10, feat_dim[1]),
                        kernel_size=(
                            (3, 3, 3, 3),
                            (3, 3, 3, 3),
                            (3, 3, 3, 3),
                        ),
                        stride=(
                            (1, 1, 1, 1),
                            (1, 1, 1, 1),
                            (1, 1, 1, 1),
                        ),
                        padding=(
                            (1, 1, 1, 1),
                            (1, 1, 1, 1),
                            (1, 1, 1, 1),
                        ),
                        group=(1,1,1),
                        residual=False
                    )
        )
        self.encoders.append(
            Encoder4D( # Encoder for conv_5
                        corr_levels=(feat_dim[1], 10, 10, feat_dim[1]),
                        kernel_size=(
                            (3, 3, 3, 3),
                            (3, 3, 3, 3),
                            (3, 3, 3, 3),
                        ),
                        stride=(
                            (1, 1, 1, 1),
                            (1, 1, 1, 1),
                            (1, 1, 1, 1),
                        ),
                        padding=(
                            (1, 1, 1, 1),
                            (1, 1, 1, 1),
                            (1, 1, 1, 1),
                        ),
                        group=(1,1,1),
                        residual=False
                    )
        )
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

    def forward(self, corr):
                
        corr = self.encoders[0](corr)[0] 
        corr = self.encoders[1](corr)[0]
        return corr
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.dtype = next(self.parameters()).dtype
        return self