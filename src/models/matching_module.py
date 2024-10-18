import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv4d import Interpolate4d, Encoder4D

class OurModel(nn.Module):
    def __init__(
        self,
        feat_dim=1
    ):
        super().__init__()

        self.encoders = Encoder4D( # Encoder for conv_5
                        corr_levels=(feat_dim, 10, 10, feat_dim),
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

    def forward(self, corr):
                
        corr = self.encoders(corr)[0] #+ corr
        
        return corr
    
    def to(self, *args, **kwargs):
        # 모델의 모든 파라미터와 텐서를 장치 및 dtype으로 이동
        super().to(*args, **kwargs)
        self.dtype = next(self.parameters()).dtype
        return self