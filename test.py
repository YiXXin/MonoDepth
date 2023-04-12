from networks import bi_encoder
from networks import swin_transformer
import torch

from ipdb import set_trace
set_trace()
image = torch.randn(6, 3, 192, 640)
model = swin_transformer.SwinTransformer()
# model = bi_encoder.biformer_tiny()