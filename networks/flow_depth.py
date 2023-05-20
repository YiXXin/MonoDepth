import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import torchvision as tv

from .utils.warp_utils import flow_warp
from .correlation_native import Correlation

def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True)
        )

def spatial_flatten(x):
  return torch.reshape(x, [-1, x.shape[1] * x.shape[2], x.shape[-1]])

def unstack_and_split(x, batch_size, num_channels=3):   # x: [20, 4, 128, 224]
    """Unstack batch dimension and split into channels and alpha mask."""
    unstacked = einops.rearrange(x, '(b s) c h w -> b s c h w', b=batch_size)   # [4, 5, 4, 128, 224]
    channels, masks = torch.split(unstacked, [num_channels, 1], dim=2)   # [4, 5, 3, 128, 224], [4, 5, 1, 128, 224]
    return channels, masks


def spatial_broadcast(slots, resolution):
    """Broadcast slot features to a 2D grid and collapse slot dimension."""
    # `slots` has shape: [batch_size, num_slots, slot_size].
    slots = torch.reshape(slots, [-1, slots.shape[-1]])[:, None, None, :]
    grid = einops.repeat(slots, 'b_n i j d -> b_n (tilei i) (tilej j) d', tilei=resolution[0], tilej=resolution[1])
    # `grid` has shape: [batch_size*num_slots, height, width, slot_size].
    return grid

def conv_recover(in_planes, out_planes, kernel_size, stride, dilation = 1, isReLU=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True),
            # nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,dilation=dilation,
            #           padding='same', bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True),
            nn.Identity()
        )

def conv_resize(inputs, size):
    return F.interpolate(inputs, size=size, mode='bilinear', align_corners=False)
    # deconv = conv_recover(in_planes, out_planes, kernel_size, stride, dilation = dilation)
    # return deconv

class Discriminator(nn.Module):
    # can reduce 25% of training time.
    def __init__(self, img, flow_masked, mask, f=0.25):
        super(Discriminator, self).__init__()
        batch_size = img.shape[0]
        C = flow_masked.shape[1]  # 2
        
        self.ones_x = torch.ones_like(flow_masked)[:, 0:1, :, :]  # [16,1,192,384]
        # Augmentation of the flow
        self.flow_masked_resize = torch.cat([flow_masked, self.ones_x, 1.0-mask], dim=1)  # [16,4,192,384]
        self.flow_in_channels = self.flow_masked_resize.shape[1]  # 4

        self.aconv1 = conv_recover(96,int(64*f), kernel_size=7, stride=2, dilation=1,isReLU=True)
        self.aconv2 = conv_recover(int(64*f),int(128*f), kernel_size=5, stride=2, dilation=1,isReLU=True)
        self.aconv3 = conv_recover(int(128*f),int(256*f), kernel_size=5, stride=2, dilation=1,isReLU=True)
        self.aconv31 = conv_recover(int(256*f),int(256*f), kernel_size=3, stride=1, dilation=1,isReLU=True)
        self.aconv4 = conv_recover(int(256*f),int(512*f), kernel_size=3, stride=2, dilation=1,isReLU=True)
        self.aconv41 = conv_recover(int(512*f),int(512*f), kernel_size=3, stride=1, dilation=1,isReLU=True)
        self.aconv5 = conv_recover(int(512*f),int(512*f), kernel_size=3, stride=2, dilation=1,isReLU=True)
        self.aconv51 = conv_recover(int(512*f),int(512*f), kernel_size=3, stride=1, dilation=1,isReLU=True)
        self.aconv6 = conv_recover(int(512*f),int(512*f), kernel_size=3, stride=2, dilation=1,isReLU=True)

        self.bconv1 = conv_recover(self.flow_in_channels,int(64*f), kernel_size=7, stride=2, dilation=1,isReLU=True)
        self.bconv2 = conv_recover(int(64*f),int(128*f), kernel_size=5, stride=2, dilation=1, isReLU=True)
        self.bconv3 = conv_recover(int(128*f),int(256*f), kernel_size=5, stride=2, dilation=1, isReLU=True)
        self.bconv31 = conv_recover(int(256*f),int(256*f), kernel_size=3, stride=1, dilation=1, isReLU=True)
        self.bconv4 = conv_recover(int(256*f),int(512*f), kernel_size=3, stride=2, dilation=1,isReLU=True)
        self.bconv41 = conv_recover(int(512*f),int(512*f), kernel_size=3, stride=1, dilation=1,isReLU=True)
        self.bconv5 = conv_recover(int(512*f),int(512*f), kernel_size=3, stride=2, dilation=1,isReLU=True)
        self.bconv51 = conv_recover(int(512*f),int(512*f), kernel_size=3, stride=1, dilation=1,isReLU=True)
        self.bconv6 = conv_recover(int(512*f),int(512*f), kernel_size=3, stride=2, dilation=1,isReLU=True)

        self.deconv5 = conv_recover(int(512*2*f),int(512*f),kernel_size=3,stride=1,isReLU=True)   #ATTN: kernel_size 4->3
        
        self.flow5 = conv_recover(int(512*3*f), C, kernel_size=3, stride=1, isReLU=False)  # activation=tf.identity ?
        self.deconv4 = conv_recover(int(512*3*f),int(512*f),kernel_size=3,stride=1,isReLU=True)   #ATTN: kernel_size 4->3
        self.upflow4 =conv_recover(C, C, kernel_size=3, stride=1,isReLU=False)   #ATTN: kernel_size 4->3

        self.flow4 = conv_recover(int(512*3*f+C), C, kernel_size=3, stride=1, isReLU=False)
        self.deconv3 = conv_recover(int(512*3*f+C),int(256*f),kernel_size=3,stride=1,isReLU=True)   #ATTN: kernel_size 4->3
        self.upflow3 =conv_recover(C, C, kernel_size=3, stride=1,isReLU=False)   #ATTN: kernel_size 4->3

        self.flow3 = conv_recover(int(256*3*f+C), C, kernel_size=3, stride=1,isReLU=False)
        self.deconv2 = conv_recover(int(256*3*f+C),int(128*f),kernel_size=3,stride=1,isReLU=True)   #ATTN: kernel_size 4->3
        self.upflow2 =conv_recover(C, C, kernel_size=3, stride=1,isReLU=False)   #ATTN: kernel_size 4->3

        self.flow2 = conv_recover(int(128*3*f+C), C, kernel_size=3, stride=1,isReLU=False)
        self.deconv1 = conv_recover(int(128*3*f+C),int(64*f),kernel_size=3,stride=1,isReLU=True)   #ATTN: kernel_size 4->3
        self.upflow1 =conv_recover(C, C, kernel_size=3, stride=1,isReLU=False)   #ATTN: kernel_size 4->3

        self.flow1 = conv_recover(int(64*3*f+C), C, kernel_size=5, stride=1,isReLU=False)

    def forward(self, img):       
        orisize = img.shape # [16,3,192,384]  [4, 128, 48, 104]

        # from ipdb import set_trace
        # set_trace()

        a1 = self.aconv1(img) # [16, 16, 96, 192]  [4, 16, 24, 52]
        a2 = self.aconv2(a1) # [16, 32, 48, 96] [4, 32, 12, 26]
        a3 = self.aconv3(a2) # [16, 64, 24, 48] [4, 64, 6, 13]
        a31 = self.aconv31(a3) # [16, 64, 24, 48] [4, 64, 6, 13]
        a4 = self.aconv4(a31) # [16, 128, 12, 24] [4, 128, 3, 7]
        a41 = self.aconv41(a4) # [16, 128, 12, 24] [4, 128, 3, 7]
        a5 = self.aconv5(a41) # [16, 128, 6, 12] [4, 128, 2, 4]
        a51 = self.aconv51(a5) # [16, 128, 6, 12] [4, 128, 2, 4]
        a6 = self.aconv6(a51) # [16, 128, 3, 6] [4, 128, 1, 2]
        

        b1 = self.bconv1(self.flow_masked_resize) # [16, 16, 96, 192] [4, 16, 24, 52]
        b2 = self.bconv2(b1) # [16, 32, 48, 96] [4, 32, 12, 26]
        b3 = self.bconv3(b2) # [16, 64, 24, 48] [4, 64, 6, 13]
        b31 = self.bconv31(b3) # [16, 64, 24, 48] [4, 64, 6, 13]
        b4 = self.bconv4(b31) # [16, 128, 12, 24] [4, 128, 3, 7]
        b41 = self.bconv41(b4) # [16, 128, 12, 24] [4, 128, 3, 7]
        b5 = self.bconv5(b41) # [16, 128, 6, 12] [4, 128, 2, 4]
        b51 = self.bconv51(b5) # [16, 128, 6, 12] [4, 128, 2, 4]
        b6 = self.bconv6(b51) # [16, 128, 3, 6] [4, 128, 1, 2]

        conv6 = conv_resize(torch.cat([a6, b6], dim=1),(b51.shape[2],b51.shape[3]))  # [16, 256, 6, 12] [4, 256, 2, 4]
        d5 = self.deconv5(conv6)  # [16, 128, 6, 12] [4, 128, 2, 4]
        concat5 = torch.cat([d5,b5,a51],dim=1)  # [16, 384, 6, 12] [4, 384, 2, 4]

        f5 = self.flow5(concat5) # [16, 2, 6, 12] [4, 2, 2, 4]
        conv5 = conv_resize(concat5,(b41.shape[2],b41.shape[3])) # [16, 384, 12, 24] [4, 384, 3, 7]
        d4 = self.deconv4(conv5)  # [16, 128, 12, 24] [4, 128, 3, 7]
        f5_resize = conv_resize(f5,(b41.shape[2],b41.shape[3]))  # [16, 2, 12, 24] [4, 2, 3, 7]
        uf4 = self.upflow4(f5_resize)  # [16, 2, 12, 24] [4, 2, 3, 7]
        concat4 = torch.cat([d4,b41,a41,uf4],dim=1)  # [16, 386, 12, 24] [4, 386, 3, 7]
        
        f4 = self.flow4(concat4) # [16, 2, 12, 24] [4, 2, 3, 7]
        conv4 = conv_resize(concat4,(b31.shape[2],b31.shape[3])) # [16, 386, 24, 48] [4, 386, 6, 13]
        d3 = self.deconv3(conv4)  # [16, 64, 24, 48] [4, 64, 6, 13]
        f4_resize = conv_resize(f4,(b31.shape[2],b31.shape[3]))  # [16, 2, 24, 48] [4, 2, 6, 13]
        uf3 = self.upflow3(f4_resize)  # [16, 2, 24, 48] [4, 2, 6, 13]
        concat3 = torch.cat([d3,b31,a31,uf3],dim=1)  # [16, 194, 24, 48] [4, 194, 6, 13]

        f3 = self.flow3(concat3) # [16, 2, 24, 48] [4, 2, 6, 13]
        conv3 = conv_resize(concat3,(b2.shape[2],b2.shape[3])) # [16, 194, 48, 96] [4, 194, 12, 26]
        d2 = self.deconv2(conv3)  # [16, 32, 48, 96] [4, 32, 12, 26]
        f3_resize = conv_resize(f3,(b2.shape[2],b2.shape[3]))  # [16, 2, 48, 96] [4, 2, 12, 26]
        uf2 = self.upflow2(f3_resize)  # [16, 2, 48, 96] [4, 2, 12, 26]
        concat2 = torch.cat([d2,b2,a2,uf2],dim=1)  # [16, 98, 48, 96] [4, 98, 12, 26]

        f2 = self.flow2(concat2) # [16, 2, 48, 96] [4, 2, 12, 26]
        conv2 = conv_resize(concat2,(b1.shape[2],b1.shape[3])) # [16, 98, 96, 192] [4, 98, 24, 52]
        d1 = self.deconv1(conv2)  # [16, 16, 96, 192] [4, 16, 24, 52]
        f2_resize = conv_resize(f2,(b1.shape[2],b1.shape[3]))  # [16, 2, 96, 192] [4, 2, 24, 52]
        uf1 = self.upflow1(f2_resize)  # [16, 2, 96, 192] [4, 2, 24, 52]
        concat1 = torch.cat([d1,b1,a1,uf1],dim=1)  # [16, 50, 96, 192] [4, 50, 24, 52]

        f1 = self.flow1(concat1)  # [16, 2, 96, 192] [4, 2, 24, 52]
        pred_flow = conv_resize(f1,(orisize[2],orisize[3]))  # [16, 2, 192, 384] [4, 2, 48, 104]

        return pred_flow


class FeatureExtractor(nn.Module):
    def __init__(self, num_chs):
        super(FeatureExtractor, self).__init__()
        self.num_chs = num_chs
        self.convs = nn.ModuleList()

        for l, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
            layer = nn.Sequential(
                conv(ch_in, ch_out, stride=2),
                conv(ch_out, ch_out)
            )
            self.convs.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv in self.convs:
            x = conv(x)
            feature_pyramid.append(x)

        return feature_pyramid[::-1]

class FlowEstimatorDense(nn.Module):
    def __init__(self, ch_in):
        super(FlowEstimatorDense, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(ch_in + 128, 128)
        self.conv3 = conv(ch_in + 256, 96)
        self.conv4 = conv(ch_in + 352, 64)
        self.conv5 = conv(ch_in + 416, 32)
        self.feat_dim = ch_in + 448
        self.conv_last = conv(ch_in + 448, 2, isReLU=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        return x5, x_out


class FlowEstimatorReduce(nn.Module):
    # can reduce 25% of training time.
    def __init__(self, ch_in):
        super(FlowEstimatorReduce, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(128, 128)
        self.conv3 = conv(128 + 128, 96)
        self.conv4 = conv(128 + 96, 64)
        self.conv5 = conv(96 + 64, 32)
        self.feat_dim = 32
        self.predict_flow = conv(64 + 32, 2, isReLU=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x1, x2], dim=1))
        x4 = self.conv4(torch.cat([x2, x3], dim=1))
        x5 = self.conv5(torch.cat([x3, x4], dim=1))
        flow = self.predict_flow(torch.cat([x4, x5], dim=1))
        return x5, flow


class ContextNetwork(nn.Module):
    def __init__(self, ch_in):
        super(ContextNetwork, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 2),
            conv(128, 128, 3, 1, 4),
            conv(128, 96, 3, 1, 8),
            conv(96, 64, 3, 1, 16),
            conv(64, 32, 3, 1, 1),
            conv(32, 2, isReLU=False)
        )

    def forward(self, x):
        return self.convs(x)

class SlotAttention(nn.Module):
    """Slot Attention module."""

    def __init__(self, num_slots, encoder_dims, slots_iters=1, hidden_dim=128, eps=1e-8):
        """Builds the Slot Attention module.
        Args:
            slots_iters: Number of iterations.
            num_slots: Number of slots.
            encoder_dims: Dimensionality of slot feature vectors.
            hidden_dim: Hidden layer size of MLP.
            eps: Offset for attention coefficients before normalization.
        """
        super(SlotAttention, self).__init__()

        self.eps = eps
        self.slots_iters = slots_iters
        self.num_slots = num_slots
        self.scale = encoder_dims ** -0.5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.norm_input = nn.LayerNorm(encoder_dims)
        self.norm_slots = nn.LayerNorm(encoder_dims)
        self.norm_pre_ff = nn.LayerNorm(encoder_dims)

        # Parameters for Gaussian init (shared by all slots).
        # self.slots_mu = nn.Parameter(torch.randn(1, 1, encoder_dims))
        # self.slots_sigma = nn.Parameter(torch.randn(1, 1, encoder_dims))

        self.slots_embedding = nn.Embedding(num_slots, encoder_dims)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(encoder_dims, encoder_dims)
        self.project_k = nn.Linear(encoder_dims, encoder_dims)
        self.project_v = nn.Linear(encoder_dims, encoder_dims)

        # Slot update functions.
        self.gru = nn.GRUCell(encoder_dims, encoder_dims)

        hidden_dim = max(encoder_dims, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dims, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, encoder_dims)
        )

    def forward(self, inputs, num_slots=None):  # [4,78,192]
        # from ipdb import set_trace
        # set_trace()
        # inputs has shape [batch_size, num_inputs, inputs_size].
        inputs = self.norm_input(inputs)  # Apply layer norm to the input.   [4,78,192]
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].  [4,78,192]
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].  [4,78,192]

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots   # 128

        # random slots initialization,
        # mu = self.slots_mu.expand(b, n_s, -1)
        # sigma = self.slots_sigma.expand(b, n_s, -1)
        # slots = torch.normal(mu, sigma)

        # learnable slots initialization
        slots = self.slots_embedding(torch.arange(0, n_s).expand(b, n_s).to(self.device))  # [4,128,192]

        # Multiple rounds of attention.
        for _ in range(self.slots_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].  [4,128,192]
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale   # [4,128,78]
            attn = dots.softmax(dim=1) + self.eps   # [4,128,78]
            attn = attn / attn.sum(dim=-1, keepdim=True)  # weighted mean.   [4,128,78]

            updates = torch.einsum('bjd,bij->bid', v, attn)  
            # `updates` has shape: [batch_size, num_slots, slot_size]. [4,128,192]

            # Slot update.
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )   # [512,192]
            slots = slots.reshape(b, -1, d)   # [4,128,192]
            slots = slots + self.mlp(self.norm_pre_ff(slots))  # [4,128,192]

        return slots, attn

class PWCFlow(nn.Module):
    def __init__(self, cfg):
        super(PWCFlow, self).__init__()
        self.cfg =cfg
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 192]
        self.output_level = 2
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        # self.resnet18_model = resnet18Flow()
        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)

        self.upsample = self.cfg.upsample
        self.n_frames = self.cfg.n_frames
        self.reduce_dense = self.cfg.reduce_dense

        self.slots_iters = self.cfg.slots_iters   
        self.num_slots = self.cfg.num_slots
        self.in_out_channels = self.cfg.in_out_channels

        # from ipdb import set_trace
        # set_trace()

        self.encoder_dims = [192, 128, 96]
        # self.encoder_dims = [512, 256, 128]
        self.layer_norm = nn.ModuleList([
            nn.LayerNorm(self.encoder_dims[0]),  # LayerNorm(torch.Size([256]), eps=1e-05, elementwise_affine=True)
            nn.LayerNorm(self.encoder_dims[1]),
            nn.LayerNorm(self.encoder_dims[2])
        ])

        self.mlp = nn.ModuleList([nn.Sequential(
            nn.Linear(self.encoder_dims[0], self.encoder_dims[0]),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_dims[0], self.encoder_dims[0])
        ),
        nn.Sequential(
            nn.Linear(self.encoder_dims[1], self.encoder_dims[1]),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_dims[1], self.encoder_dims[1])
        ),
        nn.Sequential(
            nn.Linear(self.encoder_dims[2], self.encoder_dims[2]),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_dims[2], self.encoder_dims[2])
        )
        ])

        self.slot_attention = nn.ModuleList([
        SlotAttention(
            slots_iters=self.slots_iters,
            num_slots=self.num_slots,
            encoder_dims=self.encoder_dims[0],
            hidden_dim=self.encoder_dims[0]),
        SlotAttention(
            slots_iters=self.slots_iters,
            num_slots=self.num_slots,
            encoder_dims=self.encoder_dims[1],
            hidden_dim=self.encoder_dims[1]),
        SlotAttention(
            slots_iters=self.slots_iters,
            num_slots=self.num_slots,
            encoder_dims=self.encoder_dims[2],
            hidden_dim=self.encoder_dims[2]),
        ])

        self.decoder_cnn = nn.ModuleList()
        self.decoder_cnn.append(
            nn.Sequential(
            nn.Conv2d(self.encoder_dims[0], 64, kernel_size=5, padding=2, stride=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.encoder_dims[0], kernel_size=5, padding=2, stride=1)
            # nn.Conv2d(self.encoder_dims, self.encoder_dims + 1, kernel_size=5, padding=2, stride=1)
            )
        )
        self.decoder_cnn.append(
            nn.Sequential(
            nn.Conv2d(self.encoder_dims[1], 64, kernel_size=5, padding=2, stride=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.encoder_dims[1], kernel_size=5, padding=2, stride=1)
            )
        )
        self.decoder_cnn.append(
            nn.Sequential(
            nn.Conv2d(self.encoder_dims[2], 64, kernel_size=5, padding=2, stride=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.encoder_dims[2], kernel_size=5, padding=2, stride=1)
            )
        )

        self.flow_predictor = nn.ModuleList()
        self.flow_predictor.append(
            nn.Sequential(
            nn.Conv2d(self.encoder_dims[0], 32, kernel_size=3, stride=1,dilation=1,padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=3, stride=1,dilation=1,padding=1, bias=True),
            nn.ReLU(inplace=True)
            )
        )
        self.flow_predictor.append(
            nn.Sequential(
            nn.Conv2d(self.encoder_dims[1], 32, kernel_size=3, stride=1,dilation=1,padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=3, stride=1,dilation=1,padding=1, bias=True),
            nn.ReLU(inplace=True)
            )
        )
        self.flow_predictor.append(
            nn.Sequential(
            nn.Conv2d(self.encoder_dims[2], 32, kernel_size=3, stride=1,dilation=1,padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=3, stride=1,dilation=1,padding=1, bias=True),
            nn.ReLU(inplace=True)
            )
        )

        
        self.corr = Correlation(pad_size=self.search_range, kernel_size=1,
                                max_displacement=self.search_range, stride1=1,
                                stride2=1, corr_multiply=1)

        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in = 32 + (self.dim_corr + 2) * (self.n_frames - 1)

        if self.reduce_dense:
            self.flow_estimators = FlowEstimatorReduce(self.num_ch_in)
        else:
            self.flow_estimators = FlowEstimatorDense(self.num_ch_in)

        self.context_networks = ContextNetwork(
            (self.flow_estimators.feat_dim + 2) * (self.n_frames - 1))

        self.conv_1x1 = nn.ModuleList([conv(self.encoder_dims[0], 32, kernel_size=1, stride=1, dilation=1),
                                       conv(self.encoder_dims[1], 32, kernel_size=1, stride=1, dilation=1),
                                       conv(self.encoder_dims[2], 32, kernel_size=1, stride=1, dilation=1),
                                       conv(32, 32, kernel_size=1, stride=1, dilation=1)])

    def num_parameters(self):
        return sum(
            [p.data.nelement() if p.requires_grad else 0 for p in self.parameters()])

    def init_weights(self):
        for layer in self.named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def make_encoder(self, in_channels, encoder_arch):
        layers = []
        down_factor = 0
        for v in encoder_arch:
            if v == 'MP':
                layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]
                down_factor += 1
            else:
                conv1 = nn.Conv2d(in_channels, v, kernel_size=5, padding=2)
                conv2 = nn.Conv2d(v, v, kernel_size=5, padding=2)

                layers += [conv1, nn.InstanceNorm2d(v, affine=True), nn.ReLU(inplace=True),
                           conv2, nn.InstanceNorm2d(v, affine=True), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers), 2 ** down_factor

    def forward_2_frames(self, x1_pyramid, x2_pyramid):
        # outputs
        flows = []
        recons_res = []
        masks_res = []
        x1_slots_res = []
        recons_flow_res = []
        h_init = [x1_pyramid[0].shape[2],x1_pyramid[1].shape[2],x1_pyramid[2].shape[2]]    

        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        flow = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()

        # x1_pyramid[0]: [4, 192, 6, 13], x1_pyramid[1]:  [4, 128, 12, 26], x1_pyramid[2]:  [4, 96, 24, 52]
        # x1_pyramid[3]: [4, 64, 48, 104], x1_pyramid[4]: [4, 32, 96, 208], x1_pyramid[5]:  [4, 16, 192, 416], x1_pyramid[6]: [4, 3, 384, 832]

        # x1_pyramid[0]: [4,192,7,16], 
        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x2_warp = x2
            else:
                if (x2.shape[2] % 2) == 0 and (x2.shape[3] % 2) == 0:
                    flow = F.interpolate(flow * 2, scale_factor=2,
                                        mode='bilinear', align_corners=True)
                    x2_warp = flow_warp(x2, flow)
                elif (x2.shape[2] % 2) == 0 and (x2.shape[3] % 2) != 0:
                    flow = F.interpolate(flow * 2, size = [flow.shape[2]*2, flow.shape[3]*2-1],
                                        mode='bilinear', align_corners=True)
                    x2_warp = flow_warp(x2, flow)
                elif (x2.shape[2] % 2) != 0 and (x2.shape[3] % 2) == 0:
                    flow = F.interpolate(flow * 2, size = [flow.shape[2]*2-1, flow.shape[3]*2],
                                        mode='bilinear', align_corners=True)
                    x2_warp = flow_warp(x2, flow)
                else:
                    flow = F.interpolate(flow * 2, size = [flow.shape[2]*2-1, flow.shape[3]*2-1],
                                        mode='bilinear', align_corners=True)
                    x2_warp = flow_warp(x2, flow)

            if l < 3:
                # bind slots (slots=8)
                # from ipdb import set_trace
                # set_trace()
                x = einops.rearrange(x1, 'b c h w -> b h w c')  # [4, 3, 10, 192]
                x = spatial_flatten(x)  # Flatten spatial dimensions (treat image as set).   # [4, 30, 192]
                x = self.mlp[l](self.layer_norm[l](x))  # Feedforward network on set.   # [4, 30, 192]
                # `x` has shape: [batch_size, width*height, input_size].

                # Slot Attention module.
                x1_slots,attn = self.slot_attention[l](x)   # [4, 2, 192]   [4,2,30]
                # `slots` has shape: [batch_size, num_slots, slot_size].

                # ---- modify attention mask -------
                attn = einops.rearrange(attn, 'b n (h w) -> b n h w', h = h_init[l])   # [4,2,3,10]

                attn_mask = F.one_hot(torch.argmax(attn, dim=1), num_classes = self.num_slots).float()  # [4,3,10,2]
                attn_mask = einops.rearrange(attn_mask, 'b h w n -> b n h w')  # [4, 2, 3, 10]
                # offset = torch.einsum('bnc, bnhw -> bchw', [x1_slots, attn_mask])  # [4,192,6,13]
                offset = torch.einsum('bnc, bnhw -> bchw', [x1_slots, attn])  # [4,192,3,10]

                
                # ------modify recons flow---------0427----------- 
                # recons = self.decoder_cnn[l](x1 + offset)   # [4, 192, 6, 13])
                recons = self.decoder_cnn[l](offset)   # [4, 192, 3, 10])   0427
                recons_flow = self.flow_predictor[l](recons)   # [4, 2, 3, 10], [4, 2, 12, 26], [4, 2, 24, 52]
                # flow_predictor: 192->2 [6,13]
                # MSE/MAE loss 
                # [4,2,12,26] [4,2,24,52] [4,2,48,104]

                # recon_combined_res.append(recon_combined)
                # recons_res.append(recons)  # 0427 recons_res->recons_flow_res
                recons_flow_res.append(recons_flow)
                masks_res.append(attn_mask)
                x1_slots_res.append(x1_slots)

            # correlation
            out_corr = self.corr(x1, x2_warp)   # [4,81,3,10]
            out_corr_relu = self.leakyRELU(out_corr)

            # concat and estimate flow
            x1_1by1 = self.conv_1x1[l](x1)   # [4, 32, 3, 10]
            x_intm, flow_res = self.flow_estimators(
                torch.cat([out_corr_relu, x1_1by1, flow], dim=1))    # [4, 32, 3, 10]  [4, 22, 3, 10]
            flow = flow + flow_res  # [4, 2, 3, 10]

            flow_fine = self.context_networks(torch.cat([x_intm, flow], dim=1))
            flow = flow + flow_fine

            flows.append(flow)
            # flows[0]: [4,2,12,26], flows[2]: [4,2,24,52], flows[3]: [4,2,48,104] 

            # upsampling or post-processing
            if l == self.output_level:   # output_level = 4
                break
        # from ipdb import set_trace
        # set_trace()
        # masks_res[0]:[4, 2, 3, 10]  [4, 2, 6, 20]  [4, 2, 12, 40]
        # mask = masks_res[2]  # [4,8,48,104]
        masks = torch.split(masks_res[-1], 1, dim=1) # [4, 1, 12, 40] tuple len=slot_num
        # complementary_masks = [(1.0 - mask) for mask in masks]  # [4, 1, 48, 104] list

        flow_masks = [flows[2] * (1.0 - mask) for mask in masks] # [4, 2, 12, 40]
        # flow_complementary_masks = [flow * (1.0 - complementary_mask) for complementary_mask in complementary_masks]  # [4, 2, 48, 104]
        recover = nn.ModuleList([Discriminator(x1_pyramid[2], flow_mask, mask) for (flow_mask,mask) in zip(flow_masks,masks)])
        # complementary_recover = nn.ModuleList([Discriminator(x1_pyramid[2], flow_complementary_mask, complementary_mask) for (flow_complementary_mask,complementary_mask) in zip(flow_complementary_masks,complementary_masks)])
        recover.to(self.device)
        # complementary_recover.to(self.device)
        pred_flows = [recover[l](x1_pyramid[2]) for l in range(self.num_slots)]  # [4, 2, 12, 40] len=slot_num
        # pred_complementary_flows = [complementary_recover[l](x1_pyramid[2]) for l in range(8)]

        recover_from_image = Discriminator(x1_pyramid[2], torch.zeros_like(flows[2]), torch.ones_like(masks_res[2]))
        recover_from_image.to(self.device)
        pred_flow_from_image = recover_from_image(x1_pyramid[2])  # [4,2,12,40]
        if self.upsample:
            flows_upsample = [F.interpolate(flow * 16, scale_factor = 16,
                                   mode='bilinear', align_corners=True) for flow in flows]
            mask_upsample = [F.interpolate(mask * 16, scale_factor = 16,
                                   mode='bilinear', align_corners=True) for mask in masks]
            pred_flows_upsample = [F.interpolate(pred_flow * 16, scale_factor = 16,
                                   mode='bilinear', align_corners=True) for pred_flow in pred_flows]
            pred_flow_from_image_upsample = F.interpolate(pred_flow_from_image * 16, scale_factor = 16,
                                   mode='bilinear', align_corners=True)
            # flows_res = [flows[1], flows[2], flows_upsample[0], flows_upsample[1], flows_upsample[2]]
        # flows[0]: [4,2,24,52], flows[1]: [4,2,48,104], flows[2]: [4,2,96,208],
        # flows[3]: [4,2,192,416], flows[4]: [4,2,384,832] 

        return flows_upsample[::-1], recons_flow_res[::-1], mask_upsample, x1_slots_res[::-1], pred_flows_upsample, pred_flow_from_image_upsample
        
    def forward(self, x, with_bk=False):
        n_frames = x.size(1) / 3
        # from ipdb import set_trace
        # set_trace()

        imgs = [x[:, 3 * i: 3 * i + 3] for i in range(int(n_frames))]
        x = [self.feature_pyramid_extractor(img) + [img] for img in imgs]  # len = 7
        # x = [self.resnet18_model(img) + [img] for img in imgs]
        # x[0][0]: [4, 192, 3, 10]  x[0][1]: [4, 128, 6, 20]
        # x[0][2]: [4, 96, 12, 40] x[0][3]: [4, 64, 24, 80]

        res_dict = {}
        recons = {}
        masks = {}
        x1_slots = {}
        pred_flows={}
        pred_flow_from_image={}
        if n_frames == 2:
            res_dict['flows_fw'], recons['fw'], masks['fw'], x1_slots['fw'], pred_flows['fw'], pred_flow_from_image['fw'] = self.forward_2_frames(x[0], x[1])
            if with_bk:
                res_dict['flows_bw'], recons['bw'], masks['bw'], x1_slots['bw'],pred_flows['bw'], pred_flow_from_image['bw']= self.forward_2_frames(x[1], x[0])
        else:
            raise NotImplementedError
        return res_dict, recons, masks, x1_slots, x[0], pred_flows, pred_flow_from_image