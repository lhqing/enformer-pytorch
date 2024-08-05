"""
Adapted from https://github.com/johahi/borzoi-pytorch/
"""

import torch
import math
from torch import nn, einsum

from einops import rearrange
from .modeling_enformer import relative_shift, Residual, TargetLengthCrop


def get_positional_features_central_mask(positions, features, seq_len):
    pow_rate = math.exp(math.log(seq_len + 1) / features)
    center_widths = torch.pow(
        pow_rate, torch.arange(1, features + 1, device=positions.device)
    ).float()
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).float()


def get_positional_embed(seq_len, feature_size, device):
    distances = torch.arange(-seq_len + 1, seq_len, device=device)

    feature_functions = [
        get_positional_features_central_mask,
    ]

    num_components = len(feature_functions) * 2

    if (feature_size % num_components) != 0:
        raise ValueError(
            f"feature size is not divisible by number of components ({num_components})"
        )

    num_basis_per_class = feature_size // num_components

    embeddings = []
    for fn in feature_functions:
        embeddings.append(fn(distances, num_basis_per_class, seq_len))

    embeddings = torch.cat(embeddings, dim=-1)
    embeddings = torch.cat(
        (embeddings, torch.sign(distances)[..., None] * embeddings), dim=-1
    )
    return embeddings


class Attention(nn.Module):

    def __init__(
        self,
        dim=1536,
        *,
        num_rel_pos_features=1,
        heads=8,
        dim_key=64,
        dim_value=64,
        dropout=0.0,
        pos_dropout=0.0,
    ):
        super().__init__()
        self.scale = dim_key**-0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias=False)

        self.to_out = nn.Linear(dim_value * heads, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        # relative positional encoding

        self.num_rel_pos_features = num_rel_pos_features

        self.register_buffer(
            "positions",
            get_positional_embed(
                4096, self.num_rel_pos_features, self.to_v.weight.device
            ),
            persistent=False,
        )  # 4096 as this should always be the seq len at this pos?

        self.to_rel_k = nn.Linear(num_rel_pos_features, dim_key * heads, bias=False)
        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))

        # dropouts

        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        n, h, device = x.shape[-2], self.heads, x.device

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        q = q * self.scale

        content_logits = einsum(
            "b h i d, b h j d -> b h i j", q + self.rel_content_bias, k
        )

        positions = self.pos_dropout(self.positions)
        rel_k = self.to_rel_k(positions)
        rel_k = rearrange(rel_k, "n (h d) -> h n d", h=h)
        rel_logits = einsum("b h i d, h j d -> b h i j", q + self.rel_pos_bias, rel_k)
        rel_logits = relative_shift(rel_logits)
        logits = content_logits + rel_logits
        attn = logits.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class ConvDna(nn.Module):
    def __init__(self):
        super(ConvDna, self).__init__()
        self.conv_layer = nn.Conv1d(
            in_channels=4, out_channels=512, kernel_size=15, padding="same"
        )
        self.max_pool = nn.MaxPool1d(kernel_size=2, padding=0)

    def forward(self, x):
        return self.max_pool(self.conv_layer(x))


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels=None, kernel_size=1, conv_type="standard"
    ):
        super(ConvBlock, self).__init__()
        if conv_type == "separable":
            self.norm = nn.Identity()
            depthwise_conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                groups=in_channels,
                padding="same",
                bias=False,
            )
            pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
            self.conv_layer = nn.Sequential(depthwise_conv, pointwise_conv)
            self.activation = nn.Identity()
        else:
            self.norm = nn.BatchNorm1d(in_channels, eps=0.001)
            self.activation = nn.GELU(approximate="tanh")
            self.conv_layer = nn.Conv1d(
                in_channels, out_channels, kernel_size=kernel_size, padding="same"
            )

    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv_layer(x)
        return x


class Borzoi(nn.Module):

    def __init__(self, checkpoint_path=None, enable_mouse_head=False):
        # TODO support RC and augs, add gradient functions, and much more
        # TODO rename layers to be understandable if I am feeling like adapting the state dict at some point
        super(Borzoi, self).__init__()
        self.enable_mouse_head = enable_mouse_head
        self.conv_dna = ConvDna()
        self._max_pool = nn.MaxPool1d(kernel_size=2, padding=0)
        self.res_tower = nn.Sequential(
            ConvBlock(in_channels=512, out_channels=608, kernel_size=5),
            self._max_pool,
            ConvBlock(in_channels=608, out_channels=736, kernel_size=5),
            self._max_pool,
            ConvBlock(in_channels=736, out_channels=896, kernel_size=5),
            self._max_pool,
            ConvBlock(in_channels=896, out_channels=1056, kernel_size=5),
            self._max_pool,
            ConvBlock(in_channels=1056, out_channels=1280, kernel_size=5),
        )
        self.unet1 = nn.Sequential(
            self._max_pool,
            ConvBlock(in_channels=1280, out_channels=1536, kernel_size=5),
        )
        transformer = []
        for _ in range(8):
            transformer.append(
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.LayerNorm(1536, eps=0.001),
                            Attention(
                                1536,
                                heads=8,
                                dim_key=64,
                                dim_value=192,
                                dropout=0.05,
                                pos_dropout=0.01,
                                num_rel_pos_features=32,
                            ),
                            nn.Dropout(0.2),
                        )
                    ),
                    Residual(
                        nn.Sequential(
                            nn.LayerNorm(1536, eps=0.001),
                            nn.Linear(1536, 1536 * 2),
                            nn.Dropout(0.2),
                            nn.ReLU(),
                            nn.Linear(1536 * 2, 1536),
                            nn.Dropout(0.2),
                        )
                    ),
                )
            )
        self.horizontal_conv0, self.horizontal_conv1 = ConvBlock(
            in_channels=1280, out_channels=1536, kernel_size=1
        ), ConvBlock(in_channels=1536, out_channels=1536, kernel_size=1)
        self.upsample = torch.nn.Upsample(scale_factor=2)
        self.transformer = nn.Sequential(*transformer)
        self.upsampling_unet1 = nn.Sequential(
            ConvBlock(in_channels=1536, out_channels=1536, kernel_size=1),
            self.upsample,
        )
        self.separable1 = ConvBlock(
            in_channels=1536, out_channels=1536, kernel_size=3, conv_type="separable"
        )
        self.upsampling_unet0 = nn.Sequential(
            ConvBlock(in_channels=1536, out_channels=1536, kernel_size=1),
            self.upsample,
        )
        self.separable0 = ConvBlock(
            in_channels=1536, out_channels=1536, kernel_size=3, conv_type="separable"
        )
        self.crop = TargetLengthCrop(16384 - 32)
        self.final_joined_convs = nn.Sequential(
            ConvBlock(in_channels=1536, out_channels=1920, kernel_size=1),
            nn.Dropout(0.1),
            nn.GELU(approximate="tanh"),
        )
        self.human_head = nn.Conv1d(in_channels=1920, out_channels=7611, kernel_size=1)
        if self.enable_mouse_head:
            self.mouse_head = nn.Conv1d(
                in_channels=1920, out_channels=2608, kernel_size=1
            )
        self.final_softplus = nn.Softplus()
        if checkpoint_path is not None:
            self.load_state_dict(torch.load(checkpoint_path))

    def forward(self, x):
        x = self.conv_dna(x)
        x_unet0 = self.res_tower(x)
        x_unet1 = self.unet1(x_unet0)
        x = self._max_pool(x_unet1)
        
        x_unet1 = self.horizontal_conv1(x_unet1)
        x_unet0 = self.horizontal_conv0(x_unet0)
        
        x = self.transformer(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        
        x = self.upsampling_unet1(x)
        x += x_unet1
        x = self.separable1(x)
        
        x = self.upsampling_unet0(x)
        x += x_unet0
        x = self.separable0(x)
        
        x = self.crop(x.permute(0, 2, 1))
        x = self.final_joined_convs(x.permute(0, 2, 1))

        human_out = self.final_softplus(self.human_head(x))
        if self.enable_mouse_head:
            mouse_out = self.final_softplus(self.mouse_head(x))
            return human_out, mouse_out
        else:
            return human_out
