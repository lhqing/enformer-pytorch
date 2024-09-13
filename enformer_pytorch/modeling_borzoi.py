"""
Adapted from https://github.com/johahi/borzoi-pytorch/
"""

import torch
import math
from torch import nn, einsum

from einops import rearrange
from .modeling_enformer import (
    relative_shift,
    Residual,
    TargetLengthCrop,
)
from bolero.tl.generic.module_lora_cond import ConditionalLoRALayer


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


def maybe_pass_additional_params(module, x, *args, **kwargs):
    """Maybe pass additional parameters to the module if it is a subclass of ConditionalLoRALayer."""
    if isinstance(module, ConditionalLoRALayer):
        return module(x, *args, **kwargs)
    else:
        return module(x)


class SequentialwithArgs(nn.Sequential):
    """Sequential module that can pass additional arguments to the modules."""

    no_args_modules = (nn.MaxPool1d, nn.LayerNorm, nn.Dropout, nn.ReLU, nn.GELU)

    def forward(self, x, *args, **kwargs):
        for module in self:
            if isinstance(module, self.no_args_modules):
                x = module(x)
            else:

                x = module(x, *args, **kwargs)
        return x


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

    def forward(self, x, *args, **kwargs):
        h = self.heads

        q = maybe_pass_additional_params(self.to_q, x, *args, **kwargs)
        k = maybe_pass_additional_params(self.to_k, x, *args, **kwargs)
        v = maybe_pass_additional_params(self.to_v, x, *args, **kwargs)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        q = q * self.scale

        content_logits = einsum(
            "b h i d, b h j d -> b h i j", q + self.rel_content_bias, k
        )

        positions = self.pos_dropout(self.positions)
        rel_k = maybe_pass_additional_params(self.to_rel_k, positions)
        rel_k = rearrange(rel_k, "n (h d) -> h n d", h=h)
        rel_logits = einsum("b h i d, h j d -> b h i j", q + self.rel_pos_bias, rel_k)
        rel_logits = relative_shift(rel_logits)
        logits = content_logits + rel_logits
        attn = logits.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = maybe_pass_additional_params(self.to_out, out, *args, **kwargs)
        return out


class ConvDna(nn.Module):
    def __init__(self, in_channels=4, out_channels=512, dna_kernel_size=15):
        super(ConvDna, self).__init__()
        self.conv_layer = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=dna_kernel_size,
            padding="same",
        )
        self.max_pool = nn.MaxPool1d(kernel_size=2, padding=0)

    def forward(self, x, *args, **kwargs):
        x = maybe_pass_additional_params(self.conv_layer, x, *args, **kwargs)
        return self.max_pool(x)


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

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        x = self.activation(x)
        if isinstance(self.conv_layer, nn.Sequential):
            for layer in self.conv_layer:
                x = maybe_pass_additional_params(layer, x, *args, **kwargs)
        else:
            x = maybe_pass_additional_params(self.conv_layer, x, *args, **kwargs)
        return x


class FeedForward(nn.Sequential):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__(
            nn.LayerNorm(input_dim, eps=0.001),
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, *args, **kwargs):
        for module in self:
            x = maybe_pass_additional_params(module, x, *args, **kwargs)
        return x


class TransformerLayer(nn.Module):
    def __init__(
        self,
        channels=1536,
        heads=8,
        dim_key=64,
        dim_value=192,
        dropout=0.05,
        pos_dropout=0.01,
        attention_dropout=0.2,
        feedforward_dropout=0.2,
        num_rel_pos_features=32,
    ):
        self.layers = SequentialwithArgs(
            Residual(
                SequentialwithArgs(
                    nn.LayerNorm(channels, eps=0.001),
                    Attention(
                        channels,
                        heads=heads,
                        dim_key=dim_key,
                        dim_value=dim_value,
                        dropout=dropout,
                        pos_dropout=pos_dropout,
                        num_rel_pos_features=num_rel_pos_features,
                    ),
                    nn.Dropout(attention_dropout),
                )
            ),
            Residual(
                FeedForward(
                    input_dim=channels,
                    hidden_dim=channels * 2,
                    output_dim=channels,
                    dropout=feedforward_dropout,
                ),
            ),
        )

    def forward(self, x, *args, **kwargs):
        return self.layers(x, *args, **kwargs)


class Borzoi(nn.Module):
    default_config = {
        "checkpoint_path": None,
    }

    def __init__(
        self,
        checkpoint_path,
    ):
        super().__init__()

        # =========
        # Conv DNA
        # =========
        self.conv_dna = ConvDna(
            in_channels=4,
            out_channels=512,
            dna_kernel_size=15,
        )
        self._max_pool = nn.MaxPool1d(kernel_size=2, padding=0)

        # ==============
        # Residual Tower
        # ==============
        self.res_tower = SequentialwithArgs(
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

        # ==================
        # UNet connections 1
        # ==================
        self.unet1 = SequentialwithArgs(
            self._max_pool,
            ConvBlock(in_channels=1280, out_channels=1536, kernel_size=5),
        )

        # ===========
        # Transformer
        # ===========
        transformer = [
            TransformerLayer(
                channels=1536,
                heads=8,
                dim_key=64,
                dim_value=192,
                dropout=0.05,
                pos_dropout=0.01,
                attention_dropout=0.2,
                feedforward_dropout=0.2,
                num_rel_pos_features=32,
            )
            for _ in range(8)
        ]
        self.transformer = SequentialwithArgs(*transformer)

        # ===========================
        # UNet horizontal connections
        # ===========================
        self.horizontal_conv0 = ConvBlock(
            in_channels=1280, out_channels=1536, kernel_size=1
        )
        self.horizontal_conv1 = ConvBlock(
            in_channels=1536, out_channels=1536, kernel_size=1
        )

        # ===================================
        # UNet upsampling and separable convs
        # ===================================
        self.upsample = torch.nn.Upsample(scale_factor=2)

        self.upsampling_unet1 = SequentialwithArgs(
            ConvBlock(in_channels=1536, out_channels=1536, kernel_size=1),
            self.upsample,
        )
        self.separable1 = ConvBlock(
            in_channels=1536, out_channels=1536, kernel_size=3, conv_type="separable"
        )

        self.upsampling_unet0 = SequentialwithArgs(
            ConvBlock(in_channels=1536, out_channels=1536, kernel_size=1),
            self.upsample,
        )
        self.separable0 = ConvBlock(
            in_channels=1536, out_channels=1536, kernel_size=3, conv_type="separable"
        )

        # ===================
        # Final Crop and Conv
        # ===================
        self.crop = TargetLengthCrop(16384 - 32)
        self.final_joined_convs = SequentialwithArgs(
            ConvBlock(in_channels=1536, out_channels=1920, kernel_size=1),
            nn.Dropout(0.1),
            nn.GELU(approximate="tanh"),
        )

        if checkpoint_path is not None:
            self.load_state_dict(torch.load(checkpoint_path))

    def forward(self, x, *args, **kwargs):
        # =================
        # DNA Convolution
        # In - x: (bs, 4, 524288)
        # Out - x: (bs, 512, 262144)
        # signal resolution is 1
        # =================
        x = self.conv_dna(x, *args, **kwargs)

        # =================
        # Residual Tower (7 blocks)
        # In - x: (bs, 512, 262144)
        # Out - x_unet0: (bs, 1536, 16384)
        #       x_unet1: (bs, 1536, 8192)
        #       x: (bs, 1536, 4096)
        # signal resolution is 128
        # =================
        x_unet0 = self.res_tower(x, *args, **kwargs)
        x_unet1 = self.unet1(x_unet0, *args, **kwargs)
        x = self._max_pool(x_unet1)
        # horizontal convolution before unet connections
        # conv 1x1, 1280 -> 1536 channels
        x_unet0 = self.horizontal_conv0(x_unet0, *args, **kwargs)
        # conv 1x1, 1536 -> 1536 channels
        x_unet1 = self.horizontal_conv1(x_unet1, *args, **kwargs)

        # =================
        # Transformer
        # In - x: (bs, 1536, 4096)
        # Out - x: (bs, 1536, 4096)
        # signal resolution is 128
        # =================
        x = self.transformer(x.permute(0, 2, 1), *args, **kwargs)
        x = x.permute(0, 2, 1)

        # =================
        # UNet upsampling and separable convs 1
        # In - x: (bs, 1536, 4096)
        # Out - x: (bs, 1536, 8192)
        # signal resolution is 64
        # =================
        x = self.upsampling_unet1(x, *args, **kwargs)
        x += x_unet1
        x = self.separable1(x, *args, **kwargs)

        # =================
        # UNet upsampling and separable convs 0
        # In - x: (bs, 1536, 8192)
        # Out - x: (bs, 1536, 16384)
        # signal resolution is 32
        # =================
        x = self.upsampling_unet0(x, *args, **kwargs)
        x += x_unet0
        x = self.separable0(x, *args, **kwargs)

        # =================
        # Final Crop and Conv
        # In - x: (bs, 1536, 16384)
        # Out - x: (bs, 1920, 16352)
        # signal resolution is 32
        # =================
        x = self.crop(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.final_joined_convs(x, *args, **kwargs)
        return x


class BorzoiWithOutputHead(Borzoi):
    def __init__(self, checkpoint_path=None, enable_mouse_head=True):
        super().__init__(checkpoint_path=None)

        self.human_head = nn.Conv1d(in_channels=1920, out_channels=7611, kernel_size=1)

        self.enable_mouse_head = enable_mouse_head
        if self.enable_mouse_head:
            self.mouse_head = nn.Conv1d(
                in_channels=1920, out_channels=2608, kernel_size=1
            )
        self.final_softplus = nn.Softplus()

        if checkpoint_path is not None:
            self.load_state_dict(torch.load(checkpoint_path))

    def forward(self, x, *args, **kwargs):
        x = super().forward(x, *args, **kwargs)

        human_out = self.final_softplus(self.human_head(x))
        # human_out: (bs, 7611, 16352)
        # equivalent to 16352*32 = 523264 bp signal

        if self.enable_mouse_head:
            mouse_out = self.final_softplus(self.mouse_head(x))
            # mouse_out: (bs, 2608, 16352)
            # equivalent to 16352*32 = 523264 bp signal
            return human_out, mouse_out
        else:
            return human_out
