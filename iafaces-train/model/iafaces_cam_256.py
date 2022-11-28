import math

import torch
from torch import nn
from modules.iafaces_cam.nets_builders import EqualLinear, ConvLayer, ResBlock, StyledConv, ToRGB
from utils.util import get_node_feats, get_node_box
from data_loader.celebahq import BOX


class Encoder(nn.Module):
    def __init__(
            self,
            channel,
            img_size=256,
            num_block=5,
            feats_dim=512
    ):
        super().__init__()

        stem = [ConvLayer(3, channel, 1)]

        in_channel = channel
        for i in range(1, num_block):
            ch = channel * (2 ** i)
            stem.append(ResBlock(in_channel, ch, downsample=True, padding="reflect"))

            in_channel = ch

        self.stem = nn.Sequential(*stem)
        fs = img_size // 2 ** (num_block - 1)

        stem2 = []
        for i in range(0, 1):
            stem2.append(ResBlock(ch, ch, downsample=True, padding="reflect"))
        self.stem2 = nn.Sequential(*stem2)

        self.layers = nn.Sequential(EqualLinear(6144, feats_dim, activation="fused_lrelu"),
                                    EqualLinear(6144, feats_dim, activation="fused_lrelu"),
                                    EqualLinear(9216, feats_dim, activation="fused_lrelu"))

        self.box = get_node_box(fs, 1024, torch.from_numpy(BOX))

    def forward(self, img, mask):
        out = self.stem(img * mask)
        face = self.stem2(out)
        obj_feats = get_node_feats(out, self.box, self.layers)
        nodes = torch.cat([obj_feats], dim=1)
        return face, nodes


class Generator(nn.Module):
    def __init__(
            self,
            img_size,
            style_dim,
            in_fs=8,
            channel_multiplier=2,
            blur_kernel=[1, 3, 3, 1]
    ):
        super().__init__()

        self.size = img_size

        self.style_dim = style_dim

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        box = get_node_box(in_fs, 1024, torch.from_numpy(BOX))

        self.conv1 = StyledConv(
            self.channels[in_fs], self.channels[in_fs], 3, blur_kernel=blur_kernel, box=box
        )
        self.to_rgb1 = ToRGB(self.channels[in_fs], style_dim, upsample=False)

        self.log_size = int(math.log(img_size, 2))
        start = int(math.log(in_fs, 2))
        self.num_layers = (self.log_size - start) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[in_fs]

        off_set = int(math.log(in_fs, 2)) * 2 + 1

        for layer_idx in range(self.num_layers):
            res = (layer_idx + off_set) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(start, self.log_size):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    upsample=True,
                    blur_kernel=blur_kernel,
                    box=box
                )
            )

            box = get_node_box(2 ** i, 1024, torch.from_numpy(BOX))

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, blur_kernel=blur_kernel, box=box
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def forward(
            self,
            latent,
            style,
            noise=None,
            randomize_noise=True
    ):

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

        style2 = torch.flatten(style, 1)

        out = self.conv1(latent, style, noise=noise[0])
        skip = self.to_rgb1(out, style2)

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, style, noise=noise1)
            out = conv2(out, style, noise=noise2)
            skip = to_rgb(out, style2, skip)

            i += 2

        image = skip

        return image


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=1):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, downsample=True))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.final_conv = ConvLayer(in_channel, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)
        out = self.final_conv(out)

        out = out.view(out.shape[0], -1)
        out = self.final_linear(out)

        return out
