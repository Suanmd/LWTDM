import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# model


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=8, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, dropout=0, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            time_emb_dim, dim_out, use_affine_level=False)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        if exists(self.noise_func) and exists(time_emb):
            h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head
        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.q = nn.Conv2d(in_channel, in_channel, 1, bias=False)
        self.kv = nn.Conv2d(in_channel, in_channel * 2, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, x, xe):
        batch, channel, height, width = x.shape
        n_head = self.n_head
        head_dim = channel // n_head
        # x = self.norm(x)
        # xe = self.norm(xe)
        query = self.q(x).view(batch, n_head, head_dim * 1, height, width)
        kv = self.kv(xe).view(batch, n_head, head_dim * 2, xe.shape[-2], xe.shape[-1])
        key, value = kv.chunk(2, dim=2)
        attn = torch.einsum("bnchw, bncyx -> bnhwyx", query, key).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, xe.shape[-2], xe.shape[-1])
        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + x


class ResnetBlocks(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, norm_groups=32, dropout=0):
        super().__init__()
        self.res_block = ResnetBlock(
            dim, dim_out, time_emb_dim, norm_groups=norm_groups, dropout=dropout)

    def forward(self, x, time_emb):
        if exists(time_emb):
            x = self.res_block(x, time_emb)
        else:
            x = self.res_block(x, None)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channel=3,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=8,
        res_blocks=3,
        dropout=0,
        with_time_emb=True,
        image_size=128
    ):
        super().__init__()

        if with_time_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None


        self.res_blocks = res_blocks
        fusions = []
        num_mults = len(channel_mults)
        channel_mult = inner_channel * channel_mults[-1]
        pre_channel = channel_mult // (2**4)
        downs = [nn.Conv2d(in_channel, pre_channel, kernel_size=3, padding=1)]
        for _ in range(num_mults):
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocks(
                    pre_channel, pre_channel, time_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel//4),
                    dropout=dropout))
            if num_mults == 1:
                downs.append(nn.PixelUnshuffle(4))
                for _ in range(res_blocks):
                    fusions.append(Upsample(pre_channel))
                pre_channel = pre_channel * (2**4)
            else:
                downs.append(nn.PixelUnshuffle(2))
                for _ in range(res_blocks):
                    fusions.append(Upsample(pre_channel))
                pre_channel = pre_channel * (2**2)
        self.downs = nn.ModuleList(downs)
        self.fusions = nn.ModuleList(fusions)

        mids = []
        pre_channel = channel_mult
        for _ in range(0, attn_res):
            mids.append(ResnetBlocks(
                pre_channel, pre_channel, time_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel//4),
                dropout=dropout))
            mids.append(SelfAttention(pre_channel, norm_groups=min(norm_groups, pre_channel//4)))
        self.mid = nn.ModuleList(mids)

        ups = []
        for _ in reversed(range(num_mults)):
            for _ in range(0, res_blocks):
                ups.append(ResnetBlocks(
                    pre_channel, pre_channel, time_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel//4),
                    dropout=dropout))
            if num_mults == 1:
                ups.append(nn.PixelShuffle(4))
                pre_channel = pre_channel // (2**4)
            else:
                ups.append(nn.PixelShuffle(2))
                pre_channel = pre_channel // (2**2)
        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=min(norm_groups, pre_channel//4))


    def forward(self, x, xe, time):
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None
        n = -1
        flag = 0
        for layer in self.downs:
            if isinstance(layer, ResnetBlocks):

                x = layer(x, t)
                xe = layer(xe, None)
                # combine x and xe
                # x = x + nn.functional.interpolate(xe, scale_factor=8, mode='bilinear', align_corners=False)
                if flag % (self.res_blocks+1) == 0:
                    x = x + self.fusions[flag//(self.res_blocks+1)](xe)
                flag += 1
            else:
                x = layer(x)
                xe = layer(xe)
                n += 1

        for layer in self.mid:
            if isinstance(layer, ResnetBlocks):
                x = layer(x, t)
                xe = layer(xe, None)
            else:
                x = layer(x, xe)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocks):
                x = layer(x, t)
            else:
                x = layer(x)

        return self.final_conv(x)


if __name__ == '__main__':
    from thop import profile
    from thop import clever_format
    x = torch.randn(1,3,224,224)
    xe = torch.randn(1,3,28,28)
    t = torch.tensor([1493])

    unet = UNet(
        in_channel=3,
        out_channel=3,
        norm_groups=32,
        inner_channel=32,        # 64 or 128
        channel_mults=[8],       # [1,1] or [2,2] or [1] or [2]
        attn_res=2,              # 6 or 8
        res_blocks=2,
        dropout=0.2,
        image_size=224
    )
    flops, params = profile(unet, inputs=(x, xe, t))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)


    print('done')