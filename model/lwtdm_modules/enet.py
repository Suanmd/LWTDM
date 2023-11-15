import math
import torch
from torch import nn
from inspect import isfunction

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb

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
        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        if exists(self.mlp) and exists(time_emb):
            h += self.mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)

class CrossAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, n_div=12, norm_groups=32):
        super().__init__()

        self.n_head = n_head
        # self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.q = nn.Conv2d(in_channel, in_channel//n_div, 1, bias=False)
        self.kv = nn.Conv2d(in_channel, in_channel//n_div * 2, 1, bias=False)
        # self.out = nn.Conv2d(in_channel, in_channel//n_div, 1)
        self.out = nn.Conv2d(in_channel//n_div, in_channel, 3, padding=1)

    def forward(self, x, xe, n_div=12):
        batch, channel, height, width = x.shape
        n_head = self.n_head
        head_dim = channel // (n_head * n_div)
        # x = self.norm(x)
        # xe = self.norm(xe)
        query = self.q(x).view(batch, n_head, head_dim * 1, height, width)
        kv = self.kv(xe).view(batch, n_head, head_dim * 2, xe.shape[-2], xe.shape[-1])
        key, value = kv.chunk(2, dim=2)
        attn = torch.einsum("bnchw, bncyx -> bnhwyx", query, key).contiguous() / math.sqrt(channel//n_div)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, xe.shape[-2], xe.shape[-1])
        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel//n_div, height, width))

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

class Net(nn.Module):
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
            time_dim = inner_channel
            self.time_mlp = nn.Sequential(
                TimeEmbedding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            time_dim = None
            self.time_mlp = None

        fusions = []
        num_mults = len(channel_mults)
        channel_mult = inner_channel * channel_mults[-1]
        pre_channel = channel_mult // (2**4)
        downs = [nn.Conv2d(in_channel, pre_channel, kernel_size=3, padding=1)]
        for _ in range(num_mults):
            for _ in range(0, res_blocks-1):
                downs.append(ResnetBlocks(
                    pre_channel, pre_channel, time_emb_dim=time_dim, norm_groups=min(norm_groups, pre_channel//8),
                    dropout=dropout))
            if num_mults == 1:
                downs.append(nn.PixelUnshuffle(4))
                for _ in range(0, res_blocks-1):
                    fusions.append(Upsample(pre_channel))
                pre_channel = pre_channel * (2**4)
            else:
                downs.append(nn.PixelUnshuffle(2))
                for _ in range(0, res_blocks-1):
                    fusions.append(Upsample(pre_channel))
                pre_channel = pre_channel * (2**2)
        self.downs = nn.ModuleList(downs)
        self.fusions = nn.ModuleList(fusions)

        mids = []
        pre_channel = channel_mult
        for _ in range(0, attn_res):
            mids.append(ResnetBlocks(
                pre_channel, pre_channel, time_emb_dim=time_dim, norm_groups=min(norm_groups, pre_channel//8),
                dropout=dropout))
            mids.append(CrossAttention(pre_channel, norm_groups=min(norm_groups, pre_channel//8)))
        self.mid = nn.ModuleList(mids)

        ups = []
        for _ in reversed(range(num_mults)):
            for _ in range(0, res_blocks):
                ups.append(ResnetBlocks(
                    pre_channel, pre_channel, time_emb_dim=time_dim, norm_groups=min(norm_groups, pre_channel//8),
                    dropout=dropout))
            if num_mults == 1:
                ups.append(nn.PixelShuffle(4))
                pre_channel = pre_channel // (2**4)
            else:
                ups.append(nn.PixelShuffle(2))
                pre_channel = pre_channel // (2**2)
        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=min(norm_groups, pre_channel//8))


    def forward(self, x, xe, time):
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        n = 0
        for layer in self.downs:
            if isinstance(layer, ResnetBlocks):
                x = layer(x, t)
                xe = layer(xe, None)
                # x = x + nn.functional.interpolate(xe, scale_factor=8, mode='bilinear', align_corners=False)
                x = x + self.fusions[n](xe)
                n = n + 1
            else:
                x = layer(x)
                xe = layer(xe)

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
    net = Net(
        in_channel=3,
        out_channel=3,
        norm_groups=32,
        inner_channel=32,
        channel_mults=[12,12],
        attn_res=2,
        res_blocks=2,
        dropout=0,
        image_size=224
    )
    x = torch.randn(1,3,224,224)
    xe = torch.randn(1,3,28,28)
    t = torch.tensor([1013])
    out = net(x, xe, t)
    from thop import profile
    from thop import clever_format
    flops, params = profile(net, inputs=(x, xe, t))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
    print('done')