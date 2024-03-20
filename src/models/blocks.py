import torch 
import torch.nn as nn 
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim 

        self.denom = torch.tensor(10000)**((2*torch.arange(self.dim))/self.dim)

    def forward(self, time):
        embeddings = time[:, None] * self.denom[None, :].to(time.device)

        embeddings[::2] = embeddings[::2].sin()
        embeddings[1::2] = embeddings[1::2].cos()

        return embeddings
    

class ClassAttn(nn.Module):
    def __init__(self, cls_dim, out_dim):
        super(ClassAttn, self).__init__()
        self.in_dim = torch.tensor(out_dim)

        self.ln =nn.GroupNorm(out_dim//4 if out_dim > 4 else 1, out_dim)

        self.q_emb = nn.Linear(cls_dim, out_dim)
        self.k_emb = nn.Linear(cls_dim, out_dim)

        self.softmax = nn.Softmax(-1)

        self.out_emb = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 1),
            nn.GroupNorm(out_dim//4 if out_dim > 4 else 1, out_dim)
        )

    def forward(self, x, cls):
        res = x.clone()

        x = self.ln(x)

        q, k = self.q_emb(cls), self.k_emb(cls)
        q = q.unsqueeze(-1)
        k = k.unsqueeze(-1)

        qk = self.softmax(q @ k.permute(0, 2, 1)) / torch.sqrt(self.in_dim)

        x = torch.einsum('nclw, ncd -> nclw', x, qk) 
        x = self.out_emb(x)

        return x + res  

class ChannelAttn(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ChannelAttn, self).__init__()

        k = int(abs((math.log2(channels) / gamma) + (b / gamma)))
        k = k if k % 2 else k + 1 

        self.conv = nn.Conv2d(1, 1, [1, k], padding=[0, k//2], bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.avg_pool(x)
        attn = attn.permute(0, 2, 3, 1)
        attn = self.conv(attn)
        attn = self.sigmoid(attn)
        attn = attn.permute(0, 3, 1, 2)
        return x * attn.expand_as(x)
    

class ResnetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, t_dim, c_dim):
        super().__init__()
        self.t_mlp = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, out_dim))
        self.c_mlp = nn.Sequential(nn.SiLU(), nn.Linear(c_dim, out_dim))

        # self.conv1 = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 7, padding=3, groups=in_dim),
            nn.GroupNorm(in_dim//4 if in_dim > 4 else 1, in_dim),
            nn.Conv2d(in_dim, in_dim*2, 1),
            nn.GELU(),
            nn.Conv2d(in_dim*2, out_dim, 1)
        )
        self.norm1 = nn.GroupNorm(4 if out_dim % 4 == 0 else 1, out_dim)
        self.act1 = nn.SiLU()

        # self.conv2 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 7, padding=3, groups=out_dim),
            nn.GroupNorm(out_dim//4 if out_dim > 4 else 1, out_dim),
            nn.Conv2d(out_dim, out_dim*2, 1),
            nn.GELU(),
            nn.Conv2d(out_dim*2, out_dim, 1)
        )
        self.norm2 = nn.GroupNorm(4 if out_dim % 4 == 0 else 1, out_dim)
        self.act2 = nn.SiLU()

        self.res_conv = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, t, c):
        t = self.t_mlp(t)
        Bt, Ct = t.shape
        t = t.reshape(Bt, Ct, 1, 1)

        c = self.c_mlp(c)
        Bc, Cc = c.shape
        c = c.reshape(Bc, Cc, 1, 1)

        h = self.conv1(x)
        h = self.norm1(h)
        h = h * t 
        h = h * c
        h = self.act1(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)

        return h + self.res_conv(x)

class UnetBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 t_dim,
                 c_dim):
        super(UnetBlock, self).__init__()

        self.res_block = ResnetBlock(in_dim, out_dim, t_dim, c_dim)
        self.cls_block = ClassAttn(c_dim, out_dim)
        self.cha_block = ChannelAttn(out_dim)

    def forward(self, x, t=None, c=None):
        x = self.res_block(x, t, c)
        x = self.cls_block(x, c)
        x = self.cha_block(x)
        return x 
    

class ConvNext(nn.Sequential):
    def __init__(self, in_dim, out_dim):
        super(ConvNext, self).__init__()

        self.blocks = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 7, padding=3, groups=in_dim),
            nn.GroupNorm(in_dim//4 if in_dim > 4 else 1, in_dim),
            nn.Conv2d(in_dim, in_dim*2, 1),
            nn.GELU(),
            nn.Conv2d(in_dim*2, out_dim, 1),
        )

        self.res= nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        res = self.res(x)
        x = self.blocks(x)
        return x + res 
    