import torch 
import torch.nn as nn 

from .blocks import UnetBlock, ChannelAttn

class UNet(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 emb_dim,
                 ch_mult,
                 t_dim,
                 c_dim,
                 num_blocks):
        super(UNet, self).__init__()

        self.t_emb = nn.Sequential(
            nn.Linear(t_dim, t_dim),
            nn.GELU(),
            nn.Linear(t_dim, t_dim),
        )

        self.in_conv = nn.Conv2d(in_dim, emb_dim, 7, padding=3)

        blocks = []
        cur_dim = emb_dim 
        for i in range(1, num_blocks+1):
            next_dim = emb_dim * (2 ** (ch_mult * i))
            blocks.append(UnetBlock(cur_dim, next_dim, t_dim, c_dim))
            if i != num_blocks+1:
                blocks.append(nn.Conv2d(next_dim, next_dim, kernel_size=3, stride=2, padding=1))
            cur_dim = next_dim

        self.down_blocks = nn.Sequential(*blocks)

        intermediate_dim = cur_dim 
        self.intermediate = nn.Sequential(
            UnetBlock(intermediate_dim, intermediate_dim, t_dim, c_dim),
            ChannelAttn(intermediate_dim),
            UnetBlock(intermediate_dim, intermediate_dim, t_dim, c_dim),
        )

        blocks = []
        for i in range(num_blocks, -1, -1):
            cur_dim = emb_dim * (2 ** (ch_mult * i))
            next_dim = emb_dim * (2 ** (ch_mult * (i - 1)))
            if i > 0:
                blocks.append(nn.ConvTranspose2d(cur_dim, cur_dim, kernel_size=4, stride=2, padding=1))
                blocks.append(UnetBlock(2 * cur_dim, next_dim, t_dim, c_dim))
            else:
                blocks.append(UnetBlock(cur_dim, cur_dim, t_dim, c_dim))
                blocks.append(UnetBlock(cur_dim, out_dim, t_dim, c_dim))

        self.up_blocks = nn.Sequential(*blocks)

        self.out = nn.Conv2d(out_dim, out_dim, 7, padding=3)

    def forward(self, x, t, c):
        t = self.t_emb(t)

        x = self.in_conv(x)

        residuals = []
        for i in range(len(self.down_blocks)):
            if type(self.down_blocks[i]) == UnetBlock:
                x = self.down_blocks[i](x, t, c)
                residuals.append(x.clone())
            else:
                x = self.down_blocks[i](x) 

        for i in range(len(self.intermediate)):
            if type(self.intermediate[i]) == UnetBlock:
                x = self.intermediate[i](x, t, c)
            else:
                x = self.intermediate[i](x)

        for i in range(len(self.up_blocks)):
            if type(self.up_blocks[i]) == UnetBlock:
                if len(residuals) > 0:
                    x = self.up_blocks[i](torch.cat((x, residuals[-1]), dim=1), t, c)
                    residuals.pop()
                else:
                    x = self.up_blocks[i](x, t, c)
            else:
                x = self.up_blocks[i](x) 

        return self.out(x)
    