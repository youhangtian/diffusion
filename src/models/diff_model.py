import torch 
import torch.nn as nn 
from .unet import UNet
from .blocks import PositionalEncoding, ConvNext

from tqdm import tqdm 

def get_diff_model(cfg, num_classes, device='cuda', logger=None):
    diff_model = DiffModel(
        cfg.in_dim,
        cfg.emb_dim,
        cfg.ch_mult,
        cfg.T,
        cfg.t_dim,
        cfg.c_dim,
        cfg.num_blocks,
        num_classes, 
        device,
    )

    return diff_model


class DiffModel(nn.Module):
    def __init__(self,
                 in_dim,
                 emb_dim,
                 ch_mult,
                 T,
                 t_dim,
                 c_dim,
                 num_blocks,
                 num_classes,
                 device,
                 step_size=1,
                 ddim_scale=0.5):
        super(DiffModel, self).__init__()

        self.in_dim = in_dim
        self.num_classes = num_classes 
        self.device = device
        self.step_size = step_size 
        self.ddim_scale = ddim_scale

        self.beta_t = torch.linspace(1e-4, 0.02, T+step_size)[::step_size].to(device)
        self.alpha_t = 1 - self.beta_t
        self.alpha_bar_t = torch.stack([torch.prod(self.alpha_t[:i+1]) for i in range(len(self.alpha_t))])

        self.beta_t = self.beta_t.reshape(-1, 1, 1, 1)
        self.alpha_t = self.alpha_t.reshape(-1, 1, 1, 1)
        self.alpha_bar_t = self.alpha_bar_t.reshape(-1, 1, 1, 1)

        self.T = torch.tensor(T, device=device)
        self.unet = UNet(in_dim, in_dim, emb_dim, ch_mult, t_dim, c_dim, num_blocks).to(device)
        self.t_pos_emb = PositionalEncoding(t_dim).to(device)
        self.c_emb = nn.Linear(self.num_classes, c_dim, bias=False).to(device)

        self.out_noise = ConvNext(in_dim, in_dim).to(device)

    def forward(self, x_t, t, c, null_cls):
        x_t = x_t.to(self.device)
        t = t.to(self.device)
        c = c.to(self.device)
        null_cls = null_cls.to(self.device)

        t = self.t_pos_emb(t.to(torch.long))

        c = nn.functional.one_hot(c.to(torch.int64), self.num_classes).to(self.device).to(torch.float)
        c = self.c_emb(c)
        c[null_cls == 1] *= 0 

        out_unet = self.unet(x_t, t, c)
        out_noise = self.out_noise(out_unet)

        return out_noise
    
    def noise_batch(self, x_0, t):
        x_0 = x_0.to(self.device)
        t = t.to(self.device)

        epsilon = torch.randn_like(x_0, device=self.device)

        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bar_t[t])
        sqrt_1_minus_alpha_bar_t = torch.sqrt(1 - self.alpha_bar_t[t])
        
        return sqrt_alpha_bar_t * x_0 + sqrt_1_minus_alpha_bar_t * epsilon, epsilon

    def unnoise_batch(self, x_t, t, class_label=-1, w=0.0):
        x_t = x_t.to(self.device)
        t = torch.tensor([t], device=self.device)

        if class_label == -1:
            out_noise = self.forward(x_t, t, torch.tensor([0]), torch.tensor([1]))
        else:
            if w == 0:
                out_noise_uncond = 0
            else:
                out_noise_uncond = self.forward(x_t, t, torch.tensor([0]), torch.tensor([1]))

            out_noise_cond = self.forward(x_t, t, torch.tensor([class_label]), torch.tensor([0]))

            out_noise = out_noise_cond + w * (out_noise_cond - out_noise_uncond)

        one_minus_alpha_t = 1 - self.alpha_t[t]
        sqrt_1_minus_alpha_bar_t = torch.sqrt(1 - self.alpha_bar_t[t])
        sqrt_alpha_t = torch.sqrt(self.alpha_t[t])

        x_0_pred = (x_t - one_minus_alpha_t / sqrt_1_minus_alpha_bar_t * out_noise) / sqrt_alpha_t
        random_noise = torch.randn(out_noise.shape, device=self.device) if t > 1 else 0

        out = x_0_pred + torch.sqrt(self.beta_t[t]) * random_noise  
        return out

    def tensor_to_img(self, tensor):
        return (tensor * 127.5 + 127.5).cpu().detach().int().clamp(0, 255).permute(1, 2, 0).numpy()

    @torch.no_grad()
    def sample_img(self, class_label=-1):
        x = torch.randn((1, 3, 64, 64), device=self.device)

        imgs = [self.tensor_to_img(x[0])]
        ddpm_times = list(reversed(range(1, self.T+self.step_size, self.step_size)))
        for t in tqdm(ddpm_times):
            x = self.unnoise_batch(x, t, class_label)
            imgs.append(self.tensor_to_img(x[0]))

        return imgs
