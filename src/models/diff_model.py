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
        self.unet = UNet(in_dim, in_dim*2, emb_dim, ch_mult, t_dim, c_dim, num_blocks).to(device)
        self.t_pos_emb = PositionalEncoding(t_dim).to(device)
        self.c_emb = nn.Linear(self.num_classes, c_dim, bias=False).to(device)

        self.out_mean = ConvNext(in_dim, in_dim).to(device)
        self.out_var = ConvNext(in_dim, in_dim).to(device)

    def forward(self, x_t, t, c, null_cls):
        x_t = x_t.to(self.device)
        t = t.to(self.device)
        c = c.to(self.device)
        null_cls = null_cls.to(self.device)

        t = self.t_pos_emb(t.to(torch.long))

        c = nn.functional.one_hot(c.to(torch.int64), self.num_classes).to(self.device).to(torch.float)
        c = self.c_emb(c)
        c[null_cls == 1] *= 0 

        out = self.unet(x_t, t, c)

        noise_mean, noise_var = out[:, :self.in_dim], out[:, self.in_dim:]

        noise_mean = self.out_mean(noise_mean)
        noise_var = self.out_var(noise_var)

        return noise_mean, noise_var
    
    def noise_batch(self, x_0, t):
        x_0 = x_0.to(self.device)
        t = t.to(self.device)

        epsilon = torch.randn_like(x_0, device=self.device)

        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bar_t[t])
        sqrt_1_minus_alpha_bar_t = torch.sqrt(1 - self.alpha_bar_t[t])
        
        return sqrt_alpha_bar_t * x_0 + sqrt_1_minus_alpha_bar_t * epsilon, epsilon
    
    def get_mean_t(self, noise_mean, x_t, t, corrected=True):
        beta_t = self.beta_t[t]
        sqrt_alpha_t = torch.sqrt(self.alpha_t[t])
        alpha_bar_t = self.alpha_bar_t[t]
        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bar_t[t])
        sqrt_1_minus_alpha_bar_t = torch.sqrt(1 - self.alpha_bar_t[t])
        alpha_bar_t_1 = self.alpha_bar_t[t-1]
        sqrt_alpha_bar_t_1 = torch.sqrt(self.alpha_bar_t[t-1])

        t = t.reshape(t.shape + (1,)*3)

        if not corrected:
            return (1 / sqrt_alpha_t) * (x_t - (beta_t / sqrt_1_minus_alpha_bar_t) * noise_mean)
        
        mean_t = torch.where(
            t == 0,
            (1 / sqrt_alpha_t) * (x_t - (beta_t / sqrt_1_minus_alpha_bar_t) * noise_mean),
            (sqrt_alpha_bar_t_1 * beta_t) / (1 - alpha_bar_t) * torch.clamp(
                (1 / sqrt_alpha_bar_t) * x_t - (sqrt_1_minus_alpha_bar_t / sqrt_alpha_bar_t) * noise_mean,
                -1,
                1
            ) + (((1 - alpha_bar_t_1) * sqrt_alpha_t) / (1 - alpha_bar_t)) * x_t
        )

        return mean_t

    def get_var_t(self, noise_var, t):
        beta_t = self.beta_t[t]
        beta_tilde_t = ((1 - self.alpha_bar_t[t-1]) / (1 - self.alpha_bar_t[t])) * self.beta_t[t]

        var_t = torch.exp(torch.clamp(
            noise_var * torch.log(beta_t) + (1 - noise_var) * torch.log(beta_tilde_t),
            torch.tensor(-30, device=beta_t.device),
            torch.tensor(30, device=beta_t.device)
        ))

        return var_t 

    def unnoise_batch(self, x_t, t_ddim, t_ddpm, class_label=-1, w=0.0, corrected=False):
        x_t = x_t.to(self.device)
        t_ddim = torch.tensor([t_ddim], device=self.device)
        t_ddpm = torch.tensor([t_ddpm], device=self.device)

        if class_label == -1:
            noise_mean, noise_var = self.forward(x_t, t_ddpm, torch.tensor([0]), torch.tensor([1]))
        else:
            if w == 0:
                noise_mean_uncond = noise_var_uncond = 0
            else:
                noise_mean_uncond, noise_var_uncond = self.forward(x_t, t_ddpm, torch.tensor([0]), torch.tensor([1]))

            noise_mean_cond, noise_var_cond = self.forward(x_t, t_ddpm, torch.tensor([class_label]), torch.tensor([0]))

            noise_mean = noise_mean_cond + w * (noise_mean_cond - noise_mean_uncond)
            noise_var = noise_var_cond + w * (noise_var_cond - noise_var_uncond)

        var_t = self.get_var_t(noise_var, t_ddim)

        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bar_t[t_ddim])
        sqrt_1_minus_alpha_bar_t = torch.sqrt(1 - self.alpha_bar_t[t_ddim])
        alpha_bar_t_1 = self.alpha_bar_t[t_ddim-1]
        sqrt_alpha_bar_t_1 = torch.sqrt(self.alpha_bar_t[t_ddim-1])
        beta_tilde_t = ((1 - self.alpha_bar_t[t_ddim-1]) / (1 - self.alpha_bar_t[t_ddim])) * self.beta_t[t_ddim]

        var_t = self.ddim_scale * var_t
        beta_tilde_t = self.ddim_scale * beta_tilde_t 

        x_0_pred = (x_t - sqrt_1_minus_alpha_bar_t * noise_mean) / sqrt_alpha_bar_t
        if corrected: x_0_pred = x_0_pred.clamp(-1, 1)
        x_t_dir_pred = torch.sqrt(torch.clamp(1 - alpha_bar_t_1 - beta_tilde_t, 0, torch.inf)) * noise_mean
        random_noise = torch.randn(noise_mean.shape, device=self.device) * torch.sqrt(var_t)

        out = sqrt_alpha_bar_t_1 * x_0_pred + x_t_dir_pred + random_noise  
        return out

    def tensor_to_img(self, tensor):
        return (tensor * 127.5 + 127.5).cpu().detach().int().clamp(0, 255).permute(1, 2, 0).numpy()

    @torch.no_grad()
    def sample_img(self, class_label=-1):
        x = torch.randn((1, 3, 64, 64), device=self.device)

        imgs = [self.tensor_to_img(x[0])]
        ddpm_times = list(reversed(range(1, self.T+self.step_size, self.step_size)))
        ddim_times = list(reversed(range(1, len(ddpm_times)+1)))
        for t_ddim, t_ddpm in tqdm(zip(ddim_times, ddpm_times)):
            x = self.unnoise_batch(x, t_ddim, t_ddpm, class_label)
            imgs.append(self.tensor_to_img(x[0]))

        return imgs
