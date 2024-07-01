import os
import torch
from torch.utils.tensorboard import SummaryWriter 

from tqdm import tqdm 
import numpy as np

from accelerate import Accelerator

from src.utils import get_config_from_yaml, get_logger
from src.data import get_dataloader
from src.models.diff_model import get_diff_model

def main():
    cfg = get_config_from_yaml('cfg.yaml')

    accelerator = Accelerator(
        mixed_precision='fp16',
        gradient_accumulation_steps=1,
    )
    device = accelerator.device

    if accelerator.is_main_process:
        logger = get_logger(cfg.output_dir)
        writer = SummaryWriter(log_dir=os.path.join(cfg.output_dir, 'tensorboard'))
    else:
        logger = None 
        writer = None

    if logger: logger.info(cfg)

    train_dl = get_dataloader(cfg.data, logger)

    num_classes = len(os.listdir(cfg.data.path))
    diff_model = get_diff_model(cfg.model, num_classes, device, logger)

    opt = torch.optim.AdamW(diff_model.parameters(), lr=cfg.train.lr, eps=1e-4)

    t_dist = torch.distributions.uniform.Uniform(float(1)-float(0.499), float(cfg.model.T)+float(0.499))

    train_dl = accelerator.prepare(train_dl)
    diff_model = accelerator.prepare(diff_model)
    opt = accelerator.prepare(opt)
    if logger: logger.info(f'accelerator prepared! device: {device}')

    steps = 0 
    for epoch in range(cfg.train.epochs):
        accelerator.wait_for_everyone()

        for x_0, labels in tqdm(train_dl):
            t = t_dist.sample((x_0.shape[0],)).to(device)
            t = torch.round(t).to(torch.long)

            probs = torch.rand(x_0.shape[0])
            null_cls = torch.where(probs < cfg.model.p_uncond, 1, 0).to(torch.bool).to(device)

            with torch.no_grad():
                x_t, epsilon = diff_model.module.noise_batch(x_0, t)

            with accelerator.autocast():
                out_noise = diff_model(x_t, t, labels, null_cls)

            loss = ((out_noise - epsilon)**2).flatten().mean()

            accelerator.backward(loss)
            opt.step()
            opt.zero_grad()

            with torch.no_grad():
                if writer:
                    writer.add_scalar(f'train/loss', loss, steps)

                if steps % cfg.train.log_freq == 0 and logger is not None:
                    logger.info(f'epoch{epoch}, steps{steps}, loss: {loss:.6f} ------')
                
                if steps % cfg.train.sample_img_freq == 0 and writer is not None:
                    diff_model.eval()

                    imgs1 = diff_model.module.sample_img()
                    index = list(range(0, len(imgs1), len(imgs1)//10))
                    if (len(imgs1)-1) not in index: index.append(len(imgs1)-1)
                    img1 = np.concatenate(np.array(imgs1)[index], axis=1)
                    
                    imgs2 = diff_model.module.sample_img(labels[0])
                    imgs2[0] = diff_model.module.tensor_to_img(x_0[0])
                    img2 = np.concatenate(np.array(imgs2)[index], axis=1)

                    img = np.concatenate((img1, img2))

                    writer.add_image(f'image/sample_img', img/256, steps, dataformats='HWC')

                    diff_model.train()

            steps += 1

    accelerator.wait_for_everyone()
    if logger:
        logger.info(f'project done')

if __name__ == '__main__':
    main()
