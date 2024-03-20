from torchvision import datasets, transforms
from torch.utils.data import DataLoader 

def get_dataloader(cfg, shuffle=True, drop_last=False, logger=None):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(size=(64,64), scale=(0.8, 1.0), ratio=(0.8, 1.25)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])

    dataset = datasets.ImageFolder(cfg.path, transform)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=shuffle,
        drop_last=drop_last
    )

    return dataloader 
