from scripts_coco.test_precision_flag_single import FlagDataset

import os

import torch
from torchvision import transforms


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


class Main:
    def __init__(self):
        self.crop_size = 299
        self.coco_path = "/home/nganltp/laso/data/flags"
        self.copy_label = [5]
        self.dataset: FlagDataset = None
        self.copy_to = "/home/nganltp/laso/data/flags/error"

    def init(self):
        scaler = transforms.Resize(self.crop_size)
        val_transform = transforms.Compose(
            [
                scaler,
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
        self.dataset = FlagDataset(
            root_dir=self.coco_path,
            set_name='train',
            transform=val_transform,
        )
        os.removedirs(self.copy_to)
        os.makedirs(self.copy_to, exist_ok=False)

    def run(self):
        for label in self.copy_label:
            self.dataset.copy_label(label, self.copy_to)


if __name__ == "__main__":
    main = Main()
    main.init()
    main.run()
