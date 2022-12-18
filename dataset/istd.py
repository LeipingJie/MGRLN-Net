import enum
import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset.tools import ImageFolders

class ISTD_Dataset(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        return_name: bool,
        scale_size: tuple,
        crop_size: tuple
    ):
        super().__init__()
        # Set all input args as attributes
        self.__dict__.update(locals())

        self.t_shadow = transforms.Compose(
            #[transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]
            [transforms.ToTensor(), ]
        )
        self.t_mask = transforms.Compose(
            [transforms.ToTensor(), ]
        )
        self.t_free = transforms.Compose(
            [transforms.ToTensor(), ]
        )

        self.prepare_data()

    def prepare_data(self):
        self.train_root = os.path.join(self.data_dir, 'train')
        self.val_root = os.path.join(self.data_dir, 'test')
        
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_data = ImageFolders('train', self.train_root, self.scale_size, self.crop_size,
                                           ['train_A', 'train_B', 'train_C'], 
                                           ['shadow', 'mask', 'free'], 
                                           [self.t_shadow, self.t_mask, self.t_mask], 
                                           self.return_name)

        # Assign test dataset for use in dataloader(s)
        if stage == 'val' or stage is None:
            self.val_data = ImageFolders('val', self.val_root, None, None, 
                                           ['test_A', 'test_B', 'test_C'], 
                                           ['shadow', 'mask', 'free'], 
                                           [self.t_shadow, self.t_mask, self.t_mask], 
                                           self.return_name)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=1, num_workers=self.num_workers, )

if __name__ == '__main__':
    #import matplotlib.pyplot as plt
    dst = ISTD_Dataset(r'../../ISTD_Dataset', 1, 0, True, (448, 448), (400, 400))
    dst.setup()
    ds = dst.val_dataloader()
    for i, d in enumerate(ds):
        print(d['shadow'].shape, d['mask'].shape, d['free'].shape, d['name'])
        '''
        plt.subplot(131)
        plt.imshow(d['shadow'].squeeze().permute(1,2,0).numpy())
        plt.subplot(132)
        plt.imshow(d['mask'].squeeze().numpy())
        plt.subplot(133)
        plt.imshow(d['free'].squeeze().permute(1,2,0).numpy())
        plt.show()
        '''
        if i>5: break