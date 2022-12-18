import os

import torch
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from dataset.istd import ISTD_Dataset
from dataset.aistd import AISTD_Dataset
from dataset.srd import SRD_Dataset

from config import get_args
from model import create_model
from utils import calc_RMSE, backup_code


class SRNetFramework(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hparams.learning_rate = args.lr
        self.hparams.weight_decay = args.wd
        self.save_hyperparameters()
        self.model = create_model(args)

    def forward(self, x):
        return self.model(x)

    def _calculate_train_loss(self, batch, batch_idx):
        shadow_img = batch['shadow']
        mask_img = batch['mask']
        free_img = batch['free']
        
        pred_masks, pred_rgbs = self.model(shadow_img)

        # mask loss
        loss_masks = []
        for i, pred_mask in enumerate(pred_masks):
            loss_mask = F.mse_loss(pred_mask, mask_img)
            loss_masks.append(loss_mask)
            self.log(f'train/mask_loss_{i}', loss_mask)

        # rgb loss
        loss_rgbs = []
        for i, pr in enumerate(pred_rgbs):
            loss_rgb = F.mse_loss(pr, free_img)
            loss_rgbs.append(loss_rgb)
            self.log(f'train/rgb_loss_{i}', loss_rgb)

        loss = 1.5*sum(loss_masks) + sum(loss_rgbs)

        self.log('train/train_loss', loss)
        return loss

    def _calculate_val_loss_acc(self, batch):
        shadow_img = batch['shadow']
        mask_img = batch['mask']
        free_img = batch['free']

        pred_masks, pred_rgbs = self.model(shadow_img)

        # mask loss
        loss_masks = []
        for i, pred_mask in enumerate(pred_masks):
            loss_mask = F.mse_loss(pred_mask, mask_img)
            loss_masks.append(loss_mask)
            self.log(f'val/mask_loss_{i}', loss_mask)

        # rgb loss
        loss_rgbs = []
        for i, pr in enumerate(pred_rgbs):
            loss_rgb = F.mse_loss(pr, free_img)
            loss_rgbs.append(loss_rgb)
            self.log(f'val/rgb_loss_{i}', loss_rgb)

        loss = loss_mask + sum(loss_rgbs)

        pred_img = pred_rgbs[-1]

        # calculate rmse
        free_img_np = free_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        pred_img_np = pred_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        mask_img_np = mask_img.squeeze(0).permute(1, 2, 0).cpu().numpy()

        diff = calc_RMSE(free_img_np, pred_img_np)

        shadow_rmse = (diff * mask_img_np).sum()
        nonshadow_rmse = (diff * (1 - mask_img_np)).sum()
        all_rmse = diff.sum()

        eval_shadow_sum = mask_img_np.sum()
        eval_nonshadow_sum = (1 - mask_img_np).sum()
        eval_sum =  mask_img_np.sum() + (1 - mask_img_np).sum()

        self.log('val/val_loss', loss)
        return loss, shadow_rmse, nonshadow_rmse, all_rmse, eval_shadow_sum, eval_nonshadow_sum, eval_sum
        
    def training_step(self, batch, batch_idx):
        train_loss = self._calculate_train_loss(batch, batch_idx)
        info = {'loss':train_loss}
        return info


    def validation_step(self, batch, batch_idx):
        '''
        val_loss, mae, sdist, smask, ndist, nmask = self._calculate_val_loss_acc(batch)
        info = {'loss':val_loss, 'mae':torch.from_numpy(mae), 'sdist':sdist, 'smask':smask, 'ndist':ndist, 'nmask':nmask}
        info['progress_bar'] = {'mae':mae}
        return info
        '''
        val_loss, shadow_rmse, nonshadow_rmse, all_rmse, eval_shadow_sum, eval_nonshadow_sum, eval_sum = self._calculate_val_loss_acc(batch)
        info = {'loss':val_loss, 'shadow_rmse':shadow_rmse, 'nonshadow_rmse': nonshadow_rmse, 'all_rmse': all_rmse, \
                'eval_shadow_sum':eval_shadow_sum, 'eval_nonshadow_sum':eval_nonshadow_sum, 'eval_sum':eval_sum}

        return info
        

    def validation_epoch_end(self, outputs):
        shadow_rmse = torch.stack([torch.FloatTensor([x['shadow_rmse']]) for x in outputs]).mean()
        nonshadow_rmse = torch.stack([torch.FloatTensor([x['nonshadow_rmse']]) for x in outputs]).mean()
        all_rmse = torch.stack([torch.FloatTensor([x['all_rmse']]) for x in outputs]).mean()
        eval_shadow_sum = torch.stack([torch.FloatTensor([x['eval_shadow_sum']]) for x in outputs]).mean()
        eval_nonshadow_sum = torch.stack([torch.FloatTensor([x['eval_nonshadow_sum']]) for x in outputs]).mean()
        eval_sum = torch.stack([torch.FloatTensor([x['eval_sum']]) for x in outputs]).mean()

        all_mae = all_rmse/eval_sum
        s_mae = shadow_rmse/eval_shadow_sum
        ns_mae = nonshadow_rmse/eval_nonshadow_sum

        self.log('all_mae', all_mae, prog_bar=True)
        self.log('shadow', s_mae, prog_bar=True)
        self.log('non-shadow', ns_mae, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
                                                     self.args.lr, 
                                                     epochs=self.args.epochs, 
                                                     steps_per_epoch=self.args.steps_per_epoch, 
                                                     cycle_momentum=True, 
                                                     base_momentum=0.85, 
                                                     max_momentum=0.95, 
                                                     last_epoch=-1, 
                                                     pct_start=self.args.pct_start, 
                                                     div_factor=self.args.div_factor, 
                                                     final_div_factor=self.args.final_div_factor)
        return [optimizer], [lr_scheduler]


def main(hparams):
    ##################################################################
    # dataset
    ##################################################################
    scale_size = (hparams.scale_size, hparams.scale_size)
    crop_size = (hparams.crop_size, hparams.crop_size)
    dm = ISTD_Dataset(data_dir=hparams.root_istd, batch_size=hparams.bs, num_workers=hparams.n_workers, return_name=True, scale_size=scale_size, crop_size=crop_size)
    dm.setup()
    hparams.steps_per_epoch = len(dm.val_dataloader())

    _save_dir, _name = os.getcwd(), f'logs_{hparams.tag}'
    ##################################################################
    # save code
    ##################################################################
    code_save_dir = os.path.join(_save_dir, _name)
    backup_code(code_save_dir)
    
    ##################################################################
    # call backs
    ##################################################################

    # learning rate
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # logger
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=_save_dir,
        version=None,
        name=_name
    )
    
    # checkpoint saver
    checkpoint_callback = ModelCheckpoint(
        monitor='all_mae',
        filename='srnet-{epoch:02d}-{all_mae:.6f}',
        save_top_k=5,
        mode='min',
        save_last=True
    )

    model = SRNetFramework(hparams)
    trainer = Trainer(
        max_epochs=hparams.epochs,
        gpus=len(params.gpus.split(',')),
        accelerator='gpu',
        default_root_dir=hparams.save_path,
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        progress_bar_refresh_rate=1,
        precision=16 if hparams.amp else 32, 
        check_val_every_n_epoch=2,
        #overfit_batches=10
    )

    # If True, we plot the computation graph in tensorboard
    trainer.logger._log_graph = True
    
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    params = get_args()
    seed_everything(params.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = params.gpus

    main(hparams=params)