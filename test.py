import os, argparse

import torch
import torch.nn.functional as F

from dataset.istd import ISTD_Dataset
from model import create_model
from utils import calc_RMSE
from PIL import Image

import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SRNet inference', fromfile_prefix_chars='@', conflict_handler='resolve')
    parser.add_argument('--backbone', default='b5', type=str, help='backbone name')
    parser.add_argument('--sm', action='store_true', help='whether save mask images')
    parser.add_argument('--root_istd', default='./data/ISTD_Dataset', type=str, help='ISTD dataset root directory')
    parser.add_argument('--tag', required=True, type=str, help='checkpoint tag')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')
    parser.add_argument('--model_name', default='accv2022', type=str, help='model name to choose')
    parser.add_argument('--pretrained', default='./pretrained/last.ckpt', type=str, help='path to pretrained model')

    args = parser.parse_args()
    
    tag = args.tag
    val_root = args.root_istd
    output_dir = f'outputs_{tag}'
    os.makedirs(output_dir, exist_ok=True)

    model = create_model(args)
    # load checkpoint
    checkpoint = torch.load(args.pretrained, map_location=torch.device("cpu"))
    checkpoint_ = {}
    for k, v in checkpoint['state_dict'].items():
        if not k.startswith('model.'):
            continue

        k = k[6:] # remove 'model.'
        checkpoint_[k] = v

    model.load_state_dict(checkpoint_, strict=True)
    # model to gpu
    model = model.cuda()
    model.eval()
    dst = ISTD_Dataset(val_root, 1, 0, True, None, None)
    dst.setup(stage='val')
    val_data = dst.val_dataloader()

    lab_maes = []
    sdists, smasks, ndists, nmasks = [], [], [], []

    eval_shadow_rmse = 0
    eval_nonshadow_rmse = 0
    eval_rmse = 0

    eval_shadow_sum = 0
    eval_nonshadow_sum = 0
    eval_sum = 0

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_data)):
            shadow_img = batch['shadow'].cuda()
            mask_img = batch['mask'].cuda()
            free_img = batch['free'].cuda()

            #print(batch['name'])
            name = os.path.join(output_dir, batch['name'][0])
            pred_masks, pred_rgbs = model(shadow_img)
            pred_img = pred_rgbs[-1]
 
            # save image
            Image.fromarray((pred_img.detach().cpu().squeeze().permute(1,2,0).numpy()*255).astype(np.uint8)).save(name)
            if args.sm:
                pred_mask = pred_masks[-1]
                mask_name = os.path.join(output_dir, batch['name'][0][:-4]+'_mask.png')
                Image.fromarray((pred_mask.detach().cpu().squeeze().numpy()*255).astype(np.uint8)).save(mask_name)

            # calculate rmse
            free_img_np = free_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            pred_img_np = pred_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            mask_img_np = mask_img.squeeze(0).permute(1, 2, 0).cpu().numpy()

            diff = calc_RMSE(free_img_np, pred_img_np)

            shadow_rmse = (diff * mask_img_np).sum()
            nonshadow_rmse = (diff * (1 - mask_img_np)).sum()
            whole_rmse = diff.sum()

            eval_shadow_rmse += shadow_rmse
            eval_nonshadow_rmse += nonshadow_rmse
            eval_rmse += whole_rmse

            eval_shadow_sum += mask_img_np.sum()
            eval_nonshadow_sum += (1 - mask_img_np).sum()
            eval_sum = eval_sum + mask_img_np.sum() + (1 - mask_img_np).sum()


    all_mse, s_mse, ns_mse = eval_rmse/eval_sum, eval_shadow_rmse/eval_shadow_sum, eval_nonshadow_rmse/eval_nonshadow_sum

    print(f'all: %.4f, shadow: %.4f, non-shadow: %.4f'%(all_mse, s_mse, ns_mse))
