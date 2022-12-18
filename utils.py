import os, shutil
from PIL import Image
import numpy as np
import torch
from skimage.color import rgb2lab

def vis(s, f, m, name):
    Image.fromarray(s).save(name)

def mae_labspace(s, f, m, eval_size=(480, 640), name=None):
    '''
    s: predicted shadow image
    f: ground truth shadaow free image
    m: shadow mask image
    '''
    s = (s.detach().cpu().squeeze().permute(1,2,0).numpy()*255).astype(np.uint8)
    f = (f.detach().cpu().squeeze().permute(1,2,0).numpy()*255).astype(np.uint8)
    m = (m.detach().cpu().squeeze().numpy()*255).astype(np.uint8)

    if name is not None:
        vis(s, f, m, name)

    summask = eval_size[0]*eval_size[1]
    
    smask = m.astype(bool).astype(np.uint8)
    nmask = 1 - smask

    f = f.astype(np.float32)/255
    s = s.astype(np.float32)/255

    f = rgb2lab(f)
    s = rgb2lab(s)

    #abs lab
    mae = np.abs(f - s)

    # rmse
    lab_mae = np.sum(mae, axis=(0,1))/summask
    
    # rmse in shadow, original way, per pixel
    dist = np.abs(f - s) * smask[:, :, np.newaxis].repeat(3, axis=-1)
    sdist, smask = dist.sum(), smask.sum()
    
    # rmse in non-shadow, original way, per pixel
    dist = np.abs(f - s) * nmask[:, :, np.newaxis].repeat(3, axis=-1)
    ndist, nmask = dist.sum(), nmask.sum()

    return lab_mae, sdist, float(smask), ndist, float(nmask)

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
            # image_numpy = image_numpy.convert('L')
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image

    return np.clip(image_numpy, 0, 255).astype(imtype)

def calc_RMSE(real_img, fake_img):
    # convert to LAB color space
    real_lab = rgb2lab(real_img)
    fake_lab = rgb2lab(fake_img)
    return abs(real_lab - fake_lab)

def backup_code(save_dir, dirs_to_save = ['dataset', 'model']):
    fs = os.listdir('./')
    os.makedirs(save_dir, exist_ok=True)
    save_dir = os.path.join(save_dir, 'code')
    os.makedirs(save_dir, exist_ok=True)

    # save dirs
    for dts in dirs_to_save:
        d = os.path.join(save_dir, dts)
        if os.path.exists(d):
            shutil.rmtree(d)

        shutil.copytree(dts, d)

    # save files
    for f in fs:
        if not os.path.isdir(f) and (f.endswith('.py') or f.endswith('.sh')):
            shutil.copy(f, os.path.join(save_dir, f))