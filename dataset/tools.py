import os, random
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', ]

def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    imname = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                imname.append(fname)

    return imname


def default_loader(path):
    return Image.open(path)

    
class ImageFolders(data.Dataset):
    def __init__(self, mode, root, scale_size, crop_size, subdirs=None, dict_names=None, transform=None, return_name=False, loader=default_loader):
        self.mode = mode
        self.scale_size = scale_size
        self.crop_size = crop_size
        self.n_subfolder, self.dict_names = len(subdirs), dict_names
        self.image_folders = [os.path.join(root, sd) for sd in subdirs]
        self.names = make_dataset(self.image_folders[0])
        if len(self.names) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.transform = transform
        self.return_name = return_name
        self.loader = loader

        print(f'>>> Total {len(self.names)} images found for stage {mode}')

    def __getitem__(self, index):
        elems = {}
        #try:
        name = self.names[index]
        for i in range(self.n_subfolder):
            path = os.path.join(self.image_folders[i], name)
            img = self.loader(path)
            elems[self.dict_names[i]] = img
        
        if self.mode=='train':
            # random crop
            elems = self.random_crop(elems, self.crop_size)
            # resize
            elems = self.resize(elems, self.scale_size)
            # random flip
            elems = self.random_flip(elems)

        for k, v in elems.items():
            if self.transform[i] is not None:
                elems[k] = self.transform[i](elems[k])

        if self.return_name:
            elems['name'] = name
        #except Exception as e:
        #    print('error while read data items: ', str(e))

        return elems

    def __len__(self):
        return len(self.names)

    ### data augmentation
    def random_flip(self, elems):
        new_elems = {}
        # random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            for k, v in elems.items():
                new_elems[k] = elems[k].transpose(Image.FLIP_LEFT_RIGHT)

            return new_elems
        return elems

    def random_crop(self, elems, crop_size):
        new_elems = {}
        first_key = list(elems.keys())[0]
        orig_size = elems[first_key].size
        
        # random crop
        w0, h0 = orig_size[0], orig_size[1]
        w1, h1 = crop_size[0], crop_size[1]

        x = random.randint(0, w0 - w1)
        y = random.randint(0, h0 - h1)

        for k, v in elems.items():
            new_elems[k] = elems[k].crop((x, y, x+w1, y+h1))

        return new_elems
    
    def resize(self, elems, new_size):
        new_elems = {}
        for k, v in elems.items():
            new_elems[k] = elems[k].resize(new_size)

        return new_elems