import argparse

def get_args(mode='train'):
    # Arguments
    parser = argparse.ArgumentParser(description='ShadowRemoval', fromfile_prefix_chars='@', conflict_handler='resolve')
    parser.add_argument('--model_name', default='accv2022', type=str, help='model name to choose')
    parser.add_argument('--seed', default=2021, type=int, help='random seed for reproducing results')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')

    parser.add_argument('--lr', '--learning-rate', default=0.000375, type=float, help='max learning rate')
    parser.add_argument('--lr_steps', default='20,40', type=str, help='number of total epochs to run')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, help='weight decay')

    parser.add_argument('--pct_start', default=0.10, type=float, help='The percentage of the cycle (in number of steps) spent increasing the learning rate. Default: 0.3')
    parser.add_argument('--div-factor', '--div_factor', default=30, type=float, help="Determines the initial learning rate via initial_lr = max_lr/div_factor Default: 25")
    parser.add_argument('--final-div-factor', '--final_div_factor', default=150, type=float, help="Determines the minimum learning rate via min_lr = initial_lr/final_div_factor Default: 1e4")

    # amp
    parser.add_argument('--amp', action='store_true', help='whether use mixed precision')
    
    parser.add_argument('--bs', default=8, type=int, help='batch size')
    parser.add_argument('--n_workers', default=8, type=int, help='num of workers')
    # tag
    parser.add_argument('--tag', default='default', type=str, help='tag for one experiment')
    # size
    parser.add_argument('--scale_size', default=448, type=int, help='resize width and height for input image')
    parser.add_argument('--crop_size', default=256, type=int, help='resize width and height for input image')

    parser.add_argument('--gpus', default='0', type=str, help='single gpu id')
    parser.add_argument('--backbone', default='b5', type=str, help='backbone name')
    parser.add_argument('--mode', default=mode, type=str, help='current mode')
    
    parser.add_argument('--dataset_name', default='istd', type=str, choices=['istd', 'aistd', 'srd'], help='which dataset')
    parser.add_argument('--root_istd', default='./data/ISTD_Dataset', type=str, help='ISTD dataset root directory')
    parser.add_argument('--root_aistd', default='./data/AISTD_Dataset', type=str, help='adjusted ISTD dataset root directory')
    parser.add_argument('--root_srd', default='./data/SRD', type=str, help='SRD dataset root directory')
    
    parser.add_argument("--save_path", default='./multiscale_{}_{}/', type=str, help="path to save checkpoints")

    args = parser.parse_args()

    return args
