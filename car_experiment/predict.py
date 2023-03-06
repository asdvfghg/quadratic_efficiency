import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from evaluate import evaluate, evaluate_val
from unet.NestedUNet import NestedUNet
from unet.qunet import QUNet
from unet.r2unet import R2U_Net
from utils.data_loading import BasicDataset, CarvanaDataset
from unet import UNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        # unet++
        # output = output[-1]
        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='checkpoints_R2UNET/checkpoint_epoch5.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    # parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    # parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    # parser.add_argument('--viz', '-v', action='store_true',
    #                     help='Visualize the images as they are processed')
    # parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
    args = get_args()
    # in_files = args.input
    # out_files = get_output_filenames(args)
    dir_img = 'data/carvana-image-masking-challenge/train'
    dir_mask = 'data/carvana-image-masking-challenge/train_masks'
    output = 'results'
    val_percent = 0.1
    dataset = CarvanaDataset(dir_img, dir_mask, args.scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    val_loader =  DataLoader(val_set, shuffle=False, drop_last=True,  batch_size=1, num_workers=0, pin_memory=False)
    # net = QUNet(n_channels=3, n_classes=2, bilinear=args.bilinear)
    # net = UNet(n_channels=3, n_classes=2, bilinear=args.bilinear)
    # net = NestedUNet(True, n_channels=3, n_classes=2, bilinear=args.bilinear)
    net = R2U_Net(n_channels=3, n_classes=2, bilinear=args.bilinear)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    val_score = evaluate_val(net, val_loader, device, output)


    logging.info('Model loaded!')

    #
    # for i, filename in enumerate(in_files):
    #     logging.info(f'\nPredicting image {filename} ...')
    #     img = Image.open(filename)


        # mask = predict_img(net=net,
        #                    full_img=img,
        #                    scale_factor=args.scale,
        #                    out_threshold=args.mask_threshold,
        #                    device=device)

        # if not args.no_save:
        #     out_filename = out_files[i]
        #     result = mask_to_image(mask)
        #     result.save(out_filename)
        #     logging.info(f'Mask saved to {out_filename}')
        #
        # if args.viz:
        #     logging.info(f'Visualizing results for image {filename}, close to continue...')
        #     plot_img_and_mask(img, mask)
