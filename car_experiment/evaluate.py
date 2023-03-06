import os.path

import torch
import torch.nn.functional as F
from tqdm import tqdm
import cv2 as cv
from PIL import Image
import numpy as np
from utils.dice_score import multiclass_dice_coeff, dice_coeff


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


def evaluate(net, dataloader, device, model_name):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            if model_name == 'unet++':
                mask_pred = mask_pred[-1]
            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches


def evaluate_val(net, dataloader, device, outfilepath):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    i = 0
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                # unet++
                # mask_pred = mask_pred[-1]
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)
        m_true = mask_to_image(mask_true[0].float().cpu().numpy())
        m_true.save(os.path.join(outfilepath, 'true_masks\\%d.png' % (i)))
        m_pred = mask_to_image(mask_pred.argmax(dim=1)[0].float().cpu().numpy())
        m_pred.save(os.path.join(outfilepath, 'pred_masks\\%d.png' % (i)))
        img = Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8).reshape(640, 960, 3))
        img.save(os.path.join(outfilepath, 'raw_image/%d.jpg' % (i)))

        i = i + 1
    # Fixes a potential division by zero error

    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches