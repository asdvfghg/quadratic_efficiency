import argparse
import logging
import os.path
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet.qunet import QUNet
from unet.r2unet import R2U_Net
from utils.data_loading import BasicDataset, CarvanaDataset, CrackDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet
#
dir_img = Path('../autodl-tmp/train/train')
dir_mask = Path('../autodl-tmp/train_masks/train_masks')
dir_checkpoint = Path('checkpoints')

# dir_img = Path('data/traincrop')
# dir_mask = Path('data/train_masks_crop')
# dir_checkpoint = Path('checkpoints')

# dir_img = Path('data/carvana-image-masking-challenge/train')
# dir_mask = Path('data/carvana-image-masking-challenge/train_masks')
# dir_checkpoint = Path('checkpoints')


def group_parameters(m):
    weight_r, weight_g, weight_b, bias_r, bias_g, bias_b, w, b = [], [], [], [], [], [], [], []
    for name, p in m.named_parameters():
        if 'weight_r' in name:
            weight_r += [p]
        if 'weight_g' in name:
            weight_g += [p]
        if 'weight_b' in name:
            weight_b += [p]
        if 'bias_r' in name:
            bias_r += [p]
        if 'bias_g' in name:
            bias_g += [p]
        if 'bias_b' in name:
            bias_b += [p]
        if 'weight' in name[-6:]:
            w += [p]
        if 'bias' in name[-4:]:
            b += [p]

    return (weight_r, weight_g, weight_b, bias_r, bias_g, bias_b, w, b)

def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-4,
              sub_learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False,
              model_name: str = 'unet',
              dataset_name: str = 'car'):
    # 1. Create dataset
    try:
        if dataset_name == 'car':
            dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
        if dataset_name == 'crack':
            dataset = CrackDataset(dir_img, dir_mask, img_scale)

    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True,  **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    if model_name == 'unet':
        optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    if model_name == 'unet++':
        optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    if model_name == 'r2unet':
        optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    if model_name == 'qunet':
        group = group_parameters(net)
        optimizer = optim.AdamW([{"params": group[0], "lr": learning_rate},  # weight_r
                                     {"params": group[1], "lr": sub_learning_rate * learning_rate},  # weight_g
                                     {"params": group[2], "lr": sub_learning_rate * learning_rate},  # weight_b
                                     {"params": group[3], "lr": learning_rate},  # bias_r
                                     {"params": group[4], "lr": sub_learning_rate * learning_rate},  # bias_g
                                     {"params": group[5], "lr": sub_learning_rate * learning_rate},  # bias_b
                                     {"params": group[6], "lr": learning_rate},
                                     {"params": group[7], "lr": learning_rate}],
                                    lr=learning_rate, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)  # goal: maximize Dice score

    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                loss = 0.0
                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    if model_name == 'unet++':
                        for output in masks_pred:
                            loss += criterion(output, true_masks)
                        loss /= len(masks_pred)
                    else:
                        loss = criterion(masks_pred, true_masks) \
                               + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                           F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                           multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                if model_name == 'qunet':
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.01)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            # histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, val_loader, device, model_name)
                        scheduler.step(val_score)
                        if model_name == 'unet++':
                            masks_pred = masks_pred[-1]
                        logging.info('Validation Dice score: {}'.format(val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)

            torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'checkpoint_epoch{}.pth'.format(epoch)))
            wandb.save('*.pth')
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--sub learning rate' '-slr', metavar='SLR', type=float, default=1e-4,
                        help='Sub Learning rate', dest='slr')
    parser.add_argument('--model', '-m', metavar='MODEL', type=str, default='r2unet',
                        help='Model Net')
    parser.add_argument('--dataset', '-d', metavar='DATASET', type=str, default='car',
                        help='DataSet Name')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    if args.model == 'unet':
        net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    if args.model == 'qunet':
        net = QUNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    if args.model == 'unet++':
        net = NestedUNet(True, n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    if args.model == 'r2unet':
        net = R2U_Net(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  sub_learning_rate=args.slr,
                  save_checkpoint=True,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp,
                  model_name=args.model,
                  dataset_name=args.dataset)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
