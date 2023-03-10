'''
author:zhujunwen
Guangdong University of Technology
'''
import argparse
import logging
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch import autograd, optim, nn
from tqdm import tqdm

from UNet import Unet,resnet34_unet
from attention_unet import AttU_Net
from channel_unet import myChannelUnet
from qunet import QUNet
from r2unet import R2U_Net
from unetpp import NestedUNet
from dataset import *
from metrics import *
from torchvision.transforms import transforms
import torch.nn.functional as F

from torchvision.models import vgg16

from utils.dice_score import dice_loss

dir_img = Path('data/train')
dir_mask = Path('data/train_masks')
dir_checkpoint = Path('checkpoints')

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

def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--deepsupervision', default=0)
    parse.add_argument("--action", type=str, help="train/test/train&test", default="train&test")
    parse.add_argument("--epoch", type=int, default=20)
    parse.add_argument('--arch', '-a', metavar='ARCH', default='qunet',
                       help='UNet/resnet34_unet/unet++/myChannelUnet/Attention_UNet/segnet/r2unet/fcn32s/fcn8s')
    parse.add_argument("--batch_size", type=int, default=8)
    parse.add_argument('--dataset', default='dsb2018Cell',  # dsb2018_256
                       help='dataset name:liver/esophagus/dsb2018Cell/corneal/driveEye/isbiCell/kaggleLung')
    parse.add_argument('--learning_rate', '-l', type=float, default=1e-3)
    parse.add_argument('--sub_learning_rate', '-slr', type=float, default=1e-4)
    # parse.add_argument("--ckp", type=str, help="the path of model weight file")
    parse.add_argument("--log_dir", default='result/log', help="log dir")
    parse.add_argument("--threshold",type=float,default=None)
    parse.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parse.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    args = parse.parse_args()
    return args

def getLog(args):
    dirname = os.path.join(args.log_dir,args.arch,str(args.batch_size),str(args.dataset),str(args.epoch))
    filename = dirname +'/log.log'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logging.basicConfig(
            filename=filename,
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
    return logging

def getModel(args):
    if args.arch == 'UNet':
        model = Unet(3, 1).to(device)
    if args.arch == 'resnet34_unet':
        model = resnet34_unet(1,pretrained=False).to(device)
    if args.arch == 'unet++':
        args.deepsupervision = True
        model = NestedUNet(args,3,1).to(device)
    if args.arch =='Attention_UNet':
        model = AttU_Net(3,1).to(device)
    if args.arch == 'segnet':
        model = SegNet(3,1).to(device)
    if args.arch == 'r2unet':
        model = R2U_Net(3,1).to(device)
    if args.arch == 'fcn32s':
        model = get_fcn32s(1).to(device)
    if args.arch == 'myChannelUnet':
        model = myChannelUnet(3,1).to(device)
    if args.arch == 'fcn8s':
        assert args.dataset !='esophagus' ,"fcn8s???????????????????????????esophagus?????????esophagus????????????80x80?????????5??????2?????????????????????2.5x2.5????????????????????????????????????????????????resize??????????????????????????????fcn"
        model = get_fcn8s(1).to(device)
    if args.arch == 'cenet':
        from cenet import CE_Net_
        model = CE_Net_().to(device)
    if args.arch == 'qunet':
        model = QUNet(3, 1).to(device)
    return model

def getDataset(args):
    train_dataloaders, val_dataloaders ,test_dataloaders= None,None,None
    if args.dataset =='liver':  #E:\??????\new\u_net_liver-master\data\liver\val
        train_dataset = LiverDataset(r"train", transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = LiverDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataloaders = val_dataloaders
    if args.dataset =="esophagus":
        train_dataset = esophagusDataset(r"train", transform=x_transforms,target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = esophagusDataset(r"val", transform=x_transforms,target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataloaders = val_dataloaders
    if args.dataset == "dsb2018Cell":
        train_dataset = dsb2018CellDataset(r"train", transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = dsb2018CellDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = dsb2018CellDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if args.dataset == 'corneal':
        train_dataset = CornealDataset(r'train',transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = CornealDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = CornealDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if args.dataset == 'driveEye':
        train_dataset = DriveEyeDataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = DriveEyeDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = DriveEyeDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if args.dataset == 'isbiCell':
        train_dataset = IsbiCellDataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = IsbiCellDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = IsbiCellDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if args.dataset == 'kaggleLung':
        train_dataset = LungKaggleDataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = LungKaggleDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = LungKaggleDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if args.dataset == 'car':
        dataset = CarvanaDataset(dir_img, dir_mask, args.scale)
        n_val = int(len(dataset) * args.val / 100)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

        loader_args = dict(batch_size=args.batch_size, num_workers=0, pin_memory=False)
        train_dataloaders = DataLoader(train_set, shuffle=True, **loader_args)
        val_dataloaders = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    return train_dataloaders, val_dataloaders, test_dataloaders

def val(model,best_iou,val_dataloaders):
    model= model.eval()
    with torch.no_grad():
        i=0   #???????????????i??????
        miou_total = 0
        hd_total = 0
        dice_total = 0
        num = len(val_dataloaders)  #????????????????????????
        #print(num)
        for x, _,pic,mask in val_dataloaders:
            x = x.to(device)
            y = model(x)
            if args.deepsupervision:
                img_y = torch.squeeze(y[-1]).cpu().numpy()
            else:
                img_y = torch.squeeze(y).cpu().numpy()  #?????????????????????????????????????????????numpy??????????????????????????????????????????????????????????????????batchsize

            hd_total += get_hd(mask[0], img_y)
            miou_total += get_iou(mask[0],img_y)  #????????????????????????miou???????????????miou???
            dice_total += get_dice(mask[0],img_y)
            if i < num:i+=1   #???????????????????????????
        aver_iou = miou_total / num
        aver_hd = hd_total / num
        aver_dice = dice_total/num
        print('Miou=%f,aver_hd=%f,aver_dice=%f' % (aver_iou,aver_hd,aver_dice))
        logging.info('Miou=%f,aver_hd=%f,aver_dice=%f' % (aver_iou,aver_hd,aver_dice))
        if aver_iou > best_iou:
            print('aver_iou:{} > best_iou:{}'.format(aver_iou,best_iou))
            logging.info('aver_iou:{} > best_iou:{}'.format(aver_iou,best_iou))
            logging.info('===========>save best model!')
            best_iou = aver_iou
            print('===========>save best model!')
            torch.save(model.state_dict(), r'./saved_model/'+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'.pth')
        return best_iou,aver_iou,aver_dice,aver_hd

def train(model, criterion, optimizer, train_dataloader,val_dataloader, args):
    best_iou,aver_iou,aver_dice,aver_hd = 0,0,0,0
    num_epochs = args.epoch
    threshold = args.threshold
    loss_list = []
    iou_list = []
    dice_list = []
    hd_list = []
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-8)  # goal: maximize Dice score
    for epoch in range(num_epochs):
        model = model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        with tqdm(total=dt_size, desc=f'Epoch {epoch}/{num_epochs}', unit='img') as pbar:
            for x, y,_, mask in train_dataloader:
            # for batch in train_dataloader:
                step += 1
                inputs = x.to(device)
                labels = y.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                if args.arch == 'qunet':
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
                if args.deepsupervision:
                    outputs = model(inputs)
                    loss = 0
                    for output in outputs:
                        loss += criterion(output, labels)
                    loss /= len(outputs)
                else:
                    output = model(inputs)
                    loss = criterion(output, labels)
                if threshold!=None:
                    if loss > threshold:
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                else:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                # print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))
                logging.info("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))
                pbar.update(x.shape[0])
                pbar.set_postfix({'loss (batch)': loss.item()})
            # scheduler.step(epoch_loss)

        best_iou, aver_iou, aver_dice, aver_hd = val(model, best_iou, val_dataloader)
        loss_list.append(epoch_loss)
        iou_list.append(aver_iou)
        dice_list.append(aver_dice)
        hd_list.append(aver_hd)
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        logging.info("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    loss_plot(args, loss_list)
    metrics_plot(args, 'iou&dice',iou_list, dice_list)
    metrics_plot(args,'hd',hd_list)
    return model


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))

def test(val_dataloaders,save_predict=False):
    logging.info('final test........')
    if save_predict ==True:
        dir = os.path.join(r'./saved_predict',str(args.arch),str(args.batch_size),str(args.epoch),str(args.dataset))
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print('dir already exist!')
    model.load_state_dict(torch.load(r'./saved_model/'+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'.pth', map_location='cpu'))  # ????????????????????????
    model.eval()

    if not os.path.exists(os.path.join(dir, 'raw_img')):
        os.makedirs(os.path.join(dir, 'raw_img'))
    if not os.path.exists(os.path.join(dir, 'pred_masks')):
        os.makedirs(os.path.join(dir, 'pred_masks'))
    if not os.path.exists(os.path.join(dir, 'true_masks')):
        os.makedirs(os.path.join(dir, 'true_masks'))

    #plt.ion() #??????????????????
    with torch.no_grad():
        i=0   #???????????????i??????
        miou_total = 0
        hd_total = 0
        dice_total = 0
        num = len(val_dataloaders)  #????????????????????????
        for pic,_,pic_path,mask_path in val_dataloaders:
            pic = pic.to(device)
            predict = model(pic)
            if args.deepsupervision:
                predict = torch.squeeze(predict[-1]).cpu().numpy()
            else:
                predict = torch.squeeze(predict).cpu().numpy()  #?????????????????????????????????????????????numpy??????????????????????????????????????????????????????????????????batchsize
            #img_y = torch.squeeze(y).cpu().numpy()  #?????????????????????????????????????????????numpy??????????????????????????????????????????????????????????????????batchsize

            iou = get_iou(mask_path[0],predict)
            miou_total += iou  #????????????????????????miou???????????????miou???
            hd_total += get_hd(mask_path[0], predict)
            dice = get_dice(mask_path[0],predict)
            dice_total += dice

            # fig = plt.figure()
            # ax1 = fig.add_subplot(1, 3, 1)
            # ax1.set_title('input')
            # plt.imshow(Image.open(pic_path[0]))
            # #print(pic_path[0])
            # ax2 = fig.add_subplot(1, 3, 2)
            # ax2.set_title('predict')
            # plt.imshow(predict,cmap='Greys_r')
            # ax3 = fig.add_subplot(1, 3, 3)
            # ax3.set_title('mask')
            # plt.imshow(Image.open(mask_path[0]), cmap='Greys_r')

            raw_img = Image.open(pic_path[0])
            pred_mask = mask_to_image(predict)
            true_mask = Image.open(mask_path[0])

            #print(mask_path[0])
            if save_predict == True:
                if args.dataset == 'driveEye':
                    saved_predict = dir + '/' + mask_path[0].split('\\')[-1]
                    saved_predict = '.'+saved_predict.split('.')[1] + '.tif'
                    plt.savefig(saved_predict)
                else:
                    # plt.savefig(dir +'/'+ mask_path[0].split('\\')[-1])


                    raw_img.save(os.path.join(dir, 'raw_img\\%s' % (pic_path[0].split('\\')[-1])))
                    pred_mask.save(os.path.join(dir, 'pred_masks\\%s' % (mask_path[0].split('\\')[-1])))
                    true_mask.save(os.path.join(dir, 'true_masks\\%s' % (mask_path[0].split('\\')[-1])))
            #plt.pause(0.01)
            print('iou={},dice={}'.format(iou,dice))
            if i < num:i+=1   #???????????????????????????
        #plt.show()
        print('Miou=%f,aver_hd=%f,dv=%f' % (miou_total/num,hd_total/num,dice_total/num))
        logging.info('Miou=%f,aver_hd=%f,dv=%f' % (miou_total/num,hd_total/num,dice_total/num))
        #print('M_dice=%f' % (dice_total / num))

if __name__ =="__main__":
    x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
    ])

    # mask??????????????????tensor
    y_transforms = transforms.ToTensor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args = getArgs()
    logging = getLog(args)
    print('**************************')
    print('models:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s' % \
          (args.arch, args.epoch, args.batch_size,args.dataset))
    logging.info('\n=======\nmodels:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s\n========' % \
          (args.arch, args.epoch, args.batch_size,args.dataset))
    print('**************************')
    model = getModel(args)
    train_dataloaders,val_dataloaders,test_dataloaders = getDataset(args)
    criterion = nn.BCELoss()
    if args.arch == 'qunet':
        group = group_parameters(model)
        optimizer = optim.AdamW([{"params": group[0], "lr": args.learning_rate},  # weight_r
                                 {"params": group[1], "lr": args.sub_learning_rate * args.learning_rate},  # weight_g
                                 {"params": group[2], "lr": args.sub_learning_rate * args.learning_rate},  # weight_b
                                 {"params": group[3], "lr": args.learning_rate},  # bias_r
                                 {"params": group[4], "lr": args.sub_learning_rate * args.learning_rate},  # bias_g
                                 {"params": group[5], "lr": args.sub_learning_rate * args.learning_rate},  # bias_b
                                 {"params": group[6], "lr": args.learning_rate},
                                 {"params": group[7], "lr": args.learning_rate}],
                                lr=args.learning_rate, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-8)


    if 'train' in args.action:
        train(model, criterion, optimizer, train_dataloaders,val_dataloaders, args)
    if 'test' in args.action:
        test(test_dataloaders, save_predict=True)