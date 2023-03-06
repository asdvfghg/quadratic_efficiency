import torch.utils.data as data
import PIL.Image as Image
from sklearn.model_selection import train_test_split
import os
import cv2
from glob import glob
import imageio
import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class LiverDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.train_root = r"data\liver\train"
        self.val_root = r"data\liver\val"
        self.test_root = self.val_root
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        assert self.state =='train' or self.state == 'val' or self.state =='test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root
        if self.state == 'test':
            root = self.test_root

        pics = []
        masks = []
        n = len(os.listdir(root)) // 2  # 因为数据集中一套训练数据包含有训练图和mask图，所以要除2
        for i in range(n):
            img = os.path.join(root, "%03d.png" % i)  # liver is %03d
            mask = os.path.join(root, "%03d_mask.png" % i)
            pics.append(img)
            masks.append(mask)
            #imgs.append((img, mask))
        return pics,masks

    def __getitem__(self, index):
        #x_path, y_path = self.imgs[index]
        x_path = self.pics[index]
        y_path = self.masks[index]
        origin_x = Image.open(x_path)
        origin_y = Image.open(y_path)
        # origin_x = cv2.imread(x_path)
        # origin_y = cv2.imread(y_path,cv2.COLOR_BGR2GRAY)
        if self.transform is not None:
            img_x = self.transform(origin_x)
        if self.target_transform is not None:
            img_y = self.target_transform(origin_y)
        return img_x, img_y,x_path,y_path

    def __len__(self):
        return len(self.pics)

class esophagusDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.train_root = r"E:\datasets\data_sta_all\train_data"
        self.val_root = r"E:\datasets\data_sta_all\test_data"
        self.test_root = self.val_root
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        assert self.state =='train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root
        if self.state == 'test':
            root = self.test_root
        pics = []
        masks = []
        n = len(os.listdir(root)) // 2  # 因为数据集中一套训练数据包含有训练图和mask图，所以要除2
        for i in range(n):
            img = os.path.join(root, "%05d.png" % i)  # liver is %03d
            mask = os.path.join(root, "%05d_mask.png" % i)
            pics.append(img)
            masks.append(mask)
            #imgs.append((img, mask))
        return pics,masks

    def __getitem__(self, index):
        #x_path, y_path = self.imgs[index]
        x_path = self.pics[index]
        y_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        origin_x = cv2.imread(x_path)
        origin_y = cv2.imread(y_path,cv2.COLOR_BGR2GRAY)
        if self.transform is not None:
            img_x = self.transform(origin_x)
        if self.target_transform is not None:
            img_y = self.target_transform(origin_y)
        return img_x, img_y,x_path,y_path

    def __len__(self):
        return len(self.pics)

class dsb2018CellDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'data\dsb2018_256'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths = None,None
        self.train_mask_paths, self.val_mask_paths = None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.img_paths = glob(self.root + '\images\*')
        self.mask_paths = glob(self.root + '\masks\*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        self.test_img_paths, self.val_img_paths, self.test_mask_paths, self.val_mask_paths = \
            train_test_split(self.val_img_paths, self.val_mask_paths, test_size=0.5, random_state=41)
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths,self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths,self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths,self.test_mask_paths

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)

class CornealDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'E:\datasets\CORN\CORN\Corneal nerve curivilinear segmentation\Corneal nerve curivilinear segmentation'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.train_img_paths = glob(self.root + r'\training\train_images\*')
        self.train_mask_paths = glob(self.root + r'\training\train_mask\*')
        self.val_img_paths = glob(self.root + r'\val\val_images\*')
        self.val_mask_paths = glob(self.root + r'\val\val_mask\*')
        self.test_img_paths = glob(self.root + r'\test\test_images\*')
        self.test_mask_paths = glob(self.root + r'\test\test_mask\*')
        # self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
        #     train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths,self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths,self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths,self.test_mask_paths

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)

class DriveEyeDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'E:\datasets\DRIVE\DRIVE'
        self.pics, self.masks = self.getDataPath()
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.train_img_paths = glob(self.root + r'\training\images\*')
        self.train_mask_paths = glob(self.root + r'\training\1st_manual\*')
        self.val_img_paths = glob(self.root + r'\test\images\*')
        self.val_mask_paths = glob(self.root + r'\test\1st_manual\*')
        self.test_img_paths = self.val_img_paths
        self.test_mask_paths = self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths, self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths, self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths, self.test_mask_paths

    def __getitem__(self, index):
        imgx,imgy=(576,576)
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        #print(pic_path)
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        if mask == None:
            mask = imageio.mimread(mask_path)
            mask = np.array(mask)[0]
        pic = cv2.resize(pic,(imgx,imgy))
        mask = cv2.resize(mask, (imgx, imgy))
        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)

class IsbiCellDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'E:\datasets\isbi'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.img_paths = glob(self.root + r'\train\images\*')
        self.mask_paths = glob(self.root + r'\train\label\*')
        # self.val_img_paths = glob(self.root + r'\val\val_images\*')
        # self.val_mask_paths = glob(self.root + r'\val\val_mask\*')
        # self.test_img_paths = glob(self.root + r'\test\test_images\*')
        # self.test_mask_paths = glob(self.root + r'\test\test_mask\*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        self.test_img_paths, self.test_mask_paths = self.val_img_paths,self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths,self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths,self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths,self.test_mask_paths

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)

class LungKaggleDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'E:\datasets\finding-lungs-in-ct-data-kaggle'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths,self.test_img_paths = None,None,None
        self.train_mask_paths, self.val_mask_paths,self.test_mask_paths = None,None,None
        self.pics,self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.img_paths = glob(self.root + r'\2d_images\*')
        self.mask_paths = glob(self.root + r'\2d_masks\*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=41)
        self.test_img_paths, self.test_mask_paths = self.val_img_paths, self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths, self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths, self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths, self.test_mask_paths

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)
        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y,pic_path,mask_path

    def __len__(self):
        return len(self.pics)



class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')