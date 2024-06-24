import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
import json
import torch
import torch.nn as nn
import torch.optim as optim

from utils import clip_gradient, AvgMeter
from torch.autograd import Variable
from datetime import datetime
import torch.nn.functional as F

from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf

from mmseg import __version__
from mmseg.models.segmentors import BiRAFormer as UNet

from PIL import Image
# from keras.utils.np_utils import to_categorical   
# Convert numpy data to tensorflow data
'''
We need segent 3 classes:
    + 0 if the pixel is part of the image background (denoted by black color);
    + 1 if the pixel is part of a non-neoplastic polyp (denoted by green color);
    + 2 if the pixel is part of a neoplastic polyp (denoted by red color).
'''
import random
import imgaug
from imgaug import augmenters as iaa
# from keras.preprocessing.image import ImageDataGenerator
HEIGHT = 384

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')
torch.cuda.set_device(device)

sometimes = lambda aug: iaa.Sometimes(.5, aug)   
aug_pipe = iaa.Sequential(
            [      


                iaa.SomeOf((0, 3),
                    [
                       # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                        iaa.OneOf([
                           iaa.GaussianBlur((0, 1.5)), # blur images with a sigma between 0 and 3.0
                           iaa.AverageBlur(k=(2,5)), # blur image using local means with kernel sizes between 2 and 7
                           iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 2 and 7
                        ]),
                        iaa.Sharpen(alpha=(0, 0.02), lightness=(0.95, 1.05)), # sharpen images
                        imgaug.augmenters.blur.MotionBlur(k=(3, 7), angle=(0, 360)),
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.02*255), per_channel=0.5), # add gaussian noise to images
                       
                        
                        iaa.Add((-5, 5), per_channel=0.5), 
                        iaa.Multiply((0.95, 1.05), per_channel=0.5), 
                        iaa.ContrastNormalization((0.95, 1.05), per_channel=0.5), # improve or worsen the contrast
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )
image_datagen_args = {
		'shear_range': 0.1,
		'zoom_range': 0.2,
		'width_shift_range': 0.25,
		'height_shift_range': 0.25,
		'rotation_range': 180,
		'horizontal_flip': True,
		'vertical_flip': True,
        'fill_mode':'constant'
	}
# image_datagen = ImageDataGenerator(**image_datagen_args)
# def augment(image,mask):
#     #image *= 255
#     image = image.astype(np.uint8)
#     if random.random()<0.5:
#         seed = random.randint(0,1000000000)
#         params = image_datagen.get_random_transform(image.shape,seed = seed)
#         image = image_datagen.apply_transform(image, params)
#         params = image_datagen.get_random_transform(mask.shape,seed = seed)
#         mask = image_datagen.apply_transform(np.expand_dims(mask,-1), params)[:,:,0]
#     if random.random()<0.5:
#         image = aug_pipe.augment_image(np.array(image).astype(np.uint8)) 
#     #image = image/255.
#     return image.astype(np.float32),mask
def read_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (HEIGHT, HEIGHT))
   # image = image/255.0
  #  image = image.astype(np.float32)
    return image   
def read_mask(mask_path):
    image = cv2.imread(mask_path)
    
    image = cv2.resize(image, (HEIGHT, HEIGHT))
    """
    mask = np.zeros((HEIGHT,WIDTH))
    mask[image[:,:,1]>10]=1
    mask[image[:,:,2]>10]=2
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 100, 20])
    upper1 = np.array([10, 255, 255])
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160,100,20])
    upper2 = np.array([179,255,255])
    lower_mask = cv2.inRange(image, lower1, upper1)
    upper_mask = cv2.inRange(image, lower2, upper2)

    red_mask = lower_mask + upper_mask;
    red_mask[red_mask != 0] = 2
    
    # boundary RED color range values; Hue (36 - 70)
    green_mask = cv2.inRange(image, (36, 25, 25), (70, 255,255))
    green_mask[green_mask != 0] = 1
    
    full_mask = cv2.bitwise_or(red_mask, green_mask)
    full_mask = full_mask.astype(np.uint8)
    full_mask= cv2.dilate(full_mask, np.ones((5,5)), iterations=1)
    full_mask = cv2.erode(full_mask, np.ones((5,5)), iterations=1)  
    return full_mask.astype(np.uint8)
import random
# class NeoDataset(torch.utils.data.Dataset):

#     def __init__(self, img_paths, mask_paths, aug=True, transform=None):
#         self.img_paths = img_paths
#         self.mask_paths = mask_paths
#         self.aug = aug
#         self.transform = transform

#     def __len__(self):
#         return len(self.img_paths)

#     def __getitem__(self, idx):
      
#         img_path = self.img_paths[idx]
#         mask_path = self.mask_paths[idx]
#         # image = imread(img_path)
#         # mask = imread(mask_path)
#        # image = cv2.imread(img_path)
#        # print(img_path,mask_path)
#         image = read_image(img_path)# cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask = read_mask(mask_path)#cv2.imread(mask_path, 0)
#         # name = self.img_paths[idx].split('/')[-1]

#         if self.transform is not None:
            
#            # augmented = self.transform(image=image, mask=mask)
#             #image = augmented['image']
#            # mask = augmented['mask']
#             image,mask = augment(image,mask )
#             image = cv2.resize(image, (HEIGHT, HEIGHT))
#             mask = cv2.resize(mask, (HEIGHT, HEIGHT),interpolation=cv2.INTER_NEAREST) 
#         else:
#             image = cv2.resize(image, (HEIGHT, HEIGHT))
#             mask = cv2.resize(mask, (HEIGHT, HEIGHT),) 

#         image = image.astype('float32') / 255.
#         image = image.transpose((2, 0, 1))
        
#         #mask = mask[:,:,np.newaxis]
        
#         mask = to_categorical(mask, num_classes=3)
#         mask = mask.astype('float32')
#         mask = mask.transpose((2, 0, 1))
#        # mask[:,0,:,:]=0
#         #print(mask.shape)
#         return np.asarray(image), np.asarray(mask)
class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, img_paths, mask_paths, aug=True, transform=None, train_ratio=1.0, mode="train"):
        self.aug = aug
        self.transform = transform
        self.train_ratio = train_ratio
        self.path = "/mnt/quanhd/endoscopy/public_dataset.json"
        self.root_path = "/home/s/DATA/"
        self.mode = mode
        self.load_data_from_json()

    def __len__(self):
        return len(self.image_paths)
    
    def load_data_from_json(self):
        with open(self.path) as f:
            data = json.load(f)
        if self.mode == "train":
            all_image_paths = data[self.mode]["images"]
            kvasir_image_paths = []
            clinic_image_paths = []
            for image_path in all_image_paths:
                if "c" in image_path:
                    kvasir_image_paths.append(image_path)
                else:
                    clinic_image_paths.append(image_path)
        
            all_mask_paths = data[self.mode]["masks"]
            kvasir_mask_paths = []
            clinic_mask_paths = []
            for mask_path in all_mask_paths:
                if "c" in mask_path:
                    kvasir_mask_paths.append(mask_path)
                else:
                    clinic_mask_paths.append(mask_path)
            print(f"Pre len(all_image_paths) = {len(all_image_paths)}")
            print(f"Pre len(all_mask_paths) = {len(all_mask_paths)}")
            self.image_paths = kvasir_image_paths[:int(len(kvasir_image_paths)*self.train_ratio)] + clinic_image_paths[:int(len(clinic_image_paths)*self.train_ratio)]
            self.mask_paths = kvasir_mask_paths[:int(len(kvasir_mask_paths)*self.train_ratio)] + clinic_mask_paths[:int(len(clinic_mask_paths)*self.train_ratio)]
            print(f"After len(image_paths) = {len(self.image_paths)}")
            print(f"After len(mask_paths) = {len(self.mask_paths)}")
        elif self.mode == "test":
            self.image_paths = data[self.mode][self.ds_test]["images"]
            self.mask_paths = data[self.mode][self.ds_test]["masks"]

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_path, self.image_paths[idx])
        mask_path = os.path.join(self.root_path, self.mask_paths[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            image = cv2.resize(image, (384, 384))
            mask = cv2.resize(mask, (384, 384)) 
        else:
            image = cv2.resize(image, (384, 384))
            mask = cv2.resize(mask, (384, 384)) 

        image = image.astype('float32') / 255
        image = image.transpose((2, 0, 1))

        mask = mask[:,:,np.newaxis]
        mask = mask.astype('float32') / 255
        mask = mask.transpose((2, 0, 1))

        return np.asarray(image), np.asarray(mask)
    
epsilon = 1e-7

def recall_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon)
    return recall

def precision_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon)
    return precision

def dice_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+epsilon))

def iou_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return recall*precision/(recall+precision-recall*precision + epsilon)


class FocalLossV1(nn.Module):
    
    def __init__(self,
                alpha=0.25,
                gamma=2,
                reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        # compute loss
        logits = logits.float() # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wfocal = FocalLossV1()(pred, mask)
    wfocal = (wfocal*weit).sum(dim=(2,3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wfocal + wiou).mean()


def train(train_loader, model, optimizer, epoch, lr_scheduler, args):
    model.train()
    # ---- multi-scale training ----
    # size_rates = [0.75, 1, 1.25]
    size_rates = [256, 384, 512]
    loss_record = AvgMeter()
    dice, iou = AvgMeter(), AvgMeter()
    with torch.autograd.set_detect_anomaly(True):
        for i, pack in tqdm(enumerate(train_loader, start=1), total=len(train_loader)):
            if epoch <= 1:
                    optimizer.param_groups[0]["lr"] = (epoch * i) / (1.0 * total_step) * args.init_lr
            else:
                lr_scheduler.step()

            for rate in size_rates: 
                optimizer.zero_grad()
                # ---- data prepare ----
                images, gts = pack
                images = Variable(images).to(device)
                gts = Variable(gts).to(device)
                # ---- rescale ----
                trainsize = int(round(args.init_trainsize*rate/32)*32)
                trainsize = rate
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                # ---- forward ----
                map4, map3, map2, map1 = model(images)
                map1 = F.upsample(map1, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                map2 = F.upsample(map2, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                map3 = F.upsample(map3, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                map4 = F.upsample(map4, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                loss = structure_loss(map1, gts) + structure_loss(map2, gts) + structure_loss(map3, gts) + structure_loss(map4, gts)
                # with torch.autograd.set_detect_anomaly(True):
                #loss = nn.functional.binary_cross_entropy(map1, gts)
                # ---- metrics ----
                dice_score = dice_m(map4, gts)
                iou_score = iou_m(map4, gts)
                # ---- backward ----
                loss.backward()
                clip_gradient(optimizer, args.clip)
                optimizer.step()
                # ---- recording loss ----
                if rate == 1:
                    loss_record.update(loss.data, args.batchsize)
                    dice.update(dice_score.data, args.batchsize)
                    iou.update(iou_score.data, args.batchsize)

            # ---- train visualization ----
            if i == total_step:
                print('{} Training Epoch [{:03d}/{:03d}], '
                        '[loss: {:0.4f}, dice: {:0.4f}, iou: {:0.4f}]'.
                        format(datetime.now(), epoch, args.num_epochs,\
                                loss_record.show(), dice.show(), iou.show()))

    ckpt_path = save_path + 'last.pth'
    print('[Saving Checkpoint:]', ckpt_path)
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict()
    }
    torch.save(checkpoint, ckpt_path)
from albumentations.augmentations.geometric import  resize,rotate, Flip, Transpose
import albumentations.augmentations.crops.transforms as crop
import albumentations.augmentations.transforms as transforms
from albumentations.augmentations.blur import GaussianBlur
train_transform = Compose([
            rotate.RandomRotate90(),
            Flip(),
            transforms.HueSaturationValue(),
            transforms.RandomBrightnessContrast(),
            GaussianBlur(),
            Transpose(),
            OneOf([
                crop.RandomCrop(224, 224, p=1),
                crop.CenterCrop(224, 224, p=1)
            ], p=0.2),
            resize.Resize(384, 384)
        ], p=0.5)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int,
                        default=30, help='epoch number')
    parser.add_argument('--backbone', type=str,
                        default='b3', help='backbone version')
    parser.add_argument('--num_classes', type=int,
                        default=1, help='number output classes')
    parser.add_argument('--bottleneck', type=bool,
                        default=True, help='use bottle neck in reverse attention or not')
    parser.add_argument('--neo', type=bool,
                        default=False, help='use Neo Reverse Attention or not (Softmax Reverse Attentions)')
    parser.add_argument('--init_lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=4, help='training batch size')
    parser.add_argument('--init_trainsize', type=int,
                        default=384, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--train_path', type=str,
                        default='./data/TrainDataset', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='RaBiTB3')
    parser.add_argument('--resume_path', type=str, help='path to checkpoint for resume training',
                        default='')
    args = parser.parse_args()

    save_path = 'snapshots/{}/'.format(args.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    else:
        print("Save path existed")

    train_img_paths = []
    train_mask_paths = []
    train_img_paths = glob('{}/images/*'.format(args.train_path))
    train_mask_paths = glob('{}/masks/*'.format(args.train_path))
    train_img_paths.sort()
    train_mask_paths.sort()
    if args.num_classes ==1:
        train_dataset = Dataset(train_img_paths, train_mask_paths, transform=train_transform, train_ratio=0.5)
    # elif args.num_classes ==3:
    #     train_dataset = NeoDataset(train_img_paths, train_mask_paths, transform =train_transform )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    total_step = len(train_loader)

    model = UNet(backbone=dict(
                    type='mit_{}'.format(args.backbone),
                    style='pytorch'), 
                decode_head=None,
                neck=None,
                auxiliary_head=None,
                train_cfg=dict(),
                num_classes=args.num_classes,
                compound_coef=4,
                neo=args.neo,numrepeat = 4,bottleneck=args.bottleneck,
                test_cfg=dict(mode='whole'),
                pretrained='pretrained/mit_{}.pth'.format(args.backbone)).to(device)

    # ---- flops and params ----
    params = model.parameters()
    optimizer = torch.optim.Adam(params, args.init_lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                        T_max=len(train_loader)*args.num_epochs,
                                        eta_min=args.init_lr/1000)

    start_epoch = 1
    if args.resume_path != '':
        checkpoint = torch.load(args.resume_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    print("#"*20, "Start Training", "#"*20)
    for epoch in range(start_epoch, args.num_epochs+1):
        train(train_loader, model, optimizer, epoch, lr_scheduler, args)