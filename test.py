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
import torch
import torch.nn.functional as F
import json
from mmseg import __version__
from mmseg.models.segmentors import BiRAFormer as UNet
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# class Dataset(torch.utils.data.Dataset):
    
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
#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask = cv2.imread(mask_path, 0)

#         if self.transform is not None:
#             augmented = self.transform(image=image, mask=mask)
#             image = augmented['image']
#             mask = augmented['mask']
#         else:
#             image = cv2.resize(image, (384, 384))
#             mask = cv2.resize(mask, (384, 384)) 

#         image = image.astype('float32') / 255
#         image = image.transpose((2, 0, 1))

#         mask = mask[:,:,np.newaxis]
#         mask = mask.astype('float32') / 255
#         mask = mask.transpose((2, 0, 1))

#         return np.asarray(image), np.asarray(mask)

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self,  img_paths, mask_paths, aug=True, transform=None, train_ratio=1.0, mode="train", type="ung_thu_da_day_20230620"):
        self.aug = aug
        self.transform = transform
        self.train_ratio = train_ratio
        self.path = "/root/quanhd/endoscopy/ft_ton_thuong.json"
        self.root_path = "/root/quanhd/DATA"
        self.mode = mode
        self.type = type
        self.load_data_from_json()

    def __len__(self):
        return len(self.image_paths)
    
    def load_data_from_json(self):
        with open(self.path) as f:
            data = json.load(f)
        if self.mode == "train":
            image_paths = data[self.type][self.mode]["images"]
            mask_paths = data[self.type][self.mode]["masks"]
            print(len(image_paths))
            self.image_paths = image_paths[:int(self.train_ratio*len(image_paths))]
            self.mask_paths = mask_paths[:int(self.train_ratio*len(mask_paths))]
            print(len(self.image_paths))
        elif self.mode == "test":
            self.image_paths = data[self.type][self.mode]["images"]
            self.mask_paths = data[self.type][self.mode]["masks"]

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

def recall_np(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon)
    return recall

def precision_np(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon)
    return precision

def dice_np(y_true, y_pred):
    precision = precision_np(y_true, y_pred)
    recall = recall_np(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+epsilon))

def iou_np(y_true, y_pred):
    intersection = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    union = np.sum(y_true)+np.sum(y_pred)-intersection
    return intersection/(union+epsilon)

def get_micro_scores(gts, prs): # Micro
    mean_precision = 0
    mean_recall = 0
    mean_iou = 0
    mean_dice = 0

    total_area_intersect = 0
    total_area_union = 0
    total_pr = 0
    total_gt = 0

    for gt, pr in zip(gts, prs):
        total_area_intersect += torch.sum(torch.round(torch.clip(gt * pr, 0, 1)))
        total_area_union += torch.sum(torch.round(torch.clip(gt + pr, 0, 1)))
        total_pr += torch.sum(torch.round(torch.clip(pr, 0, 1)))
        total_gt += torch.sum(torch.round(torch.clip(gt, 0, 1)))

    mean_precision = total_area_intersect / (total_pr + epsilon)
    mean_recall = total_area_intersect / (total_gt + epsilon)
    mean_iou = total_area_intersect / (total_area_union + epsilon)
    mean_dice = 2 * total_area_intersect / (total_pr + total_gt + epsilon)

    print("Micro scores: dice={}, miou={}, precision={}, recall={}".format(mean_dice, mean_iou, mean_precision, mean_recall))

    return (mean_iou, mean_dice, mean_precision, mean_recall)

def get_scores(gts, prs):
    mean_precision = 0
    mean_recall = 0
    mean_iou = 0
    mean_dice = 0
    for gt, pr in zip(gts, prs):
        mean_precision += precision_np(gt, pr)
        mean_recall += recall_np(gt, pr)
        mean_iou += iou_np(gt, pr)
        mean_dice += dice_np(gt, pr)

    mean_precision /= len(gts)
    mean_recall /= len(gts)
    mean_iou /= len(gts)
    mean_dice /= len(gts)        
    
    print("Macro scores: dice={}, miou={}, precision={}, recall={}".format(mean_dice, mean_iou, mean_precision, mean_recall))

    return (mean_iou, mean_dice, mean_precision, mean_recall)



def inference(model, args):
    print("#"*20)
    model.eval()
    
    X_test = glob('{}/images/*'.format(args.test_path))
    print(len(X_test))
    X_test.sort()
    y_test = glob('{}/masks/*'.format(args.test_path))
    y_test.sort()

    test_dataset = Dataset(X_test, y_test, train_ratio=1.0, type=args.type, mode="test")

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    gts = []
    prs = []
    for i, pack in enumerate(test_loader, start=1):
        image, gt = pack
        gt = gt[0][0]
        gt = np.asarray(gt, np.float32)
        image = image.cuda()

        res, res2, res3, res4 = model(image)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        pr = res.round()
        gts.append(gt)
        prs.append(pr)
    get_scores(gts, prs)
    get_micro_scores(gts, prs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str,
                        default='b3')
    parser.add_argument('--weight', type=str,
                        default='')
    parser.add_argument('--type', type=str,
                        default='', help='backbone version')
    parser.add_argument('--test_path', type=str,
                        default='./data/TestDataset', help='path to dataset')
    parser.add_argument('--num_classes', type=int,
                        default=1, help='number output classes')
    parser.add_argument('--bottleneck', type=bool,
                        default=True, help='use bottle neck in reverse attention or not')
    parser.add_argument('--neo', type=bool,
                        default=False, help='use Neo Reverse Attention or not (Softmax Reverse Attentions)')
    args = parser.parse_args()

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
                pretrained='pretrained/mit_{}.pth'.format(args.backbone)).cuda()

    if args.weight != '':
        checkpoint = torch.load(args.weight)
        model.load_state_dict(checkpoint['state_dict'])

    inference(model, args)


