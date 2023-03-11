
import argparse
# import logging
# import os
# import random
# import sys
# import time
import numpy as np
import cv2
# from tqdm import tqdm
# from glob import glob
# import matplotlib.pyplot as plt

import torch
from keras.utils.np_utils import to_categorical
import torch.nn.functional as F


if __name__ == "__main__":

    HEIGHT = 384
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str,
                        default="", help='path to input video')
    parser.add_argument('--output_video', type=str,
                        default='', help='path to output video')
    parser.add_argument('--backbone', type=str,
                        default="b1", help='learning rate')
    parser.add_argument('--checkpoint', type=str,
                        default="", help='training batch size')
    parser.add_argument('--type', type=str,
                        default="onnx", help='training dataset size')

    args = parser.parse_args()
    if args.type == "torch":
        from mmseg.models.segmentors import BiRAFormer as UNet
        model = UNet(backbone=dict(
        type='mit_{}'.format(args.backbone),
        style='pytorch'), compound_coef=4,
        num_classes=3,
        neo=True, numrepeat=2, bottleneck=True)
        model.eval()
        model.cuda()
    elif args.type == "onnx":
        import onnxruntime
        session = onnxruntime.InferenceSession(args.checkpoint, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # VID_20200831_142511#VID_20200831_151500
    cap = cv2.VideoCapture(args.input_video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(
        args.output_video, fourcc, 10.0, (1280, 1080))
    kernel = np.ones((5, 5), np.uint8)
    
    while (cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break
        #out = np.zeros((780,980*2,3),dtype=np.uint8)
        h, w, _ = frame.shape
        frame = frame[:, :2*w//3, :]  # frame[90:870,200:2*w//3-100,:]#
    #     print(frame.shape)
    #     plt.imshow(frame)
    #     plt.show()
    #     break
        test_img = (frame/255.).astype(np.float32)
        test_img = cv2.resize(test_img, (HEIGHT, HEIGHT))
        if args.type == "torch":
            p = model(torch.from_numpy(np.transpose(
                np.expand_dims(test_img, axis=0), [0, 3, 1, 2])).cuda())
            p = F.upsample(p, size=(HEIGHT, HEIGHT),
                        mode='bilinear', align_corners=False)
            res = p.data.cpu().numpy().squeeze()  # .sigmoid()
        elif args.type == "onnx":
            test_img =np.transpose(np.array([test_img]),[0,3,1,2])
            ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(test_img.astype(np.float32))
            results = session.run(["output"], {"input": ortvalue})
            
            res = results[0][0]
            print(res.shape)
        # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # pr = res.round()
        p = np.transpose(res, [1, 2, 0])
        # print(p.shape)
        p = np.argmax(p, axis=-1)
        # print(p)
        p = to_categorical(p, num_classes=3).astype(np.uint8)
        # print(print(p.shape))
        p = p.astype(np.uint8)
        # opening
        p[:, :, 1] = cv2.erode(p[:, :, 1], np.ones(
            (3, 3), np.uint8), iterations=2)
        p[:, :, 1] = cv2.dilate(p[:, :, 1], np.ones(
            (3, 3), np.uint8), iterations=2)

        p[:, :, 2] = cv2.erode(p[:, :, 2], np.ones(
            (3, 3), np.uint8), iterations=2)
        p[:, :, 2] = cv2.dilate(p[:, :, 2], np.ones(
            (3, 3), np.uint8), iterations=2)
        # closing
        p[:, :, 1] = cv2.dilate(p[:, :, 1], np.ones(
            (5, 5), np.uint8), iterations=2)
        p[:, :, 1] = cv2.erode(p[:, :, 1], np.ones(
            (5, 5), np.uint8), iterations=2)

        p[:, :, 2] = cv2.dilate(p[:, :, 2], np.ones(
            (5, 5), np.uint8), iterations=2)
        p[:, :, 2] = cv2.erode(p[:, :, 2], np.ones(
            (5, 5), np.uint8), iterations=2)

        res = p.copy().astype(np.uint8)
        res[:, :, 0] = 0
        res[:, :, 1] = cv2.dilate(res[:, :, 1], kernel, iterations=1)

        res[:, :, 2] = cv2.dilate(res[:, :, 2], kernel, iterations=1)
        bound = res-p
        bound = bound*255
        # p[:,:,0]=0
        #bound +=p*128
        frame_copy = frame.copy()
        bound = cv2.resize(bound, (1280, 1080),
                           interpolation=cv2.INTER_NEAREST)
        frame_copy[bound == 255] = bound[bound == 255]
        #frame_copy[bound==128] = frame_copy[bound==128]//2+bound[bound==128]
    #     out[:,:980,:]=frame
    #     out[:,980:,:]=frame_copy
        video_writer.write(frame_copy)
    #     plt.imshow(frame_copy[:,:,::-1])
    #     plt.show()
    #     break
    cap.release()
    video_writer.release()
