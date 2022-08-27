import random
import sys

import cv2

from model.LPRNet import LPRNet, CHARS

import numpy as np
import torch



sys.path.insert(0, './yolov5')
from yolov5.utils.general import scale_coords, non_max_suppression, apply_classifier, check_img_size

from yolov5.utils.augmentations import letterbox

device = 'cuda' if torch.cuda.is_available() else 'cpu'
videopath = "video/carplate.avi"
car_model = './model/last.pt'
half = device != 'cpu'
cap = cv2.VideoCapture(videopath)
imgsz = 640



while True:
    _, image = cap.read()
    # Padded resize
    # img = letterbox(image, imgsz)[0]
    #
    # # Convert
    # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    # img = np.ascontiguousarray(img)
    img = letterbox(image, imgsz, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    #检测
    pred = model(img, augment=False, visualize=False)[0]
    pred = non_max_suppression(pred, 0.8, 0.5, None, agnostic=False)
    pred,plat_num = apply_classifier(pred, modelc, img, image)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
            # Print results
            s={}
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string
            print(s)