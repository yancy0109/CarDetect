import argparse

import cv2
import torch.backends.cudnn as cudnn

from models.experimental import *
from utils.datasets import *
from utils.utils import *
from models.LPRNet import *



imgsz = 640
device = 'cuda' if torch.cuda.is_available() else 'cpu'
half = device != 'cpu'
def loadCarModel():

    model = attempt_load("./weights/last.pt", map_location=device)  # load FP32 model
    imgsz = check_img_size(640, s=model.stride.max())
    if half:
        model.half()  # to FP16
    modelc = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0).to(device)
    modelc.load_state_dict(torch.load('./weights/Final_LPRNet_model.pth', map_location=torch.device('cpu')))
    print("load pretrained model successful!")
    modelc.to(device).eval()
    return model , modelc
if __name__ == '__main__':
    path = "daolu1.avi"
    cap = cv2.VideoCapture(path)
    model ,modelc ,imgsz = loadCarModel()
    i=0
    while True:
        _, image = cap.read()
        #Padded resize
        img = letterbox(image, imgsz)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
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
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.8, 0.5, None, agnostic=False)
        pred,plat_num = apply_classifier(pred, modelc, img, image)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                # Print results
                s={}
                for de,lic_plat in zip(det,plat_num):
                    # xyxy,conf,cls,lic_plat=de[:4],de[4],de[5],de[6:]
                    *xyxy, conf, cls=de
                    lb = ""
                    for a,i in enumerate(lic_plat):
                        # if a ==0:
                        #     continue
                        lb += CHARS[int(i)]
                    if(conf>0.8) :
                        print(lb)