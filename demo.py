import cv2
import torch
import modelLoad
from getMask import mask
import threading
import datetime
import numpy as np
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import scale_coords, xyxy2xywh, non_max_suppression, apply_classifier
from yolov5.utils.plots import Annotator, colors


#生成检测模型
model,deepsort = modelLoad.load_model()
print("--------------检测模型加载完毕--------------")

#生成遮罩图层
show_img,polygon_yellow_blue = mask()
print("--------------遮罩生成成功--------------")

imgsz = (640, 640)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
half = device != 'cpu'
#车辆识别模型数据
stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
print("--------------车辆识别模型数据加载完毕--------------")

#创建线程锁
lock = threading.Lock()


#摄像头
viedo_path= "video/main.mp4"
#开启摄像头
cap = cv2.VideoCapture(viedo_path)

passingResult = {}

#获得返回函数
def setGreenWaiting(end_time):
    lock.acquire()
    print("开始设定")
    global passingResult
    global cap
    # 开启线程锁定
    count_up = 0
    count_left = 0
    count_right = 0
    yellow_list = []
    while (datetime.datetime.now() < end_time):
        retval, image = cap.read()
        img = letterbox(image, imgsz, stride=stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img, augment=False, visualize=False)
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=(2), agnostic=False, max_det=1000)
        for i, det in enumerate(pred):
            annotator = Annotator(image, line_width=2, pil=not ascii)
            # cv2.imshow("正在读取绿灯",annotator.result())
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], image.shape).round()
                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), image)
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(c, True))
                        cv2.imshow("The image of passing car", cv2.add(annotator.result(),show_img))
                        cv2.waitKey(1)
                        x1, y1, x2, y2, track_id, class_id = output
                        # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                        y1_offset = int(y1 + ((y2 - y1) * 0.5))
                        # 撞线的点
                        y = y1_offset
                        x = x1
                        if polygon_yellow_blue[y, x] == 4:
                            # 如果撞黄线
                            if track_id not in yellow_list:
                                yellow_list.append(track_id)
                            print("撞黄线:" + yellow_list.__str__())
                        if polygon_yellow_blue[y, x] == 1:
                            # 如果撞蓝线
                            if track_id in yellow_list:
                                count_up += 1
                                print("移除" + str(track_id))
                                yellow_list.remove(track_id)
                        if polygon_yellow_blue[y, x] == 2:
                            # 如果撞蓝线
                            if track_id in yellow_list:
                                count_left += 1
                                print("移除" + str(track_id))
                                yellow_list.remove(track_id)
                        if polygon_yellow_blue[y, x] == 3:
                            # 如果撞蓝线
                            if track_id in yellow_list:
                                count_right += 1
                                print("移除" + str(track_id))
                                yellow_list.remove(track_id)
    passingResult["up"] = count_up
    passingResult["left"] = count_left
    passingResult["right"] = count_right
    # 释放线程锁
    print("设定结束")
    print(passingResult)
    lock.release()

def setGreenWait(time):
    #获取需要测量的通行时间
    current_time = datetime.datetime.now()
    end_time = current_time+datetime.timedelta(seconds=time)
    #开启线程计数
    threadSet = threading.Thread(target=setGreenWaiting,args=(end_time,))
    threadSet.start()
    # threadSet.join()

if __name__ == '__main__':
    setGreenWait(30)