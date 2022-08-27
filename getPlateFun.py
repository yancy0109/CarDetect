#车牌检测模型

# plateModel , modelc = loadCarModel()
#开启车牌检测线程
# def getPlate():
#     plateList={}
#     while True:
#         global cap
#         _, image = cap.read()
#         #Padded resize
#         img = letterbox(image, imgsz)[0]
#
#         # Convert
#         img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
#         img = np.ascontiguousarray(img)
#
#         img = torch.from_numpy(img).to(device)
#         img = img.half() if half else img.float()  # uint8 to fp16/32
#         img /= 255.0  # 0 - 255 to 0.0 - 1.0
#
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)
#
#         # Get names and colors
#         names = plateModel.module.names if hasattr(plateModel, 'module') else plateModel.names
#         colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
#         #检测
#         pred = plateModel(img, augment=False)[0]
#         pred = non_max_suppression(pred, 0.8, 0.5, None, agnostic=False)
#         pred,plat_num = apply_classifier(pred, modelc, img, image)
#         number = 1
#         # Process detections
#         for i, det in enumerate(pred):  # detections per image
#             if det is not None and len(det):
#                 # Rescale boxes from img_size to im0 size
#                 det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
#                 # Print results
#                 s={}
#                 for de,lic_plat in zip(det,plat_num):
#                     # xyxy,conf,cls,lic_plat=de[:4],de[4],de[5],de[6:]
#                     *xyxy, conf, cls=de
#                     lb = ""
#                     for a,i in enumerate(lic_plat):
#                         # if a ==0:
#                         #     continue
#                         lb += CHARS[int(i)]
#                     if(conf>0.8) :
#                         plateList[f"{number}"] = lb
#                         number+=1
#     #检测结果发送
#     data = json.dumps(plateList)
#
#     #置空
#     plateList = {}
#     #3秒进行一次检测
#     time.sleep(3)

# def getCarPlate():
#     global cap
#     _ ,image = cap.read()
#     while True:
#         _, image = cap.read()
#         # Padded resize
#         img = letterbox(image, imgsz)[0]
#         # Convert
#         img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
#         img = np.ascontiguousarray(img)
#         img = torch.from_numpy(img).to(device)
#         img = img.half() if half else img.float()  # uint8 to fp16/32
#         img /= 255.0  # 0 - 255 to 0.0 - 1.0
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)
#         # 检测
#         pred = model(img, augment=False)[0]
#         pred = non_max_suppression(pred, 0.8, 0.5, None, agnostic=False)
#         pred, plat_num = apply_classifier(pred, modelc, img, image)
#         # Process detections
#         for i, det in enumerate(pred):  # detections per image
#             if det is not None and len(det):
#                 # Rescale boxes from img_size to im0 size
#                 det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
#                 # Print results
#                 s = {}
#                 for de, lic_plat in zip(det, plat_num):
#                     # xyxy,conf,cls,lic_plat=de[:4],de[4],de[5],de[6:]
#                     *xyxy, conf, cls = de
#                     lb = ""
#                     for a, i in enumerate(lic_plat):
#                         # if a ==0:
#                         #     continue
#                         lb += CHARS[int(i)]
#                     if (conf > 0.8):
#                         print(lb)
