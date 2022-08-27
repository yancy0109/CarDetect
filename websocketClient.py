# WebSocket

import base64
import cv2
import socketClient
path = "video/main.mp4"
path1= "video/img.png"
with open(path1, 'rb') as f:
    img_byte = base64.b64encode(f.read())
img_str = img_byte.decode('ascii')
# print(img_str)

ws = socketClient.getClient("localhost","8080","1")
# ws.send(img_str)


capture = cv2.VideoCapture(0)
i = 0
while(True):
    _, image = capture.read()
    if image is None:
        break
    i += 1
    cv2.imwrite(path1, image)
    # 取值范围：0~9，数值越小，压缩比越低，图片质量越高
    # params = [cv2.IMWRITE_PNG_COMPRESSION, 9]  # ratio: 0~9
    # msg = cv2.imencode(".png", image, params)[1]\
    # msg = (np.array(msg)).tobytes()
    # print(msg)
    with open(path1, 'rb') as f:
        img_byte = base64.b64encode(f.read())
    img_str = img_byte.decode('ascii')
    ws.send("1,"+img_str)
    cv2.waitKey(1000)
# #
ws.close()