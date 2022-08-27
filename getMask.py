import cv2
import numpy as np
def mask():
    x, y = (1920, 1080)

    # 蓝色检测区域
    img = np.zeros((y, x), dtype=np.uint8)
    blue_space_up = np.array([[650, 150], [650, 250], [1200, 250], [1200, 150]], np.int32)
    polygon_blue_up = cv2.fillConvexPoly(img, blue_space_up, color=1)
    polygon_blue_up = polygon_blue_up[:, :, np.newaxis]

    img = np.zeros((y, x), dtype=np.uint8)
    blue_space_left = np.array([[0, 250], [0, 450], [300, 450], [300, 250]], np.int32)
    polygon_blue_left = cv2.fillConvexPoly(img, blue_space_left, color=2)
    polygon_blue_left = polygon_blue_left[:, :, np.newaxis]

    img = np.zeros((y, x), dtype=np.uint8)
    blue_space_right = np.array([[1600, 250], [1600, 500], [1920, 500], [1920, 250]], np.int32)
    polygon_blue_right = cv2.fillConvexPoly(img, blue_space_right, color=3)
    polygon_blue_right = polygon_blue_right[:, :, np.newaxis]

    # 黄色检测区域
    img = np.zeros((y, x), dtype=np.uint8)
    yellow_space = np.array([[0, 600], [0, 1000], [1920, 1000], [1920, 600]], np.int32)
    polygon_yellow = cv2.fillConvexPoly(img, yellow_space, color=4)
    polygon_yellow = polygon_yellow[:, :, np.newaxis]
    # 总检测区域
    polygon_yellow_blue = polygon_blue_up + polygon_blue_right + polygon_blue_left + polygon_yellow

    img = np.zeros((y, x, 3), dtype=np.uint8)
    # 区域画面展示
    show_img = cv2.add((cv2.fillConvexPoly(img, blue_space_up, color=[255, 0, 0]) \
                        + cv2.fillConvexPoly(img, blue_space_right, color=[255, 0, 0]) \
                        + cv2.fillConvexPoly(img, yellow_space, color=[0, 255, 255])),
                       cv2.fillConvexPoly(img, blue_space_left, color=[255, 0, 0]))
    return show_img,polygon_yellow_blue

if __name__ == '__main__':
    show_img,polygon_yellow_blue = mask()
    cv2.imshow("1",show_img)
    # cv2.imshow("2",polygon_yellow_blue)
    # cv2.waitKey(1000)