import datetime
import threading

import cv2
from flask import Flask, request

app = Flask(__name__)
#创建线程锁
lock = threading.Lock()

path = "video/main.mp4"
cap = cv2.VideoCapture(path)
def main():
    while (True):
        # lock.acquire()
        _, image = cap.read()

        # lock.release()
        cv2.imshow("main", image)
        cv2.waitKey(10)
threadMain = threading.Thread(target=main)
threadMain.start()

@app.route('/getImageRed')
def getImageRed():
    global cap
    time = request.args.get("time", type=int)
    current_time = datetime.datetime.now()
    end_time = current_time + datetime.timedelta(seconds=time)
    print("结束时间"+end_time.__str__())
    def Red():
        while (datetime.datetime.now() < end_time):
            lock.acquire()
            _, image = cap.read()
            print("正在读取")
            lock.release()
            cv2.imshow("red"+str(time), image)
            cv2.waitKey(10)
    threadRed = threading.Thread(target=Red)
    threadRed.start()
    return "red"
@app.route('/getImageGreen')
def getImageGreen():
    return "green"

if __name__ == '__main__':
    app.run()