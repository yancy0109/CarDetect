import torch.cuda
import sys



sys.path.insert(0, './yolov5')

from yolov5.models.common import DetectMultiBackend
from deep_sort.deep_sort import DeepSort

device = 'cuda' if torch.cuda.is_available() else 'cpu'
deep_sort_model = "osnet_x1_0"
yolo_model = 'model/yolov5m.pt'
# car_model = 'model/last.pt'
dnn = False
MAX_DIST = 0.2
MAX_IOU_DISTANCE = 0.7
MAX_AGE = 30
N_INIT = 3
NN_BUDGET = 100
def load_model():
    # initialize deepsort
    deepsort = DeepSort(deep_sort_model,
                        device,
                        max_dist=MAX_DIST,
                        max_iou_distance=MAX_IOU_DISTANCE,
                        max_age=MAX_AGE, n_init=N_INIT, nn_budget=NN_BUDGET,
                        )
    half = device != 'cpu'
    model = DetectMultiBackend(yolo_model, device=device, dnn=dnn)
    model.model.half() if half else model.model.float()
    return model , deepsort
if __name__ == '__main__':
    model, deepsort  = load_model()