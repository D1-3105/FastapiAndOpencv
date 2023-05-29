import cv2
import numpy as np
from core.model import BaseModel
from typing import Callable

relatives = {
    0: 'Star',
    1: 'Comment',
    2: 'Triangle',
    3: 'Rhombus',
    4: 'Cloud',
    5: 'Lightning',
    6: 'Heart'
}


class FigureDetector:

    def __init__(self, input_image: bytes, knn_model_class: BaseModel.__mro__):
        self.model = knn_model_class().model
        self.contour_num = 0
        self.input_image = cv2.imdecode(np.frombuffer(input_image, np.uint8), cv2.IMREAD_COLOR)
        self.colors = {
            'yellow': (
                np.array([20, 180, 200]),
                np.array([40, 255, 255]),
                np.array([10, 180, 200]),
                np.array([20, 255, 255])
            ),
            'blue': (
                np.array([110, 160, 190]),
                np.array([130, 255, 250]),
                np.array([90, 180, 190]),
                np.array([110, 255, 250])
            ),
            'green': (np.array([50, 100, 100]), np.array([70, 255, 255])),
            'red': (np.array([170, 200, 150]), np.array([180, 255, 255]))
        }
        self.contours = {key: [] for key in self.colors}
        self.masks = {key: [] for key in self.colors}

    def color_check(self):
        blurred = cv2.GaussianBlur(self.input_image, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        for col in self.colors:
            mask = cv2.inRange(hsv, self.colors.get(col)[0], self.colors.get(col)[1])
            if len(self.colors.get(col, [])) > 2:
                mask += cv2.inRange(hsv, self.colors.get(col)[2], self.colors.get(col)[3])
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:][0]
            contours = [i for i in contours if cv2.contourArea(i) > 50]
            self.contours[col] = contours
            self.masks[col] = mask

    def find_figure(self, contours: list[np.ndarray], mask):
        model = self.model
        for cnt in contours:
            [x, y, w, h] = cv2.boundingRect(cnt)
            if h > 28:
                cv2.rectangle(self.input_image, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)

                roi = mask[y:y + h, x:x + w]
                roi_small = cv2.resize(roi, (10, 10))
                roi_small = roi_small.reshape((1, 100))
                roi_small = np.float32(roi_small)
                retval, results, neigh_resp, dists = model.findNearest(roi_small, k=1)
                num = int(results[0])
                result = relatives[num]
                yield result, (x, y, w, h)

    def add_caption(self, output, col, result, *caption_args):
        [x, y, w, h] = caption_args
        text = f'{self.contour_num} {col} {result}'
        self.contour_num += 1
        cv2.putText(output, text, (x + w // 2, y + h // 2), 0, 0.6, (0, 255, 0))
