import cv2
import numpy as np
from imutils import grab_contours
from io import BytesIO
from typing import Any


class ImageProcessor:
    def __init__(self, img_bytes: BytesIO | np.ndarray | Any):
        if isinstance(img_bytes, BytesIO):
            self.img = np.asarray(bytearray(img_bytes.read()), dtype=np.uint8)
        elif isinstance(img_bytes, bytes):
            self.img = cv2.imdecode(np.asarray(bytearray(img_bytes), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        elif isinstance(img_bytes, np.ndarray):
            self.img = img_bytes
        else:
            raise ValueError(f'Image type not supported: {type(img_bytes)}')

    def _grab_contours(self):
        """
        Finds contours on image
        :return:
        """
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = grab_contours(cnts)
        return cnts

    def _draw_contours(self, ctns) -> np.ndarray:
        """
        :param ctns: contours of image
        :return: copy of image with drown cnts
        """
        copied_img: np.ndarray = self.img.copy()
        for c in ctns:
            cv2.drawContours(copied_img, [c], -1, (240, 0, 159), 3)
        return copied_img

    def make_contours(self):
        cnts = self._grab_contours()
        return cnts, self._draw_contours(cnts)

    def put_text(self, text: str):
        cv2.putText(self.img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 0, 159), 2)
        return self.img.copy()

    def put_sqrt(self):
        shapes = [self.img.shape[1]//2, self.img.shape[0]//2]
        center_coord = np.array(shapes, dtype=np.int16)
        size = np.array([20, 20], dtype=np.int16)
        start = center_coord - size
        end = center_coord + size
        cv2.rectangle(self.img, start, end, color=(0, 255, 0), thickness=-1)
        return self.img.copy()

    def put_dot(self):
        y = self.img.shape[0] - 10
        x = 25
        cv2.circle(self.img, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
        return self.img.copy()


def encode_cv2(image: np.ndarray):
    is_success, buffer = cv2.imencode(".jpg", image)
    io_buf = BytesIO(buffer)
    return io_buf
