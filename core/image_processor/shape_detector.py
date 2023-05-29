import cv2
import numpy as np
import imutils


class DetectorFactory:

    @property
    def detector(self):
        sd = ShapeDetector()
        return sd


class ShapeDetector:

    def __init__(self):
        ...

    @staticmethod
    def process_4(approx):
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
        return shape

    def shape_switch(self, approx):
        approx_len = len(approx)
        # callback dict
        shape_switch = {
            3: lambda: 'triangle',
            4: lambda: self.process_4(approx),
            5: lambda: 'pentagon',
        }
        lazy_res = shape_switch.get(approx_len)
        if lazy_res:
            return lazy_res()
        return 'circle'

    def detect(self, c):
        # initialize the shape name and approximate
        # the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        return self.shape_switch(approx)


def process(detector: ShapeDetector, image_bytes: bytes):
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 225, 255,
                           cv2.THRESH_BINARY_INV)[1]
    # find contours in the thresholded image and initialize the
    # shape detector
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        M = cv2.moments(c)
        cX = int((M["m10"] / M["m00"]))
        cY = int((M["m01"] / M["m00"]))
        shape = detector.detect(c)
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)
    return image
