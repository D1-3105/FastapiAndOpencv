import cv2
import numpy as np
import imutils
import pathlib

base_dir = pathlib.Path(__file__).parent.resolve()


def prepare_training_data():
    im = cv2.imread(str(base_dir/'datasets/dataset.jpg'))
    resized = imutils.resize(im, width=600)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    samples = np.empty((0, 100))
    responses = []
    keys = [i for i in range(48, 58)]

    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            [x, y, w, h] = cv2.boundingRect(cnt)
            if h > 28:
                cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 0, 255), 2)
                roi = thresh[y:y + h, x:x + w]
                roi_small = cv2.resize(roi, (10, 10))
                cv2.imshow('roi', roi_small)
                cv2.imshow('norm', resized)
                key = cv2.waitKey(0)
                if key == 27:  # (escape to quit)
                    exit()
                elif key in keys:
                    responses.append(int(chr(key)))
                    sample = roi_small.reshape((1, 100))
                    samples = np.append(samples, sample, 0)

    responses = np.array(responses, np.float32)
    responses = responses.reshape((responses.size, 1))  # ?
    np.savetxt(base_dir / 'datasets' / 'generalsamples.data', samples)
    np.savetxt(base_dir / 'datasets' / 'generalresponses.data', responses)


####### training part ###############
def train_model():
    from core.model import BaseModel
    model_container = BaseModel(mode='TRAIN')
    model_container.export_model()
