import cv2
import numpy as np
import pytest
import typing
from core.image_processor import ImageProcessor

if typing.TYPE_CHECKING:
    from numpy import ndarray


@pytest.fixture
def new_img():
    blank_img = np.ones((480, 640, 3), np.uint8) * 255
    cv2.rectangle(blank_img, (100, 200), (200, 300), color=(0, 0, 255), thickness=-1)
    return blank_img


def test_detection(new_img: 'ndarray'):
    proc = ImageProcessor(new_img)
    contoured = proc.make_contours()
    cv2.imshow('test_detection', contoured)
    cv2.waitKey(0)


def test_text(new_img: 'ndarray'):
    proc = ImageProcessor(new_img)
    img = proc.put_text('123')
    cv2.imshow('test_text', img)
    cv2.waitKey(0)


def test_put_sqrt(new_img: 'ndarray'):
    proc = ImageProcessor(new_img)
    img = proc.put_sqrt()
    cv2.imshow('test_sqrt', img)
    cv2.waitKey(0)


def test_put_dot(new_img: 'ndarray'):
    proc = ImageProcessor(new_img)
    img = proc.put_dot()
    cv2.imshow('test_dot', img)
    cv2.waitKey()
