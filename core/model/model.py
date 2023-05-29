import pathlib
import cv2
import numpy as np
from typing import Any
import pickle

base_dir = pathlib.Path(__file__).parent.resolve()
default_model_fp = str(base_dir / 'datasets' / 'model_ready.yml')


class BaseModel:
    model: Any

    def __init__(self, mode: str = 'USE'):
        self.model = None
        mode_callback = {
            'TRAIN': lambda: self.prepare_model(),
            'USE': lambda: self.import_model(),
            'DEFAULT': lambda: 0
        }
        mode_callback[mode]()

    def import_model(self):
        """
        Imports model pre-req from external storage
        """
        self.model = cv2.ml.KNearest_load(default_model_fp)

    def prepare_model(self):
        """
        Trains model and instantiates it
        """
        model = self._train_model()
        self.model = model

    @staticmethod
    def _train_model() -> Any:
        """
        Trains model, returns it
        """
        samples = np.loadtxt(base_dir / 'datasets' / 'generalsamples.data', np.float32)
        responses = np.loadtxt(base_dir / 'datasets' / 'generalresponses.data', np.float32)
        responses = responses.reshape((responses.size, 1))
        model = cv2.ml.KNearest_create()
        model.train(samples, cv2.ml.ROW_SAMPLE, responses)
        return model

    def export_model(self):
        """
            Exports model to external storage
        """
        self.model.save(default_model_fp)
