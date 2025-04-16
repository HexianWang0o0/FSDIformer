import os
import torch    
import torch.nn as nn   


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError  
        return None

    def _acquire_device(self):
        pass

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
