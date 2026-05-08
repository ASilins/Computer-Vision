import torch

from .hook import Hook


class EmptyCacheHook(Hook):
    def __init__(self, before_epoch=False, after_epoch=True, after_iter=False):
        self._before_epoch = before_epoch
        self._after_epoch = after_epoch
        self._after_iter = after_iter

    def after_iter(self, trainer):
        if self._after_iter and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def before_epoch(self, trainer):
        if self._before_epoch and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def after_epoch(self, trainer):
        if self._after_epoch and torch.cuda.is_available():
            torch.cuda.empty_cache()
