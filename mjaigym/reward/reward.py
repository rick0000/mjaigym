from abc import ABCMeta, abstractclassmethod, abstractmethod
from typing import Dict, List

from mjaigym.board.mj_move import MjMove
from mjaigym.kyoku import Kyoku


class Reward(metaclass=ABCMeta):
    @abstractmethod
    def step(self, mjson):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def calc(self) -> List[float]:
        raise NotImplementedError()

    @abstractclassmethod
    def calc_from_kyoku(cls, kyoku: Kyoku) -> List[float]:
        raise NotImplementedError()
