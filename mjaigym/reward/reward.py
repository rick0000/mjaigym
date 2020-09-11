from typing import List, Dict
from abc import ABCMeta, abstractclassmethod, abstractmethod
from mjaigym.kyoku import Kyoku
from mjaigym.board.mj_move import MjMove

class Reward(metaclass=ABCMeta):
    
    @abstractmethod
    def step(self, mjson):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()
    
    @abstractmethod
    def calc(self)->List[float]:
        raise NotImplementedError()    

    @abstractclassmethod
    def calc_from_kyoku(cls,kyoku:Kyoku)->List[float]:
        raise NotImplementedError()
