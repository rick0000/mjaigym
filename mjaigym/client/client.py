from abc import ABCMeta, abstractmethod
from mjaigym.board import BoardState
from typing import Dict, List, Tuple

class Client(metaclass=ABCMeta):

    def __init__(self, id, name):
        self.id = id
        self.name = name

    @abstractmethod
    def think(self, board_state:BoardState):
        raise NotImplementedError()
