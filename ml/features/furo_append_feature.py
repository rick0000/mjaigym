""" feature base class

Raises:
    NotImplementedError: [description]

Returns:
    [type]: [description]
"""
from abc import ABCMeta, abstractmethod, abstractclassmethod
from mjaigym.board import BoardState
from mjaigym.board.function.pai import Pai
from typing import List, Tuple, Dict
import numpy as np
from .feature import Feature


class FuroAppendFeature(metaclass=ABCMeta):
        
    @abstractclassmethod
    def get_length(cls)->int:
        raise NotImplementedError()
        
    @abstractclassmethod
    def calc(cls, result:np.array, board_state:BoardState, player_id:int, candidate_furo:Dict, oracle_enable_flag:bool=False):
        raise NotImplementedError()

    @classmethod
    def get_seat_order_ids(cls, player_id:int)->List[Tuple[int,int]]:
        return Feature.get_seat_order_ids(player_id)