""" feature base class

Raises:
    NotImplementedError: [description]

Returns:
    [type]: [description]
"""
from abc import ABCMeta, abstractmethod, abstractclassmethod
from mjaigym.board import BoardState
from mjaigym.board.function.pai import Pai
from typing import List, Tuple
import enum
import numpy as np
import pprint
import os


class Feature(metaclass=ABCMeta):
        
    @abstractclassmethod
    def get_length(cls)->int:
        raise NotImplementedError()
        
    @abstractclassmethod
    def calc(cls, result:np.array, board_state:BoardState, player_id:int, oracle_enable_flag:bool=False):
        raise NotImplementedError()

    @classmethod
    def get_seat_order_ids(cls, player_id:int)->List[Tuple[int,int]]:
        """returns seat list as player_id is primary view point.
        ex) 1 -> [1, 2, 3, 0],  3 -> [3, 0, 1, 2]

        Args:
            player_id (int): as primary viewpoint

        Returns:
            List[int]: [self, shimocha, toimen, kamicha]
        """
        return enumerate([
            (player_id + 0 + 4) % 4,
            (player_id + 1 + 4) % 4,
            (player_id + 2 + 4) % 4,
            (player_id + 3 + 4) % 4,
            ])

