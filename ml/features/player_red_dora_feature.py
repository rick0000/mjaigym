import numpy as np
from .feature import Feature
from mjaigym.board.mj_move import MjMove
from mjaigym.board.board_state import BoardState
from mjaigym.board.function.pai import Pai


class PlayerRedDoraFeature(Feature):
    
    @classmethod
    def get_length(cls)->int:
        return 3

    @classmethod
    def calc(cls, result:np.array, board_state:BoardState, player_id:int, oracle_enable_flag:bool=False):
        num = board_state.red_dora_nums[player_id]
        result[:num,:,:] = 1