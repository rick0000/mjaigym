import numpy as np
from .feature import Feature
from mjaigym.board.mj_move import MjMove
from mjaigym.board.board_state import BoardState
from mjaigym.board.function.pai import Pai


class FuroRedDoraFeature(Feature):
    
    RED_DORA_MAX = 3
    @classmethod
    def get_length(cls)->int:
        return cls.RED_DORA_MAX * 4 

    @classmethod
    def calc(cls, result:np.array, board_state:BoardState, player_id:int, oracle_enable_flag:bool=False):
        for target_player_id, raw_id in cls.get_seat_order_ids(player_id):
            num = board_state.furo_open_red_dora_nums[raw_id]
            start_index = cls.RED_DORA_MAX * target_player_id
            result[start_index:num,:,:] = 1