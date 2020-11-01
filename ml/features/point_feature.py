import numpy as np
from .feature import Feature
from mjaigym.board.mj_move import MjMove
from mjaigym.board.board_state import BoardState
from mjaigym.board.function.pai import Pai


class PointFeature(Feature):
    
    ONE_PLAYER_LENGTH = 80
    @classmethod
    def get_length(cls)->int:
        return cls.ONE_PLAYER_LENGTH * 4

    @classmethod
    def calc(cls, result:np.array, board_state:BoardState, player_id:int, oracle_enable_flag:bool=False):
        scores = board_state.scores
        for i_from_player, raw_id in cls.get_seat_order_ids(player_id):
            score = scores[raw_id]
            if score > 79000:
                score = 79000
            elif score < 0:
                score = 0
            score_index = score // 1000

            start_index = cls.ONE_PLAYER_LENGTH * i_from_player
            result[start_index+score_index,:,:] = 1

    