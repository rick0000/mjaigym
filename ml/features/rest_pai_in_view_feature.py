import numpy as np
from .feature import Feature
from mjaigym.board.mj_move import MjMove
from mjaigym.board.board_state import BoardState
from mjaigym.board.function.pai import Pai

class RestPaiInViewFeature(Feature):
    @classmethod
    def get_length(cls)->int:
        return 5

    @classmethod
    def calc(cls, result:np.array, board_state:BoardState, player_id:int, oracle_enable_flag:bool=False):
        rests = board_state.restpai_in_view[player_id]
        for i, num in enumerate(rests):
            result[:num+1, i, 0] = 1

        


