import numpy as np
from .feature import Feature
from mjaigym.board.mj_move import MjMove
from mjaigym.board.board_state import BoardState
from mjaigym.board.function.pai import Pai

class LastTsumoFeature(Feature):

    @classmethod
    def get_length(cls)->int:
        return 2

    @classmethod
    def calc(cls, result:np.array, board_state:BoardState, player_id:int, oracle_enable_flag:bool=False):
        if board_state.previous_action["type"] != MjMove.dahai.value:
            return
        if board_state.previous_action["actor"] != player_id:
            return

        last_dahai_pai = Pai.from_str(board_state.previous_action['pai'])
        result[0,last_dahai_pai.id] = 1
        
        if last_dahai_pai.is_red:
            result[1,:] = 1
        