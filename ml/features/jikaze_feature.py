import numpy as np
from .feature import Feature
from mjaigym.board.mj_move import MjMove
from mjaigym.board.board_state import BoardState
from mjaigym.board.function.pai import Pai

class JikazeFeature(Feature):
    
    @classmethod
    def get_length(cls)->int:
        return 4

    @classmethod
    def calc(cls, result:np.array, board_state:BoardState, player_id:int, oracle_enable_flag:bool=False):
        jikaze = board_state.jikaze[player_id]
        if jikaze == 'E':
            index = 0
        elif jikaze == 'S':
            index = 1
        elif jikaze == 'W':
            index = 2
        elif jikaze == 'N':
            index = 3
        else:
            raise Exception('not intended path')

        result[index,:] = 1

    