import numpy as np
from .feature import Feature
from mjaigym.board.mj_move import MjMove
from mjaigym.board.board_state import BoardState

class BakazeFeature(Feature):
    
    @classmethod
    def get_length(cls)->int:
        return 4

    @classmethod
    def calc(cls, result:np.array, board_state:BoardState, player_id:int, oracle_enable_flag:bool=False):
        bakaze = board_state.bakaze
        if bakaze == 'E':
            index = 0
        elif bakaze == 'S':
            index = 1
        elif bakaze == 'W':
            index = 2
        elif bakaze == 'N':
            index = 3
        else:
            raise Exception('not intended path')

        result[index,:] = 1


