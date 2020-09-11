import numpy as np
from .feature import Feature
from mjaigym.board.mj_move import MjMove
from mjaigym.board.board_state import BoardState
from mjaigym.board.function.pai import Pai


FIX_VALUE = np.zeros((5,34,1), dtype=int)
FIX_VALUE[0,0:9,:]=1
FIX_VALUE[1,9:18,:]=1
FIX_VALUE[2,18:27,:]=1
FIX_VALUE[3,27:31,:]=1
FIX_VALUE[4,31:34,:]=1

class TypeFeature(Feature):
    
    @classmethod
    def get_length(cls)->int:
        return 5

    @classmethod
    def calc(cls, result:np.array, board_state:BoardState, player_id:int, oracle_enable_flag:bool=False):
        np.copyto(result, FIX_VALUE)
    