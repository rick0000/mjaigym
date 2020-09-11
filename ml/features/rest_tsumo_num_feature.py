import numpy as np
from .feature import Feature
from mjaigym.board.mj_move import MjMove
from mjaigym.board.board_state import BoardState
from mjaigym.board.function.pai import Pai
import math
class RestTsumoNumFeature(Feature):
    @classmethod
    def get_length(cls)->int:
        return 18 # (0~70) convert to divided 4 -> math.ceil(71/4) = 18

    @classmethod
    def calc(cls, result:np.array, board_state:BoardState, player_id:int, oracle_enable_flag:bool=False):
        index = board_state.yama_rest_num // 4
        result[index,:] = 1

    