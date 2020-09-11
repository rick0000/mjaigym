import numpy as np
from .feature import Feature
from mjaigym.board.mj_move import MjMove
from mjaigym.board.board_state import BoardState
from mjaigym.board.function.pai import Pai

class TehaiFeature(Feature):
    
    @classmethod
    def get_length(cls)->int:
        return 4

    @classmethod
    def calc(cls, result:np.array, board_state:BoardState, player_id:int, oracle_enable_flag:bool=False):
        player_tehai = board_state.tehais[player_id]
        nums = [0] * 34
        for t in player_tehai:
            nums[t.id] += 1
        
        for i,n in enumerate(nums):
            if n > 0:
                result[0:n,i] = 1

    