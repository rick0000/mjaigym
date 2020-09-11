import numpy as np
from .feature import Feature
from mjaigym.board.mj_move import MjMove
from mjaigym.board.board_state import BoardState
from mjaigym.board.function.pai import Pai


class DoraFeature(Feature):
    
    @classmethod
    def get_length(cls)->int:
        return 4

    @classmethod
    def calc(cls, result:np.array, board_state:BoardState, player_id:int, oracle_enable_flag:bool=False):
        dora_markers = board_state.dora_markers
        nums = {}
        for dora_maker in dora_markers:
            pai = Pai.from_str(dora_maker)
            dora_pai = pai.succ
            if dora_pai.id not in nums:
                nums[dora_pai.id] = 0
            nums[dora_pai.id] += 1
        
        for pai_id, n in nums.items():
            if n > 0:
                result[0:n, pai_id] = 1

    