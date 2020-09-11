import numpy as np
from typing import Dict
from .furo_append_feature import FuroAppendFeature
from mjaigym.board.mj_move import MjMove
from mjaigym.board.board_state import BoardState
from mjaigym.board.function.pai import Pai
from mjaigym.board.function.rs_shanten_analysis import RsShantenAnalysis

class ShantenFuroAppendFeature(FuroAppendFeature):
    
    shanten_analysis = RsShantenAnalysis()

    @classmethod
    def get_length(cls)->int:
        return 8 # 1:furo can decrease shanten, 0~5: current_shanten

    @classmethod
    def calc(cls, result:np.array, board_state:BoardState, player_id:int, candidate_furo:Dict, oracle_enable_flag:bool=False):
        player_tehai = board_state.tehais[player_id]
        

        tehais = [0] * 34
        for pai in player_tehai:
            tehais[pai.id] += 1
        
        furo_num = len(board_state.furos[player_id])
        current_shanten = cls.shanten_analysis.calc_shanten(tehais, furo_num)

        # ignore kan
        if candidate_furo["type"] not in [MjMove.ankan.value, MjMove.daiminkan.value, MjMove.kakan.value]:
            add_pai = Pai.from_str(candidate_furo["pai"])
            tehais[add_pai.id] += 1
            added_shanten = cls.shanten_analysis.calc_shanten(tehais, furo_num)
            if current_shanten > added_shanten:
                result[0,:,0] = 1
        
        offset = 1
        target_channel = max(0,min(6,current_shanten)) + offset
        result[target_channel, :, 0] = 1

        

        
    