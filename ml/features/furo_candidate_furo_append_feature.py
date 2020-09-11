import numpy as np
from typing import Dict
from .furo_append_feature import FuroAppendFeature
from mjaigym.board.mj_move import MjMove
from mjaigym.board.board_state import BoardState
from mjaigym.board.function.pai import Pai


class FuroCandidateFuroAppendFeature(FuroAppendFeature):
    
    @classmethod
    def get_length(cls)->int:
        return 5 # furotype:3, furopai:1, furo contains red:1

    @classmethod
    def calc(cls, result:np.array, board_state:BoardState, player_id:int, candidate_furo:Dict, oracle_enable_flag:bool=False):
        target_index = None
        if candidate_furo["type"] == MjMove.chi.value:
            target_index = 0
        elif candidate_furo["type"] == MjMove.pon.value:
            target_index = 1
        elif candidate_furo["type"] in [MjMove.kakan.value, MjMove.daiminkan.value, MjMove.ankan.value]:
            target_index = 2
        else:
            raise Exception("not intended path")
        result[target_index,:] = 1


        pais = []
        if candidate_furo["type"] == MjMove.ankan.value:
            pais += candidate_furo["consumed"]
        else:
            pais += ([candidate_furo["pai"]] + candidate_furo["consumed"])
        pais = Pai.from_list(pais)

        min_pai_id = min([pai.id for pai in pais])
        result[3,min_pai_id] = 1
        
        contains_red = any([pai.is_red for pai in pais])
        if contains_red:
            result[4,:] = 1

