import numpy as np
from .feature import Feature
from mjaigym.board.mj_move import MjMove
from mjaigym.board.board_state import BoardState
from mjaigym.board.function.pai import Pai


class KyokuInfoFeature(Feature):
    
    @classmethod
    def get_length(cls)->int:
        return 4 + 10 + 9 

    @classmethod
    def calc(cls, result:np.array, board_state:BoardState, player_id:int, oracle_enable_flag:bool=False):
        kyoku = board_state.kyoku # 1-4
        result[kyoku-1,:,0] = 1

        kyotaku = int(np.clip(board_state.kyotaku-1, 0, 9))
        kyotaku_start_index = 4
        result[kyotaku_start_index+kyotaku, :, 0] = 1

        honba_start_index = kyotaku_start_index + 10
        honba = int(np.clip(board_state.honba, 0, 8))
        result[honba_start_index+honba, :, 0] = 1


    