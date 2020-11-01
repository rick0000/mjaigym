import numpy as np
from .feature import Feature
from mjaigym.board.mj_move import MjMove
from mjaigym.board.board_state import BoardState
from mjaigym.board.function.pai import Pai

class AnpaiFeature(Feature):
    
    @classmethod
    def get_length(cls)->int:
        return 4

    @classmethod
    def calc(cls, result:np.array, board_state:BoardState, player_id:int, oracle_enable_flag:bool=False):
        for i_from_player, i_raw in cls.get_seat_order_ids(player_id):
            anpais = board_state.anpais[i_raw]
            for pai in anpais:
                result[i_from_player, pai.id] = 1

