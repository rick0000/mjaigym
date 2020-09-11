import numpy as np
from .feature import Feature
from mjaigym.board.mj_move import MjMove
from mjaigym.board import BoardState

class AnkanFeature(Feature):

    @classmethod
    def get_length(cls)->int:
        return 4

    @classmethod
    def calc(cls, result:np.array, board_state:BoardState, player_id:int, oracle_enable_flag:bool=False):
        for i_from_player, i_raw in cls.get_seat_order_ids(player_id):
            furos = [f for f in board_state.furos[i_raw] if f.type == MjMove.ankan.value]
            for furo in furos:
                result[i_from_player, furo.pai_id] = 1
