import numpy as np
from .feature import Feature
from mjaigym.board.mj_move import MjMove
from mjaigym.board.board_state import BoardState

class DiscardFeature(Feature):
    ONE_PLAYER_LENGTH = 24
    @classmethod
    def get_length(cls)->int:
        return cls.ONE_PLAYER_LENGTH * 4

    @classmethod
    def calc(cls, result:np.array, board_state:BoardState, player_id:int, oracle_enable_flag:bool=False):
        for i_from_player, i_raw in cls.get_seat_order_ids(player_id):
            for j, pai in enumerate(board_state.sutehais[i_raw]):
                result[i_from_player*cls.ONE_PLAYER_LENGTH + j, pai.id] = 1

    