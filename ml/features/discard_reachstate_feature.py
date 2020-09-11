import numpy as np
from .feature import Feature
from mjaigym.board.mj_move import MjMove
from mjaigym.board.board_state import BoardState

class DiscardReachStateFeature(Feature):
    ONE_PLAYER_LENGTH = 24
    LENGTH = 4 * ONE_PLAYER_LENGTH


    @classmethod
    def get_length(cls)->int:
        return cls.ONE_PLAYER_LENGTH * 4

    @classmethod
    def calc(cls, result:np.array, board_state:BoardState, player_id:int, oracle_enable_flag:bool=False):
        for i_from_player, i_raw in cls.get_seat_order_ids(player_id):
            if board_state.reach_sutehais_index[i_raw] is not None:
                tsumogiri_start_index = i_from_player*cls.ONE_PLAYER_LENGTH + board_state.reach_sutehais_index[i_raw] + 1
                end_index = i_from_player*cls.ONE_PLAYER_LENGTH + len(board_state.sutehais[i_raw])
                if tsumogiri_start_index < end_index:
                    result[tsumogiri_start_index:end_index,:] = 1
