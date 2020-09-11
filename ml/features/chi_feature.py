import numpy as np
from .feature import Feature
from mjaigym.board.mj_move import MjMove
from mjaigym.board import BoardState

class ChiFeature(Feature):
    @classmethod
    def get_length(cls)->int:
        return 4

    @classmethod
    def calc(cls, result:np.array, board_state:BoardState, player_id:int, oracle_enable_flag:bool=False):
        for i_from_player, i_raw in cls.get_seat_order_ids(player_id):
            furos = [f for f in board_state.furos[i_raw] if f.type == MjMove.chi.value]
            counts = {}
            for f in furos:
                if f.pai_id not in counts:
                    counts[f.pai_id] = 0
                counts[f.pai_id] += 1

            for pai_id, num in counts.items():
                result[4*i_from_player:4*i_from_player+num, pai_id] = 1
    
        