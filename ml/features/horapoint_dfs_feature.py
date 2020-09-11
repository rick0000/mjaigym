import numpy as np
from .feature import Feature
from mjaigym.board.mj_move import MjMove
from mjaigym.board.board_state import BoardState
from mjaigym.board.function.pai import Pai
from mjaigym.board.function.dfs_stack import Dfs
from mjaigym.board.function.rs_shanten_analysis import RsShantenAnalysis


class HorapointDfsFeature():
    """this class has unusual interface, because handling dfs object cache.
    """
    shanten_analysis = RsShantenAnalysis()
    target_points = [3900,7700,12000]
    @classmethod
    def get_length(cls)->int:
        shanten_ch = 4 # 4 represents inputed shanten -1, 0, 1, more than 2
        return len(cls.target_points) + shanten_ch

    @classmethod
    def calc(cls, result:np.array, board_state:BoardState, player_id:int, dfs:Dfs=None):
        if dfs is None:
            dfs = Dfs()
        
        player_tehai = board_state.tehais[player_id]
        nums = [0] * 34
        for t in player_tehai:
            nums[t.id] += 1
        
        player_furos = board_state.furos[player_id]
        rest = board_state.restpai_in_view[player_id]
        rest_sum = sum(rest)
        current_shanten = cls.shanten_analysis.calc_shanten(nums, len(player_furos))
        
        
        # ignore -1, more than 2
        dfs_result = None
        
        depth = 2
        if current_shanten == 0:
            # Due to heavy cpu cost, not shanten improve change is allowed only 1 time.
            dfs_result = dfs.dfs_hora(depth, nums, player_furos, cls.target_points, rest)
        elif current_shanten == 1:
            dfs_result = dfs.dfs_hora(depth, nums, player_furos, cls.target_points, rest)
        
        if dfs_result:
            for i, point in enumerate(cls.target_points):
                result[i,:,0] = dfs_result[point]

        offset = 1
        shanten_index = len(cls.target_points) + min(2, current_shanten) # -1~2
        shanten_index += offset
        result[shanten_index,:,0] = 1

        
        