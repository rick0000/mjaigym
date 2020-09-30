import numpy as np
from .feature import Feature
from mjaigym.board.mj_move import MjMove
from mjaigym.board.board_state import BoardState
from mjaigym.board.function.pai import Pai
# from mjaigym.board.function.dfs_stack import Dfs
from mjaigym.board.function.rs_shanten_analysis import RsShantenAnalysis
from mjaigym.board.function.efficient_dfs import Dfs

dfs_count = 0

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
    def calc(cls, result:np.array, board_state:BoardState, player_id:int, oracle_feature_flag:bool, dfs=Dfs()):
        
        player_tehai = board_state.tehais[player_id]
        nums = [0] * 34
        for t in player_tehai:
            nums[t.id] += 1
        tehai_akadora_num = len([p for p in player_tehai if p.is_red])

        player_furos = board_state.furos[player_id]
        if len(player_tehai) + len(player_furos) * 3 != 14:
            return
        furo_akadora_num = 0
        for furo in player_furos:
            furo_akadora_num += len([p for p in furo.pais if p.is_red])

        player_reach = board_state.reach[player_id]


        shanten_normal, shanten_kokushi, shanten_chitoitsu = cls.shanten_analysis.calc_all_shanten(nums, len(player_furos))
        
        
        
        # # ignore -1, more than 2
        dfs_result = None
        
        oya = board_state.oya == player_id
        bakaze = board_state.bakaze
        jikaze = board_state.jikaze[player_id]
        doras = [p.succ for p in Pai.from_list(board_state.dora_markers)]
        uradoras = []
        double_reach = board_state.double_reach[player_id]
        ippatsu = board_state.ippatsu[player_id]
        rinshan = board_state.rinshan[player_id]
        haitei = board_state.haitei[player_id]
        first_turn = board_state.first_turn
        chankan = board_state.chankan[player_id]
        num_akadoras = tehai_akadora_num + furo_akadora_num

        global dfs_count
        depth = 3
        if 0 <= shanten_normal <= depth-1:
            dfs_count += 1
            if dfs_count % 100 == 0:
                print(dfs_count)
            # Due to heavy cpu cost, not shanten improve change is allowed only 1 time.
            result = dfs.dfs_with_score_normal(
                nums, 
                player_furos, 
                depth, 
                reach=player_reach,
                oya=oya,
                bakaze=bakaze,
                jikaze=jikaze,
                doras=doras,
                uradoras=uradoras,
                double_reach=double_reach,
                ippatsu=ippatsu,
                rinshan=rinshan,
                haitei=haitei,
                first_turn=first_turn,
                chankan=chankan,
                num_akadoras=num_akadoras,
                shanten_normal=shanten_normal,
            )
            
            # mpsz_combinations = dfs.dfs(nums, len(player_furos), depth)
        if 0 <= shanten_chitoitsu <= depth-1:
            # Due to heavy cpu cost, not shanten improve change is allowed only 1 time.
            result = dfs.dfs_with_score_chitoitsu(
                nums, 
                player_furos, 
                depth, 
                reach=player_reach,
                oya=oya,
                bakaze=bakaze,
                jikaze=jikaze,
                doras=doras,
                uradoras=uradoras,
                double_reach=double_reach,
                ippatsu=ippatsu,
                rinshan=rinshan,
                haitei=haitei,
                first_turn=first_turn,
                chankan=chankan,
                num_akadoras=num_akadoras,
                shanten_chitoitsu=shanten_chitoitsu,
            )

        # if dfs_result:
        #     for i, point in enumerate(cls.target_points):
        #         result[i,:,0] = dfs_result[point]

        # offset = 1
        # shanten_index = len(cls.target_points) + min(2, current_shanten) # -1~2
        # shanten_index += offset
        # result[shanten_index,:,0] = 1

        # print(datetime.datetime.now(), "end")

        
        