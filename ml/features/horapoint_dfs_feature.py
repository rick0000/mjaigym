import numpy as np
from .feature import Feature
from mjaigym.board.mj_move import MjMove
from mjaigym.board.board_state import BoardState
from mjaigym.board.function.pai import Pai
# from mjaigym.board.function.dfs_stack import Dfs
from mjaigym.board.function.rs_shanten_analysis import RsShantenAnalysis
from mjaigym.board.function.efficient_dfs import Dfs
from mjaigym.board.function.yaku_name import YAKU_CHANNEL_MAP



class HorapointDfsFeature():
    """this class has unusual interface, because handling dfs object cache.
    """
    shanten_analysis = RsShantenAnalysis()
    target_points = [3900,7700,12000]
    
    @classmethod
    def get_length(cls)->int:
        yaku_ch = len(YAKU_CHANNEL_MAP) * 2 # depth 1, depth 2, depth 3.
        shanten_ch = len([0,1,2,3]) # represents inputed shanten 0, 1, 2, over3
        point_ch = len(cls.target_points)
        return yaku_ch + point_ch + shanten_ch

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

        
        
        
        # # ignore -1, more than 2
        dfs_result = None
        
        oya = board_state.oya == player_id
        bakaze = board_state.bakaze
        jikaze = board_state.jikaze[player_id]
        doras = [p.succ for p in Pai.from_list(board_state.dora_markers)]
        uradoras = []
        num_akadoras = tehai_akadora_num + furo_akadora_num

        depth = 3
        possible_yakus = set()

        shanten_normal, shanten_kokushi, shanten_chitoitsu = cls.shanten_analysis.calc_all_shanten(nums, len(player_furos))

        if 0 <= shanten_normal <= depth-1:
            normal_horas = dfs.dfs_with_score_normal(
                nums,
                player_furos,
                depth,
                oya=oya,
                bakaze=bakaze,
                jikaze=jikaze,
                doras=doras,
                uradoras=uradoras,
                num_akadoras=num_akadoras,
                shanten_normal=shanten_normal,
            )
            for hora in normal_horas:
                hora_info = hora[1]
                yakus = [h[0] for h in hora_info["yakus"]]
                possible_yakus = possible_yakus | set(yakus)
    
        
        if 0 <= shanten_chitoitsu <= depth-1:
            # Due to heavy cpu cost, not shanten improve change is allowed only 1 time.
            chitoitsu_horas = dfs.dfs_with_score_chitoitsu(
                nums, 
                player_furos, 
                depth, 
                oya=oya,
                bakaze=bakaze,
                jikaze=jikaze,
                doras=doras,
                uradoras=uradoras,
                num_akadoras=num_akadoras,
                shanten_chitoitsu=shanten_chitoitsu,
            )
            for hora in chitoitsu_horas:
                
                hora_info = hora[0]
                yakus = [h[0] for h in hora_info["yakus"]]
                possible_yakus = possible_yakus | set(yakus)
        
        if 0 <= shanten_kokushi <= depth-1:
            # Due to heavy cpu cost, not shanten improve change is allowed only 1 time.
            kokushi_horas = dfs.dfs_with_score_kokushi(
                nums, 
                player_furos, 
                depth, 
                oya=oya,
                shanten_kokushi=shanten_kokushi,
            )
            for hora in kokushi_horas:
                hora_info = hora[0]
                yakus = [h[0] for h in hora_info["yakus"]]
                possible_yakus = possible_yakus | set(yakus)

        for possible_yaku in possible_yakus:
            if possible_yaku in YAKU_CHANNEL_MAP:
                target_channel = YAKU_CHANNEL_MAP[possible_yaku]
                result[target_channel,:,0] = 1
        # offset = 1
        # shanten_index = len(cls.target_points) + min(2, current_shanten) # -1~2
        # shanten_index += offset
        # result[shanten_index,:,0] = 1
        # print(datetime.datetime.now(), "end")

        
