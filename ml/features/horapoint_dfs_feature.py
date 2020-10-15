import numpy as np
import itertools

from .feature import Feature
from mjaigym.board.mj_move import MjMove
from mjaigym.board.board_state import BoardState
from mjaigym.board.function.pai import Pai
# from mjaigym.board.function.dfs_stack import Dfs
from mjaigym.board.function.rs_shanten_analysis import RsShantenAnalysis
from mjaigym.board.function.efficient_dfs import Dfs
from mjaigym.board.function.yaku_name import YAKU_CHANNEL_MAP
import mjaigym.loggers as lgs


class HorapointDfsFeature(Feature):
    """this class has unusual interface, because handling dfs object cache.
    """
    shanten_analysis = RsShantenAnalysis()
    target_points = [3900,7700,12000]
    DEPTH = 2
    
    YAKU_CH = len(YAKU_CHANNEL_MAP) * DEPTH # depth 1, depth 2, depth 3.
    POINT_CH = len(target_points) * DEPTH
    ONE_PLAYER_LENGTH = YAKU_CH + POINT_CH

    @classmethod
    def get_length(cls)->int:
        
        return cls.ONE_PLAYER_LENGTH * 4
    


    @classmethod
    def calc(cls, result:np.array, board_state:BoardState, player_id:int, oracle_feature_flag:bool, dfs=Dfs()):
        
        for from_player_view, seat_alined_player_id in cls.get_seat_order_ids(player_id):
            
            if not oracle_feature_flag:
                if from_player_view != 0:
                    # print("skip, oracle feature disabled.")
                    continue

            player_tehai = board_state.tehais[seat_alined_player_id]

            

            nums = [0] * 34
            for t in player_tehai:
                nums[t.id] += 1
            tehai_akadora_num = len([p for p in player_tehai if p.is_red])

            player_furos = board_state.furos[seat_alined_player_id]
            
            # num 14 check
            if len(player_tehai) + len(player_furos) * 3 != 14:
                # return
                pass
            
            furo_akadora_num = 0
            for furo in player_furos:
                furo_akadora_num += len([p for p in furo.pais if p.is_red])

            
            # # ignore -1, more than 2
            dfs_result = None
            
            oya = board_state.oya == seat_alined_player_id
            bakaze = board_state.bakaze
            jikaze = board_state.jikaze[seat_alined_player_id]
            doras = [p.succ for p in Pai.from_list(board_state.dora_markers)]
            uradoras = []
            num_akadoras = tehai_akadora_num + furo_akadora_num
            
            shanten_normal, shanten_kokushi, shanten_chitoitsu = cls.shanten_analysis.calc_all_shanten(nums, len(player_furos))

            results = []
            if 0 <= shanten_normal <= cls.DEPTH-1:
                normal_results = dfs.dfs_with_score_normal(
                    nums,
                    player_furos,
                    cls.DEPTH,
                    oya=oya,
                    bakaze=bakaze,
                    jikaze=jikaze,
                    doras=doras,
                    uradoras=uradoras,
                    num_akadoras=num_akadoras,
                    shanten_normal=shanten_normal,
                )
                results.extend(normal_results)
        
            if 0 <= shanten_chitoitsu <= cls.DEPTH-1:
                chitoitsu_results = dfs.dfs_with_score_chitoitsu(
                    nums, 
                    player_furos, 
                    cls.DEPTH, 
                    oya=oya,
                    bakaze=bakaze,
                    jikaze=jikaze,
                    doras=doras,
                    uradoras=uradoras,
                    num_akadoras=num_akadoras,
                    shanten_chitoitsu=shanten_chitoitsu,
                )
                results.extend(chitoitsu_results)
                    
            if 0 <= shanten_kokushi <= cls.DEPTH-1:
                kokushi_results = dfs.dfs_with_score_kokushi(
                    nums, 
                    player_furos, 
                    cls.DEPTH, 
                    oya=oya,
                    shanten_kokushi=shanten_kokushi,
                )
                results.extend(kokushi_results)

            results = [r for r in results if r.valid()]

            if len(results) == 0:
                continue


            if from_player_view == 0:
                # ある牌を打牌(マイナス)した際に和了可能な役か。
                # プレーヤー（14枚形）の際に適用。
                for i in range(34):
                    i_dahaiable_horas = [r for r in results if r.is_dahaiable(i)]
                    if len(i_dahaiable_horas) == 0:
                        continue
                    
                    yaku_dist_set = set()
                    point_dist_set = set()
                    for hora in i_dahaiable_horas:
                        dist = hora.distance()
                        point = hora.get_point()
                        
                        for yaku in hora.get_yakus():
                            yaku_dist_set.add((yaku, dist))
                            point_dist_set.add((point, dist))
                    
                    for (yaku, dist) in yaku_dist_set:
                        # add yaku feature
                        if yaku in YAKU_CHANNEL_MAP:
                            target_channel = YAKU_CHANNEL_MAP[yaku] + ((dist-1) * len(YAKU_CHANNEL_MAP))
                            player_offset = from_player_view*cls.ONE_PLAYER_LENGTH
                            result[player_offset + target_channel,i,0] = 1

                    for (point, dist) in point_dist_set:
                        # add hora point feature
                        for point_index, target_point in enumerate(cls.target_points):
                            if point >= target_point:
                                target_channel = cls.YAKU_CH + point_index + (dist-1) * len(cls.target_points)
                                
                                player_offset = from_player_view*cls.ONE_PLAYER_LENGTH
                                result[player_offset + target_channel,i,0] = 1
                    

            else:
                # ある牌を追加した際に和了可能な役か。
                # 自分以外のプレーヤー（13枚系）の際に適用。
                
                # 比較のためターチャの特徴量は無視
                continue

                for i in range(34):
                    i_need_horas = [r for r in results if r.is_tsumoneed(i)]
                    if len(i_need_horas) == 0:
                        continue

                    yaku_dist_set = set()
                    for hora in i_need_horas:
                        dist = hora.distance()
                        point = hora.get_point()
                        
                        for yaku in hora.get_yakus():
                            yaku_dist_set.add((yaku, dist, point))
                    
                    for (yaku, dist, point) in yaku_dist_set:
                        # add yaku feature
                        if yaku in YAKU_CHANNEL_MAP:
                            target_channel = YAKU_CHANNEL_MAP[yaku] + ((dist-1) * len(YAKU_CHANNEL_MAP))
                            player_offset = from_player_view*cls.ONE_PLAYER_LENGTH
                            result[player_offset + target_channel,i,0] = 1
                        
                        # add hora point feature
                        for point_index, target_point in enumerate(cls.target_points):
                            if point >= target_point:
                                target_channel = cls.YAKU_CH + point_index + (dist-1) * len(cls.target_points)
                                
                                player_offset = from_player_view*cls.ONE_PLAYER_LENGTH
                                result[player_offset + target_channel,i,0] = 1
        