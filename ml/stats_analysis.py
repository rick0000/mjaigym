from typing import List, Dict
from mjaigym.board.mj_move import MjMove
from mjaigym.mjson import Mjson
from pathlib import Path
import math
import pprint
import json


class Stats():
    def __init__(self, player_name):
        self.player_name = player_name
        self.game_count = 0
        self.kyoku_count = 0
        self.hora_count = 0
        self.hora_point_sum = 0
        self.houjuu_count = 0
        self.houjuu_point_sum = 0
        self.rank_nums = [0] * 4
    
    def get_dict(self):
        average_rank, lower1sigma, upper1sigma = self.average_rank_1sigma
        _, lower2sigma, upper2sigma = self.average_rank_2sigma
        return {
            "player_name":self.player_name,
            "game_count":self.game_count,
            "kyoku_count":self.kyoku_count,
            "hora_count":self.hora_count,
            "hora_point_sum":self.hora_point_sum,
            "houjuu_count":self.houjuu_count,
            "houjuu_point_sum":self.houjuu_point_sum,
            "rank_nums":self.rank_nums,
            "average_rank":average_rank,
            "lower1sigma":lower1sigma,
            "upper1sigma":upper1sigma,
            "lower2sigma":lower2sigma,
            "upper2sigma":upper2sigma,
        }

    def append_game(
            self, 
            kyoku_count, 
            hora_count, 
            hora_point_sum, 
            houjuu_count, 
            houjuu_point_sum, 
            rank
        ):
            self.game_count += 1
            self.kyoku_count += kyoku_count
            self.hora_count += hora_count
            self.hora_point_sum += hora_point_sum
            self.houjuu_count += houjuu_count
            self.houjuu_point_sum += houjuu_point_sum
            
            assert 1 <= rank and rank <= 4
            self.rank_nums[rank-1] += 1

    @property
    def average_rank_2sigma(self):
        return self._calc_average_rank(2)

    @property
    def average_rank_1sigma(self):
        return self._calc_average_rank(1)

    def _calc_average_rank(self, coeff):
        average_rank = (self.rank_nums[0] * 1 +\
            self.rank_nums[1] * 2 +\
            self.rank_nums[2] * 3 +\
            self.rank_nums[3] * 4) / sum(self.rank_nums)
        
        variance = (
                ((1.0 - average_rank)**2) * self.rank_nums[0] +\
                ((2.0 - average_rank)**2) * self.rank_nums[1] +\
                ((3.0 - average_rank)**2) * self.rank_nums[2] +\
                ((4.0 - average_rank)**2) * self.rank_nums[3]
            ) / sum(self.rank_nums)

        sigma = math.sqrt(variance)

        return [average_rank, 
            average_rank - coeff * sigma, 
            average_rank + coeff * sigma]

class StatsAnalysis():

    def __init__(self):
        self.stats_dic:Dict[str,Stats] = {}

    def calclate_stats_from_dir(self, dir_path:Path):
        dir_path = Path(dir_path)
        mjson_paths = dir_path.glob('**/*.mjson')
        mjsons = [Mjson.load(p) for p in mjson_paths]
        self.calclate_stats(mjsons)
        self.dump_stats(dir_path)
        

    def calclate_stats(self, mjsons: List[Mjson]):
        for mjson in mjsons:
            self.calclate_stat(mjson)
        self.show_stats()

    def calclate_stat(self, mjson:Mjson):
        if mjson.game.contains_game_end == False:
            print(f"skip:{mjson.path}")
            return

        results = dict(zip(mjson.game.names, [{}]*len(mjson.game.names)))
        
        kyoku_count = len(mjson.game.kyokus)
        hora_counts = [0]*4
        hora_point_sums = [0]*4

        houjuu_counts = [0]*4
        houjuu_point_sums = [0]*4
        
        for kyoku in mjson.game.kyokus:
            diff = [r - i for (r,i) in zip(kyoku.result_scores, kyoku.initial_scores)]
            
            for end_line in kyoku.end_lines:
                if end_line['type'] == MjMove.hora.value:
                    actor = end_line['actor']
                    hora_counts[actor] += 1
                    hora_point_sums[actor] += diff[actor]

                    target = end_line['target']
                    if actor != target:
                        # print("houjuu count up, actor, target", actor, target)
                        houjuu_counts[target] += 1
                        houjuu_point_sums[target] += diff[target]

        for i, name in enumerate(mjson.game.names):
            if name not in self.stats_dic:
                stats = Stats(name)
            else:
                stats = self.stats_dic[name]

            stats.append_game(
                kyoku_count=kyoku_count,
                hora_count=hora_counts[i],
                hora_point_sum=hora_point_sums[i],
                houjuu_count=houjuu_counts[i],
                houjuu_point_sum=houjuu_point_sums[i],
                rank=mjson.game.ranks[i],
            )
            self.stats_dic[name] = stats


    def dump_stats(self, dir_path:Path):
        output_file = dir_path / "stats.json"

        converted = {}
        for k in self.stats_dic.keys():
            converted[k] = self.stats_dic[k].get_dict()

        with open(output_file, "wt") as f:
            f.write(json.dumps(converted, indent=4))


    def show_stats(self):
        converted = {}
        for k in self.stats_dic.keys():
            converted[k] = self.stats_dic[k].get_dict()
        pprint.pprint(converted)

    def get_stats(self, player_name:str):
        if player_name not in self.stats_dic:
            return Stats("")
        else:
            return self.stats_dic[player_name]


if __name__ == "__main__":
    stats_analysis = StatsAnalysis()
    stats_analysis.calclate_stats_from_dir(dir_path="output/sample/train")
    