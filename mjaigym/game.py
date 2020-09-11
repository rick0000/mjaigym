import gzip
from collections import OrderedDict

import numpy as np

from .kyoku import Kyoku

class Game():
    def __init__(self, mjsons):
        self.kyokus = []
        self.game_result_label = None
        self.game_result_score = None
        self.ranks = None
        self.is_tonnnann = False
        self.game_start_line = None
        self.contains_game_end = False
        self.path = None
        self.final_lines = None
        self.names = ['p0','p1','p2','p3']
        kyoku_lines = []
        kyoku_initial_scores = [25000,25000,25000,25000]
        next_kyoku_initial_scores = [25000,25000,25000,25000]
        final_lines = list(filter(lambda x: x['type'] == 'hora' or x['type'] == 'ryukyoku', mjsons))
        self.final_lines = final_lines

        if len(final_lines) == 0:
            print('result num is 0')
            return

        final_score = final_lines[-1]['scores']
        self.game_result_score = final_score
        self.game_result_label = Game.get_final_scores_label(final_score)
        self.ranks = Game.get_rank(final_score)
        last_bakaze = 'E'
        last_kyoku_id = 1

        for i, mjson in enumerate(mjsons):
            line = mjsons[i]
            kyoku_lines.append(line)

            if line['type'] == 'start_game':
                self.game_start_line = line
                kyoku_lines = [] # remove start game line
                if 'names' in line:
                    self.names = line['names']

            elif line['type'] == 'start_kyoku':
                kyoku_initial_scores = next_kyoku_initial_scores
                last_bakaze = line['bakaze']
                last_kyoku_id = line['kyoku']
            elif line['type'] == 'ryukyoku' or line['type'] == 'hora':
                next_kyoku_initial_scores = line['scores']
            elif line['type'] == 'end_kyoku':
                kyoku = Kyoku(kyoku_lines, kyoku_initial_scores)
                self.kyokus.append(kyoku)
                kyoku_lines = []
            elif line['type'] == 'end_game':
                self.contains_game_end = True

        if (last_bakaze == 'S' and last_kyoku_id == 4) or\
            last_bakaze == 'W':
            self.is_tonnnann = True

    def _load_gzip(self, path):
        try:
            with gzip.open(path, 'rb') as f:
                return f.read().decode('utf-8-sig').splitlines()
        except:
            return []

    def _load_mjson(self, path):
        try:
            with open(path, 'rt') as f:
                return f.read().splitlines()
        except:
            return []
    
    @classmethod
    def _calc_rank_sorted(cls, scores):
        if len(scores) != 4:
                raise Exception("invalid length scores")
        d = {
            0:scores[0],
            1:scores[1],
            2:scores[2],
            3:scores[3],
        }
        orderd = OrderedDict(sorted(d.items(), key=lambda x:x[1], reverse=True))
        return orderd
        

    @classmethod
    def get_rank(cls, scores):
        orderd = cls._calc_rank_sorted(scores)
        orderd_key = list(orderd.keys())
        ranks = [None] * 4
        for i in range(4):
            ranks[orderd_key[i]] = i+1
        return ranks

    @classmethod
    def get_final_scores_label(cls, scores):
        orderd = cls._calc_rank_sorted(scores)
        orderd_key = list(orderd.keys())
        return Game._ranks.index(orderd_key)
    _ranks = [
        [0,1,2,3],
        [0,1,3,2],
        [0,2,1,3],
        [0,2,3,1],
        [0,3,1,2],
        [0,3,2,1],
        
        [1,0,2,3],
        [1,0,3,2],
        [1,2,0,3],
        [1,2,3,0],
        [1,3,0,2],
        [1,3,2,0],

        [2,0,1,3],
        [2,0,3,1],
        [2,1,0,3],
        [2,1,3,0],
        [2,3,0,1],
        [2,3,1,0],
        
        [3,0,2,1],
        [3,0,1,2],
        [3,1,0,2],
        [3,1,2,0],
        [3,2,0,1],
        [3,2,1,0],
    ]
        
