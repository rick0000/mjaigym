from typing import List, Dict
from .reward import Reward
from mjaigym.kyoku import Kyoku
from mjaigym.board import MjMove

class KyokuScoreReward(Reward):
    def __init__(self):
        self.kyoku_initial_scores = None
        self.current_scores = None
        self.current_mjson = None
        self.reset()

    def step(self, mjson):
        self.current_mjson = mjson

        if mjson["type"] == MjMove.start_kyoku.value:
            self.kyoku_initial_scores = self.current_scores

        if 'scores' in mjson:
            self.current_scores = mjson["scores"]

    def reset(self):
        self.kyoku_initial_scores = [25000] * 4
        self.current_scores = [25000] * 4
        self.current_mjson = {"type":"none"}

    def calc(self):
        if self.current_mjson["type"] != MjMove.end_kyoku.value:
            return [0,0,0,0]
        else:
            return [
                self.current_scores[0] - self.kyoku_initial_scores[0],
                self.current_scores[1] - self.kyoku_initial_scores[1],
                self.current_scores[2] - self.kyoku_initial_scores[2],
                self.current_scores[3] - self.kyoku_initial_scores[3],
            ]
    
    @classmethod
    def calc_from_kyoku(cls, kyoku:Kyoku):
        rewards = [[0] * 4 for _ in enumerate(kyoku.kyoku_mjsons)]
        
        initial_scores = kyoku.initial_scores
        end_scores = kyoku.result_scores
        diffs = [e - i for e, i in zip(end_scores, initial_scores)]
        # assert len(diffs) == 4

        # for double ron, reward is added at end_kyoku
        end_kyoku_indexs = [i for i, mjson in enumerate(kyoku.kyoku_mjsons) if mjson['type']==MjMove.end_kyoku.value]
        if len(end_kyoku_indexs) > 0:
            end_kyoku_index = end_kyoku_indexs[0]
            rewards[end_kyoku_index] = diffs

        return rewards
