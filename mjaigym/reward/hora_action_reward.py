from typing import List, Dict

from .reward import Reward
from mjaigym.board import Board
from mjaigym.kyoku import Kyoku
from mjaigym.board import MjMove

class HoraActionReward(Reward):
    """ hora actor reward is 1, other is 0.
    no discount.
    """
    
    def __init__(self):
        self.hora_counts = None
        self.current_mjson = None
        self.reset()

    def reset(self):
        self.hora_counts = [0] * 4
        self.current_mjson = {"type":"none"}

    def step(self, mjson):
        self.current_mjson = mjson

        if mjson["type"] == MjMove.start_kyoku.value:
            self.hora_counts = [0] * 4

        if mjson["type"] == MjMove.hora.value:
            self.hora_counts[mjson["actor"]] += 1



    @classmethod
    def calc_from_kyoku(cls, kyoku:Kyoku):
        rewards = [[0] * 4 for _ in enumerate(kyoku.kyoku_mjsons)]
        
        hora_index_actors = [(i, mjson['actor']) for i, mjson in enumerate(kyoku.kyoku_mjsons) if mjson['type']==MjMove.hora.value]
        reward_counts = [0] * 4
        for _, hora_actor in hora_index_actors:
            reward_counts[hora_actor] = 1

        # for double ron, reward is added at end_kyoku
        end_kyoku_indexs = [i for i, mjson in enumerate(kyoku.kyoku_mjsons) if mjson['type']==MjMove.end_kyoku.value]
        if len(end_kyoku_indexs) > 0:
            end_kyoku_index = end_kyoku_indexs[0]
            rewards[end_kyoku_index] = reward_counts

        return rewards

    def calc(self):
        if self.current_mjson["type"] != MjMove.end_kyoku.value:
            return [0,0,0,0]
        else:
            return list(self.hora_counts)
