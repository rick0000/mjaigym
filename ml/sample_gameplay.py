import os
import re
from abc import ABCMeta, abstractclassmethod, abstractmethod
import typing

from dataclasses import dataclass
from gym import spaces
import numpy as np
from collections import deque

from mjaigym.board import Board, BoardState
from mjaigym.board.function.pai import Pai
from mjaigym.board.mj_move import MjMove
from ml.framework import SceneObservation, MjAgent, InnerAgent, MjObserver
from ml.custom_observer import SampleCustomObserver
from ml.model import  Head2Model, Head34Model


class SimpleObserver(MjObserver):
    """ returns simple structure and random observation
    """

    TSUMO_OBSERVE_CHANNELS = 1
    OTHERDAHAI_OBSERVE_CHANNELS = 1
    def trainsform_dahai(self, state, id):
        return np.random.randint(low=0,high=1,size=(self.TSUMO_OBSERVE_CHANNELS,34,1))
    def trainsform_reach(self, state, id):
        return np.random.randint(low=0,high=1,size=(self.TSUMO_OBSERVE_CHANNELS,34,1))
    def trainsform_chi(self, state, id, candidate):
        return np.random.randint(low=0,high=1,size=(self.OTHERDAHAI_OBSERVE_CHANNELS,34,1))
    def trainsform_pon(self, state, id, candidate):
        return np.random.randint(low=0,high=1,size=(self.OTHERDAHAI_OBSERVE_CHANNELS,34,1))
    def trainsform_kan(self, state, id, candidate):
        return np.random.randint(low=0,high=1,size=(self.OTHERDAHAI_OBSERVE_CHANNELS,34,1))
    
    def get_tsumo_observe_channels_num(self):
        return self.TSUMO_OBSERVE_CHANNELS

    def get_otherdahai_observe_channels_num(self):
        return self.OTHERDAHAI_OBSERVE_CHANNELS


class RandomAgent(InnerAgent):
    def __init__(self, actions, model_class, model_config, epsilon=0.0):
        super(RandomAgent, self).__init__(actions, model_class, model_config, epsilon)

    def policy(self, observation):
        if not self.initialized:
            self.initialize(observation)
        p = np.ones(self.actions.n)
        return p

    def estimate(self, observation):
        raise NotImplementedError()

    def initialize(self, observation):
        self.initialized = True


def main():
    board = Board(game_type="tonpu")
    from mjaigym.reward.kyoku_score_reward import KyokuScoreReward
    
    obs = SampleCustomObserver(board, KyokuScoreReward)
    
    from mjaigym.config import ModelConfig
    model_config = ModelConfig(resnet_repeat=10, mid_channels=128, learning_rate=0.001)
    
    actions = obs.action_space
    agent_class = RandomAgent
    dahai_agent = agent_class(actions["dahai_agent"], Head34Model, model_config)
    reach_agent = agent_class(actions["reach_agent"], Head2Model, model_config)
    chi_agent = agent_class(actions["chi_agent"], Head2Model, model_config)
    pon_agent = agent_class(actions["pon_agent"], Head2Model, model_config)
    kan_agent = agent_class(actions["kan_agent"], Head2Model, model_config)

    mj_agent = MjAgent(
        dahai_agent,
        reach_agent,
        chi_agent,
        pon_agent,
        kan_agent
        )
    
    # mj_agent.play(obs, episode_count=25600, render=False)
    mj_agent.play_multiprocess(obs, episode_count=25600, render=False)


if __name__ == "__main__":
    main()