import os
import re
from abc import ABCMeta, abstractmethod
import typing
import copy
import random
import multiprocessing
import json
import datetime
import uuid

import torch
if torch.cuda.is_available():
    from torch import multiprocessing
    from torch.multiprocessing import Pool, Process, set_start_method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
else:
    from multiprocessing import Pool, Process, set_start_method
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
from gym import spaces
import numpy as np
from collections import deque, namedtuple
from tqdm import tqdm

from mjaigym.board import Board, BoardState
from mjaigym.board.function.pai import Pai
from mjaigym.board.mj_move import MjMove
from ml.model import Model


class Evaluator():
    @classmethod
    def evaluate(cls, agents, env, episode_count, render=False, use_multiprocess=False):
        params = [(copy.deepcopy(env), render, agents) for i in range(episode_count)]
        results = deque()

        if use_multiprocess:
            with multiprocessing.get_context('spawn').Pool(processes=multiprocessing.cpu_count()) as pool:
                with tqdm(total=episode_count) as t:
                    for result in pool.imap_unordered(cls._play_one, params):
                        results.append(result)
                        t.update(1)
                pool.close()
                pool.terminate()
        else:
            for param in tqdm(params):
                result = cls._play_one(param)
                results.append(result)

        return list(results)

    @classmethod
    def _play_one(cls, args):
        env = args[0]
        render = args[1]
        agents = args[2]

        random.shuffle(agents)
        state, reward, done, info = env.reset()
        done = False
        while not done:
            if render:
                env.render()
            
            actions = {}
            for player_id in range(4):
                player_possible_actions = info["possible_actions"][player_id]
                board_state = info["board_state"]
                player_observation = state[player_id]
                actions[player_id] = agents[player_id].think_one_player(player_observation, player_possible_actions, player_id, board_state)

            n_state, reward, done, info = env.step(actions)
            state = n_state

        return env




def self_play_nn_agent():
    from mjaigym.reward.kyoku_score_reward import KyokuScoreReward
    from ml.custom_observer import SampleCustomObserver
    from mjaigym.config import ModelConfig
    from ml.model import  Head2SlModel, Head34SlModel
    from ml.agent import MjAgent, InnerAgent
    from .evaluator import Evaluator

    board = Board(game_type="one_kyoku")
    obs = SampleCustomObserver(board, KyokuScoreReward)
    model_config = ModelConfig(resnet_repeat=40, mid_channels=128, learning_rate=0.001)
    actions = obs.action_space
    agent_class = InnerAgent
    dahai_agent = agent_class(actions["dahai_agent"], Head34SlModel, model_config)
    reach_agent = agent_class(actions["reach_agent"], Head2SlModel, model_config)
    chi_agent = agent_class(actions["chi_agent"], Head2SlModel, model_config)
    pon_agent = agent_class(actions["pon_agent"], Head2SlModel, model_config)
    kan_agent = agent_class(actions["kan_agent"], Head2SlModel, model_config)

    mj_agent = MjAgent(
        dahai_agent,
        reach_agent,
        chi_agent,  
        pon_agent,
        kan_agent
        )
    
    agents = [mj_agent] * 4

    evaluate_result = Evaluator.evaluate(agents, obs, 1024)
    




def self_play_rulebase_agent():
    from mjaigym.reward.kyoku_score_reward import KyokuScoreReward
    from ml.custom_observer import SampleCustomObserver
    from mjaigym.config import ModelConfig
    from ml.model import  Head2SlModel, Head34SlModel
    from ml.agent import MjAgent, InnerAgent, MaxUkeireMjAgent
    from .evaluator import Evaluator

    board = Board(game_type="one_kyoku")
    obs = SampleCustomObserver(board, KyokuScoreReward)
    
    agents = [
        MaxUkeireMjAgent(id=0, name="player0"),
        MaxUkeireMjAgent(id=1, name="player1"),
        MaxUkeireMjAgent(id=2, name="player2"),
        MaxUkeireMjAgent(id=3, name="player3"),
    ]

    evaluate_result = Evaluator.evaluate(agents, obs, 128)
    
    
    for result in evaluate_result:
        result.dump("output/sample/test")


if __name__ == "__main__":
    self_play_rulebase_agent()