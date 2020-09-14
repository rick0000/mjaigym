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
from pathlib import Path
import time

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
from mjaigym.tcp.server import Server
from ml.agent_tcp_wrapper import AgentTcpWrapper
import mjaigym.loggers as lgs
from mjaigym.reward.kyoku_score_reward import KyokuScoreReward
from ml.custom_observer import SampleCustomObserver
from mjaigym.config import ModelConfig
from ml.model import  Head2SlModel, Head34SlModel
from ml.agent import MjAgent, InnerAgent
from mjaigym.board import ClientBoard
from mjaigym.agent import MaxUkeireMjAgent 


def on_message(mes, agent_name):
    print(f"{agent_name}<-server", mes)

def on_send(mes, agent_name):
    print(f"{agent_name}->server", mes)

def on_message_noshow(mes, agent_name):
    pass

def on_send_noshow(mes, agent_name):
    pass


class Evaluator():
    @classmethod
    def evaluate(cls, agents, env, episode_count, render=False, use_multiprocess=False):
        params = [(render, agents, copy.deepcopy(env)) for i in range(episode_count)]
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

    
        for result in results:
            result.dump("output/sample/test")

        return list(results)

    @classmethod
    def _play_one(cls, args):
        render = args[0]
        agents = args[1]
        env = args[2]

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


    @classmethod
    def tcp_evaluate(cls, agents, env, episode_count):
        port = 48000
        client_limit = 32
        logdir = Path("evaluate") / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        room_name = "self_play"
        
        server = Server(port, client_limit, logdir)
        server.run_async()

        # make clients
        for i in range(episode_count):
            lgs.logger_main.info(f"start episode {i}")
            client_threads = []
            for j, agent in enumerate(agents):
                agent_wrapper = AgentTcpWrapper(
                    "localhost",
                    port,
                    "pass",
                    env=copy.deepcopy(env),
                    agent=agent,
                    name=f"agent{j}",
                    room_name=room_name+str(i),
                    on_message_handler=on_message,
                    on_send_handler=on_send,
                )
                agent_wrapper.run()
                client_threads.append(agent_wrapper.recieve_thread)

            for client_thread in client_threads:
                client_thread.join()

            lgs.logger_main.info(f"end episode {i}")

        while len(server.rooms) > 0:
            time.sleep(1) # wait for tcp server paifu dump

    @classmethod
    def tcp_evaluate_manue(cls, agent, env, episode_count):
        port = 48000
        client_limit = 32
        logdir = Path("evaluate") / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        room_name = "manue"
        
        server = Server(port, client_limit, logdir)
        server.run_async()

        # make clients
        for i in range(episode_count):
            lgs.logger_main.info(f"start episode {i}")
            
            agent_wrapper = AgentTcpWrapper(
                "localhost",
                port,
                "pass",
                env=copy.deepcopy(env),
                agent=agent,
                name=f"agent",
                room_name=room_name+str(i),
            )
            agent_wrapper.run()
            agent_wrapper.recieve_thread.join()

            lgs.logger_main.info(f"end episode {i}")

        while len(server.rooms) > 0:
            time.sleep(1) # wait for tcp server paifu dump

        



def local_nn_agent():
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
        
    

def loacl_rulebase_agent():
    board = Board(game_type="one_kyoku")
    obs = SampleCustomObserver(board, KyokuScoreReward)
    
    agents = [
        MaxUkeireMjAgent(id=0, name="player0"),
        MaxUkeireMjAgent(id=1, name="player1"),
        MaxUkeireMjAgent(id=2, name="player2"),
        MaxUkeireMjAgent(id=3, name="player3"),
    ]

    evaluate_result = Evaluator.evaluate(agents, obs, 128)
    

def tcp_evaluate():
    board = ClientBoard()
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

    Evaluator.tcp_evaluate(agents, obs, 1)



def tcp_evaluate_manue():
    board = ClientBoard()
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
    
    Evaluator.tcp_evaluate_manue(mj_agent, obs, 1)

if __name__ == "__main__":
    tcp_evaluate()