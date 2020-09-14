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
from ml.agent import MaxUkeireMjAgent


def on_message(mes, agent_name):
    print(f"{agent_name}<-server", mes)

def on_send(mes, agent_name):
    print(f"{agent_name}->server", mes)

def on_message_noshow(mes, agent_name):
    pass

def on_send_noshow(mes, agent_name):
    pass


class OnlineEvaluator():

    @classmethod
    def tcp_evaluate_manue(cls, agent, env, episode_count):
        host = "game.mjaigym.net"
        # host = "localhost"
        port = 48000
        client_limit = 32
        logdir = Path("evaluate") / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        room_name = "manue"
        
        # make clients
        for i in range(episode_count):
            lgs.logger_main.info(f"start episode {i}")
            
            agent_wrapper = AgentTcpWrapper(
                host,
                port,
                "pass",
                env=copy.deepcopy(env),
                agent=agent,
                name=f"agent",
                room_name=room_name+str(i),
                on_message_handler=on_message,
                on_send_handler=on_send,
            )
            agent_wrapper.run()
            agent_wrapper.recieve_thread.join()

            lgs.logger_main.info(f"end episode {i}")




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
    
    OnlineEvaluator.tcp_evaluate_manue(mj_agent, obs, 1)

if __name__ == "__main__":
    tcp_evaluate_manue()