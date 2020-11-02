""" module for paifu supervised learning.
"""
from collections import namedtuple, deque
from pathlib import Path
import copy
import math
import gc
import random
import typing
import pickle
import datetime
import itertools
import time
import os
import joblib

from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, Process, set_start_method, Queue

from mjaigym.board.function.pai import Pai
from mjaigym.board.mj_move import MjMove
from mjaigym.mjson import Mjson
from ml.framework import MjObserver, TensorBoardLogger
from mjaigym.board import ArchiveBoard
from mjaigym.reward import KyokuScoreReward
from .custom_observer import SampleCustomObserver
import mjaigym.loggers as lgs
from ml.framework import Experience
from mjaigym.config import ModelConfig
from ml.model import  Head2SlModel, Head34SlModel, Head34Value1SlModel
from ml.agent import InnerAgent, MjAgent, DahaiTrainableAgent, FixPolicyAgent, DahaiActorCriticAgent
from ml.s_a_r_generator import StateActionRewardGenerator


@dataclass
class StateActionRewards:
    dahai_queue:deque
    reach_queue:deque
    chi_queue:deque
    pon_queue:deque
    kan_queue:deque

    @classmethod
    def create_empty(cls, length):
        return StateActionRewards(
            deque(maxlen=length),
            deque(maxlen=length),
            deque(maxlen=length),
            deque(maxlen=length),
            deque(maxlen=length)
            )

    def register_experience_to_sars(self, experiences:deque):
        for experience in experiences:
            for i in range(4):
                player_state = experience.state[i]
                # create dahai s_a_r
                if player_state.dahai_observation is not None\
                    and not experience.board_state.reach[i]\
                    and experience.action["type"] == MjMove.dahai.value:

                    label = Pai.str_to_id(experience.action["pai"])

                    self.dahai_queue.append(tuple((
                        player_state.dahai_observation,
                        label,
                        reward,
                    )))
                
                    # create reach ...
    


class SlTrainer():
    def __init__(
            self,
            train_dir,
            test_dir,
            log_dir,
            session_name,
            in_on_tsumo_channels,
            in_other_dahai_channels,
            use_multiprocess=True,
            udpate_interbal=64,
            reward_discount_rate=0.99,
            batch_size=256,
            evaluate_per_update=5
        ):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.use_multiprocess = use_multiprocess
        self.udpate_interbal = udpate_interbal
        self.in_on_tsumo_channels = in_on_tsumo_channels
        self.in_other_dahai_channels = in_other_dahai_channels
            

        self.mjson_path_queue = Queue(256)
        self.experiences = Queue()

        self.reward_discount_rate = reward_discount_rate
        
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.evaluate_per_update = evaluate_per_update
        self.session_name = session_name
        self.session_dir = Path(self.log_dir) / self.session_name
        
        self.tfboard_logger = TensorBoardLogger(log_dir=self.log_dir, session_name=self.session_name)
        

    def train_loop(self, agent, env, load_model=None):

        processses = []
        # run file glob process
        
        test_size = 1024
        test_s_a_rs = self.prepare_test_data(self.test_dir, env, test_size, env.get_tsumo_observe_channels_num())
        
        # prepare to generator, queue
        experiences_queue = Queue()
        s_a_rs_generator = StateActionRewardGenerator(
            self.train_dir,
            experiences_queue,
            sampling_rate=0.1,
            reward_discount_rate=0.99
        )
        process_num = multiprocessing.cpu_count()
        s_a_rs_generator.start(env, process_num)

        # run consume process
        self.consume_data(experiences_queue, agent, test_s_a_rs, load_model=load_model)
        

    def consume_data(
            self, 
            experience_queue:Queue, 
            agent:MjAgent, 
            test_s_a_rs:StateActionRewards, 
            load_model=None
            ):
        s_a_rs = StateActionRewards.create_empty(self.batch_size*10)

        game_count = 0
        dahai_update_count = 0
        # consume first observation for load
        if load_model:
            agent.load(load_model, self.in_on_tsumo_channels, self.in_other_dahai_channels)

        while True:
            
            if experience_queue == 0:
                time.sleep(0.5)
                continue
            
            experiences = experience_queue.get()
            game_count += 1
            s_a_rs.register_experience_to_sars(experiences)

            # train model
            if len(s_a_rs.dahai_queue) == s_a_rs.dahai_queue.maxlen:
                dahai_update_count += 1
                lgs.logger_main.info("start dahai train")
                self.train_dahai(s_a_rs.dahai_queue, agent, game_count)
                
                s_a_rs.dahai_queue.clear()
                lgs.logger_main.info(f"end train")
                
                if dahai_update_count % 20 == 0:
                    # save
                    agent.save(self.session_dir, game_count)

                if dahai_update_count % 5 == 0:
                    self.evaluate_dahai(test_s_a_rs.dahai_queue, agent, game_count)

            else:
                pass
                print(f"dahai {len(s_a_rs.dahai_queue)}/{s_a_rs.dahai_queue.maxlen}", end='\r')

    def prepare_test_data(self, test_dir, env, test_size, tsumo_observe_channels_num):
        cache_file_name = f"cache/test_{test_size}_{tsumo_observe_channels_num}"
        try:
            lgs.logger_main.info("load cached test data")
            return joblib.load(cache_file_name)
        except:
            lgs.logger_main.info("failed to load")
            pass
        lgs.logger_main.info("create cached test data")
        result = self._prepare_test_data(test_dir, env, test_size)

        try:
            os.makedirs(os.path.dirname(cache_file_name), exist_ok=True)
            joblib.dump(result, cache_file_name)
        except:
            print("fail to save change cache")
        return result
        
    def _prepare_test_data(self, test_dir, env, test_size):

        test_experiences_queue = Queue()
        test_s_a_rs_generator = StateActionRewardGenerator(
            test_dir,
            test_experiences_queue,
            sampling_rate=0.01,
            reward_discount_rate=0.99
        )
        test_s_a_rs_generator.start(env, 1)
        
        # prepare test data (refactor to function)
        test_s_a_rs = StateActionRewards.create_empty(test_size)
        
        while len(test_s_a_rs.dahai_queue) < test_size:
            experiences = test_experiences_queue.get()
            test_s_a_rs.register_experience_to_sars(experiences)
            print(f"test data prepare... {len(test_s_a_rs.dahai_queue)}/{test_size}", end="\r")
        test_s_a_rs_generator.terminate()
        return test_s_a_rs

    def train_dahai(
            self,
            dahai_state_action_rewards:StateActionRewards,
            agent:MjAgent,
            game_count:int):
        result = agent.update_dahai(dahai_state_action_rewards)
        lgs.logger_main.info("update result")
        for key, value in result.items():
            lgs.logger_main.info(f"{key}:{value:.03f}")
        for key, value in result.items():
            self.tfboard_logger.write(key, value, game_count)
        rs = np.array([r[2] for r in dahai_state_action_rewards])
        lgs.logger_main.info(f"rewards var:{np.var(rs):.03f}, max:{rs.max():.03f}, min:{rs.min():.03f}, mean:{rs.mean():.03f}")
                

    def evaluate_dahai(
            self,
            dahai_state_action_rewards:StateActionRewards,
            agent:MjAgent,
            game_count:int):
        lgs.logger_main.info("-------------------------------------")
        lgs.logger_main.info("test result")
        result = agent.evaluate_dahai(dahai_state_action_rewards)
        for key, value in result.items():
            lgs.logger_main.info(f"{key}:{value:.03f}")
        
        for key, value in result.items():
            self.tfboard_logger.write(key, value, game_count)
        rs = np.array([r[2] for r in dahai_state_action_rewards])
        lgs.logger_main.info(f"rewards var:{np.var(rs):.03f}, max:{rs.max():.03f}, min:{rs.min():.03f}, mean:{rs.mean():.03f}")
        lgs.logger_main.info("-------------------------------------")
        



if __name__ == "__main__":
    train_dir = "/data/mjson/train"
    test_dir = "/data/mjson/test"
    log_dir ="/mnt/sdc/experiments/output/logs"
    session_name = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    model_config = ModelConfig(
            resnet_repeat=20,
            mid_channels=256,
            learning_rate=10**-4,
            batch_size=256,
        )
    model_config.save(Path(log_dir)/session_name/"config.yaml")
    
    env = SampleCustomObserver(board=ArchiveBoard(), reward_calclator_cls=KyokuScoreReward, oracle_rate=1.0)
    actions = env.action_space
    
    sl_trainer = SlTrainer(
            train_dir,
            test_dir,
            log_dir=log_dir,
            session_name=session_name,
            in_on_tsumo_channels=env.get_tsumo_observe_channels_num(),
            in_other_dahai_channels=env.get_otherdahai_observe_channels_num(),
            use_multiprocess=True,
            udpate_interbal=64,
            batch_size=model_config.batch_size,
            evaluate_per_update=10
        )
    
    
    dahai_agent = DahaiActorCriticAgent(actions["dahai_agent"], Head34Value1SlModel, model_config)
    reach_agent = FixPolicyAgent(np.array([0.0, 1.0])) # always do reach
    chi_agent = FixPolicyAgent(np.array([1.0, 0.0])) # never chi
    pon_agent = FixPolicyAgent(np.array([1.0, 0.0])) # never pon
    kan_agent = FixPolicyAgent(np.array([1.0, 0.0])) # never kan
    mj_agent = MjAgent(dahai_agent, reach_agent, chi_agent, pon_agent, kan_agent)
    
    # load_dir = Path("/home/rick/dev/python/mjaigym/output/logs/20201009_211309/000730")
    load_dir = None
    sl_trainer.train_loop(mj_agent, env, load_dir)