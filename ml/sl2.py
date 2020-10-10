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


class Memory:
    def __init__(self, length=256*100):
        self.queue = Queue(256*100)

    def append(self, data):
        self.queue.put(data)

    def consume(self):
        return self.queue.get()

    def __len__(self):
        return self.queue.qsize()


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
            

        self.mjson_path_queue = Memory(256)
        self.experiences = Memory()

        self.reward_discount_rate = reward_discount_rate
        
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.evaluate_per_update = evaluate_per_update
        self.session_name = session_name
        self.session_dir = Path(self.log_dir) /self.session_name
        
        self.tfboard_logger = TensorBoardLogger(log_dir=self.log_dir, session_name=self.session_name)
        

    def train_loop(self, agent, env, load_model=None):



        generate_proc_num = multiprocessing.cpu_count()
        generate_proc_num = max(1, generate_proc_num)
        # generate_proc_num = 1

        processses = []
        # run file glob process
        mjson_path = "/data/mjson/train/"
        
        grob_process = Process(
            target=self.generate_mjson_path, 
            args=(self.mjson_path_queue, mjson_path),
            daemon=True)
        grob_process.start()
        processses.append(grob_process)
        # run generate process
        
        # debug
        # self.generate_data(0,self.mjson_path_queue, self.experiences, env)
        
        for i in range(generate_proc_num):
            p = Process(
                target=self.generate_data, 
                args=(i, self.mjson_path_queue, self.experiences, env),
                daemon=True)
            p.daemon
            p.start()
            processses.append(p)
        
        # run consume process
        self.consume_data(self.experiences, agent, load_model=load_model)
        # p = Process(target=self.consume_data, args=(self.experiences, agent))
        # p.start()
        # processses.append(p)
        
        # wait process finish
        for p in processses:
            p.join()

    def generate_mjson_path(self, mjson_memory:Memory, mjson_dir):
        mjson_paths = Path(self.train_dir).glob("**/*.mjson")
        for mjson_path in mjson_paths:
            # this function blocks when full
            mjson_memory.append(mjson_path)


    def generate_data(self, process_number:int, input_memory:Memory, experience_memory:Memory, env):
        while True:
            # print(f"generate data@{process_number}")
            mjson_path = input_memory.consume()
            experiences = self._analyze_one_game((mjson_path, copy.deepcopy(env)))
            
            if len(experience_memory) > 64:
                # print("full queue, remove sample")
                experience_memory.consume()
            experience_memory.append(experiences)
        
            # print(f"experience_memory:{len(experience_memory)}")
            time.sleep(0.1)

    def consume_data(self, experience_memory:Memory, agent:MjAgent, load_model=None):
        game_chunk_experiences = deque(maxlen=64)
        onetime_update_samples = 6
        update_count = 0
        # consume first observation for load
        if load_model:
            agent.load(load_model, self.in_on_tsumo_channels, self.in_other_dahai_channels)

        while True:
            if len(experience_memory) > 0:
                update_count += 1
                # train model
                lgs.logger_main.info("start train")
                for i in range(onetime_update_samples):
                    sample = experience_memory.consume()
                    game_chunk_experiences.append(sample)
                lgs.logger_main.info(f"add {onetime_update_samples} new samples")


                # """ skip fill buffer check
                if len(game_chunk_experiences) != game_chunk_experiences.maxlen:
                    lgs.logger_main.info(f"game buffer not full {len(game_chunk_experiences)}/{game_chunk_experiences.maxlen}, end train...")
                    continue
                

                self.train(game_chunk_experiences, agent, update_count)
                lgs.logger_main.info(f"end train")
                if update_count % 10 == 0:
                    agent.save(self.session_dir,update_count)
            
            time.sleep(0.1)
                
    def train(self, game_chunk_experiences, agent:MjAgent, update_count):
        flatten_experiences = list(itertools.chain.from_iterable(game_chunk_experiences))
        update_result = agent.update(flatten_experiences)
    
        for key, value in update_result.items():
            self.tfboard_logger.write(key, value, update_count)


    def _analyze_one_game(self, args):
        try:
            mjson_path, env = args
            
            one_game_experience = []
             
            mjson = Mjson.load(mjson_path)
            for kyoku in mjson.game.kyokus:

                state, reward, done, info = env.reset()
                
                states = []
                actions = []
                rewards = []
                board_states = []
                for action in kyoku.kyoku_mjsons:
                    next_state, reward, done, info = env.step(action)
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    board_states.append(info["board_state"])
                    
                    state = next_state

                if len(rewards) == 0:
                    continue

                # calclate kyoku discount rewards.
                last_reward = np.array(rewards[-1])
                one_kyoku_length = len(kyoku.kyoku_mjsons)
                reversed_discount_rewards = [last_reward * math.pow(self.reward_discount_rate, i) for i in range(one_kyoku_length)]
                discount_rewards = list(reversed(reversed_discount_rewards))
                # 1000 diff as loss 1
                discount_rewards = [d/1000.0 for d in discount_rewards]

                reward_dicided_experience = [Experience(states[i], actions[i], discount_rewards[i][actions[i]["actor"]], board_states[i])  for i in range(one_kyoku_length) if "actor" in actions[i]]
                one_game_experience.extend(reward_dicided_experience)

            # d = [a for a in one_game_experience if a.action["type"]=="dahai"]
            # print("one game dahai len",len(d))
            del env
            return one_game_experience
        except KeyboardInterrupt:
            raise
        except Exception as e:
            lgs.logger_main.warn(f"fail to analyze {args}")
            import traceback
            print(traceback.format_exc())
            return []


if __name__ == "__main__":
    train_dir = "/data/mjson/train"
    test_dir = "/data/mjson/test"
    log_dir ="./output/logs"
    session_name = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    model_config = ModelConfig(
            resnet_repeat=50,
            mid_channels=256,
            learning_rate=0.0001,
            batch_size=256,
        )
    model_config.save(Path(log_dir)/session_name/"config.yaml")
    
    env = SampleCustomObserver(board=ArchiveBoard(), reward_calclator_cls=KyokuScoreReward)
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