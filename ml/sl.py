""" module for paifu supervised learning.
"""
from collections import deque, namedtuple
from pathlib import Path
import copy
import math
import gc
import random
import typing
import pickle
import datetime
import itertools

import numpy as np
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, Process, set_start_method


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
from ml.model import  Head2SlModel, Head34SlModel
from ml.agent import InnerAgent, MjAgent, DahaiTrainableAgent, FixPolicyAgent


class SlTrainer():
    def __init__(
            self,
            train_dir,
            test_dir,
            log_dir,
            session_name,
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
        
        self.reward_discount_rate = reward_discount_rate
        self.experiences = deque()
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.evaluate_per_update = evaluate_per_update
        self.session_name = session_name
        self.session_dir = Path(self.log_dir) /self.session_name
        
        self.tfboard_logger = TensorBoardLogger(log_dir=self.log_dir, session_name=self.session_name)
        

    def train_loop(self, agent, env):
        lgs.logger_main.info("start train loop")
        mjson_paths = Path(self.train_dir).glob("**/*.mjson")
        
        mjson_buffer = deque(maxlen=self.udpate_interbal)

        

        for i, mjson_path in enumerate(mjson_paths):
            mjson_buffer.append((i, mjson_path, copy.deepcopy(env)))
            if not (i > 0 and i % self.udpate_interbal == 0):
                continue
            
            # start analyze
            results = deque()
            if self.use_multiprocess:
                with multiprocessing.get_context('spawn').Pool(processes=multiprocessing.cpu_count()) as pool:
                    with tqdm(total=len(mjson_buffer)) as t:
                        for one_mjson_datasets in pool.imap_unordered(self._analyze_one_game, mjson_buffer):
                            results.append(one_mjson_datasets)
                            t.update(1)
                    pool.close()
                    pool.terminate()
            else:
                for mjson in tqdm(mjson_buffer):
                    one_mjson_datasets = self._analyze_one_game(mjson)
                    
                    results.append(one_mjson_datasets)

            results = list(itertools.chain.from_iterable(results))

            # update agent
            update_result = agent.update(results)
            results.clear()
            for key, value in update_result.items():
                self.tfboard_logger.write(key, value, i)
            

            # test evaluate
            if i % (self.udpate_interbal * self.evaluate_per_update) == 0:
                test_experience = self._get_test_experience(self.test_dir, env)
                evaluate_result = agent.evaluate(test_experience)
                for key, value in evaluate_result.items():
                    self.tfboard_logger.write(key, value, i)
                test_experience.clear()

                # save agent
                agent.save(self.session_dir, i)

            # clear buffer
            gc.collect()

    

    def _analyze_one_game(self, args):
        try:
            mjson_id, mjson_path, env = args
            
            one_game_experience = deque()
             
            mjson = Mjson.load(mjson_path)
            for kyoku in mjson.game.kyokus:

                state, reward, done, info = env.reset()
                
                states = deque()
                actions = deque()
                rewards = deque()
                board_states = deque()
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
                reward_dicided_experience = [Experience(states[i], actions[i], discount_rewards[i], board_states[i])  for i in range(one_kyoku_length)]
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
            return deque()

    def _get_test_experience(self, mjson_dir, env, max_game_num=500):
        mjson_paths = Path(mjson_dir).glob("**/*.mjson")
        lgs.logger_main.info("start generate test dataset")
        test_path = Path(mjson_dir) / "test_dataset.pkl"
        test_path.parent.mkdir(parents=True, exist_ok=True)
        if test_path.is_file():
            with open(test_path, "rb") as f:
                return pickle.load(f)
        
        use_mjson = []
        for i, mjson_path in enumerate(mjson_paths):
            if i >= max_game_num:
                break
            use_mjson.append((i, mjson_path, copy.deepcopy(env)))
        
        results = deque()
        with multiprocessing.get_context('spawn').Pool(processes=multiprocessing.cpu_count()) as pool:
            with tqdm(total=len(use_mjson)) as t:
                for one_mjson_datasets in pool.imap_unordered(self._analyze_one_game, use_mjson):
                    results.extend(one_mjson_datasets)
                    t.update(1)
                pool.close()
                pool.terminate()
        
        sample_size = max_game_num*10
        sample_size = min(sample_size, len(results))
        use_results = random.choices(results, k=sample_size)

        with open(test_path, "wb") as f:
            pickle.dump(use_results, f)

        return use_results


if __name__ == "__main__":
    train_dir = "/data/mjson/train"
    test_dir = "/data/mjson/test"
    log_dir ="./output/logs"
    session_name = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    model_config = ModelConfig(
            resnet_repeat=40,
            mid_channels=128,
            learning_rate=0.0005,
            batch_size=128,
        )
    model_config.save(Path(log_dir)/session_name/"config.yaml")
    sl_trainer = SlTrainer(
            train_dir,
            test_dir,
            log_dir=log_dir,
            session_name=session_name,
            use_multiprocess=True,
            udpate_interbal=64,
            batch_size=model_config.batch_size,
            evaluate_per_update=10
        )
        
    env = SampleCustomObserver(board=ArchiveBoard(), reward_calclator_cls=KyokuScoreReward)
    actions = env.action_space
    
    dahai_agent = DahaiTrainableAgent(actions["dahai_agent"], Head34SlModel, model_config)
    reach_agent = FixPolicyAgent(np.array([0.0, 1.0])) # always do reach
    chi_agent = FixPolicyAgent(np.array([1.0, 0.0])) # never chi
    pon_agent = FixPolicyAgent(np.array([1.0, 0.0])) # never pon
    kan_agent = FixPolicyAgent(np.array([1.0, 0.0])) # never kan
    mj_agent = MjAgent(dahai_agent, reach_agent, chi_agent, pon_agent, kan_agent)
    
    sl_trainer.train_loop(mj_agent, env)