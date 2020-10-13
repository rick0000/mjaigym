import time
import copy
import math
import random
from collections import deque
from pathlib import Path
import multiprocessing
from multiprocessing import Queue
from multiprocessing import Pool, Process, set_start_method, Queue

import numpy as np
from dataclasses import dataclass

from mjaigym.board.function.pai import Pai
from mjaigym.board.mj_move import MjMove
from mjaigym.mjson import Mjson
from ml.framework import Experience, MjObserver
import mjaigym.loggers as lgs


class StateActionRewardGenerator():
    def __init__(
            self, 
            mjson_dir:str, 
            experience_queue:Queue,
            sampling_rate:float=0.05,
            reward_discount_rate:float=0.99,
            ):
        self.mjson_dir = mjson_dir
        self.experience_queue = experience_queue
        self.sampling_rate = sampling_rate
        self.reward_discount_rate = reward_discount_rate

        self.mjson_path_queue = Queue(maxsize=128)
        
        self.processes = []


    def start(self, env, process_num):
        # run file path glob process
        glob_process = Process(
            target=self._generate_mjson_path, 
            daemon=True)
        glob_process.start()
        self.processes.append(glob_process)
        
        # run generate process        
        for i in range(process_num):
            p = Process(
                target=_generate_state_action_reward, 
                args=(
                    i,
                    self.mjson_path_queue,
                    self.experience_queue,
                    env,
                    self.reward_discount_rate,
                    self.sampling_rate
                    ),
                daemon=True)
            
            p.start()
            self.processes.append(p)

    def terminate(self):
        for p in self.processes:
            p.terminate()

    def _generate_mjson_path(self):
        mjson_paths = Path(self.mjson_dir).glob("**/*.mjson")
        for mjson_path in mjson_paths:
            # this function blocks when full
            self.mjson_path_queue.put(mjson_path)
            
            time.sleep(0.1)
            
def _generate_state_action_reward(
        process_number:int,
        mjson_path_queue:Queue,
        experience_queue:Queue,
        env:MjObserver,
        reward_discount_rate:float,
        sampling_ratio:float):
       
    while True:
        # print(f"generate data@{process_number}")
        mjson_path = mjson_path_queue.get()
        experiences = _analyze_one_game((mjson_path, copy.deepcopy(env), reward_discount_rate))
        
        sample_num = int(sampling_ratio * len(experiences))

        if sample_num > 0:
            # sampling
            sampled_experiences = random.sample(experiences, sample_num)
            # print(f"process{process_number}, sampled:{len(experiences)}->{len(sampled_experiences)}")
            
            # delay evaluate here
            [s.calclate() for s in sampled_experiences]
            experience_queue.put(sampled_experiences)

def _analyze_one_game(args):
    try:
        mjson_path, env, reward_discount_rate = args
        
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
            reversed_discount_rewards = [last_reward * math.pow(reward_discount_rate, i) for i in range(one_kyoku_length)]
            discount_rewards = list(reversed(reversed_discount_rewards))
            
            # reward 1000.0 diff as loss 1.0
            discount_rewards = [d/1000.0 for d in discount_rewards]

            reward_dicided_experience = [Experience(states[i], actions[i], discount_rewards[i][actions[i]["actor"]], board_states[i])  for i in range(one_kyoku_length) if "actor" in actions[i]]
            one_game_experience.extend(reward_dicided_experience)

        del env
        return one_game_experience
    except KeyboardInterrupt:
        raise
    except Exception as e:
        lgs.logger_main.warn(f"fail to analyze {args}")
        import traceback
        print(traceback.format_exc())
        return []



