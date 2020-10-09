"""Agents wrapper and inner agents"""

from collections import deque
from pathlib import Path
import copy
import random
import typing
import multiprocessing

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
import numpy as np
from tqdm import tqdm

from mjaigym.board.function.pai import Pai
from mjaigym.board.mj_move import MjMove
import mjaigym.loggers as lgs
from ml.framework import Experience
from ml.model import Model, Head2SlModel, Head34SlModel
from mjaigym.client import MaxUkeireClient


class MjAgent():
    """ 打牌、立直、チー、ポン、カン用の内部エージェントのラッパー。
    モデルによる判断が必要な場合のみ内部エージェントで行動確率を計算する。
    ラッパー内だけで行動が決定できる場合は内部エージェントを通さず行動を返す。
    """

    def __init__(
            self,
            dahai_agent,
            reach_agent,
            chi_agent,
            pon_agent,
            kan_agent,
            binary_action_prob_thresh=0.5,
        ):

        self.dahai_agent = dahai_agent
        self.reach_agent = reach_agent
        self.chi_agent = chi_agent
        self.pon_agent = pon_agent
        self.kan_agent = kan_agent
        self.binary_action_prob_thresh = binary_action_prob_thresh

    def save(self, save_dir, i):
        i_str = str(i).zfill(6)
        self.dahai_agent.save(save_dir / i_str / "dahai.pth")
        self.reach_agent.save(save_dir / i_str / "reach.pth")
        self.chi_agent.save(save_dir / i_str / "chi.pth")
        self.pon_agent.save(save_dir / i_str / "pon.pth")
        self.kan_agent.save(save_dir / i_str / "kan.pth")

    def play(self, env, episode_count, render):
        for i in range(episode_count):
            state, reward, done, info = env.reset()
            done = False
            while not done:
                if render:
                    env.render()
                actions = self.think_all_player(state, info["possible_actions"],  info['board_state'])
                n_state, reward, done, info = env.step(actions)
                state = n_state
        return env

    def play_multiprocess(self, env, episode_count, render):
        params = [(copy.deepcopy(env), render) for i in range(episode_count)]
        results = deque()
        with multiprocessing.get_context('spawn').Pool(processes=multiprocessing.cpu_count()) as pool:
            with tqdm(total=episode_count) as t:
                for one_mjson_dataset in pool.imap_unordered(self._play_one, params):
                    results.append(one_mjson_dataset)
                    t.update(1)
                pool.close()
                pool.terminate()
        return list(results)
    
    def _play_one(self, args):
        env, render = args
        state, reward, done, info = env.reset()
        done = False
        while not done:
            if render:
                env.render()
            actions = self.think_all_player(state, info["possible_actions"], info['board_state'])
            n_state, reward, done, info = env.step(actions)
            state = n_state

    def think_one_player(self, player_observation, player_possible_actions, player_id, board_state):
        if len(player_possible_actions) == 1:
            return player_possible_actions[0]

        # if can hora, always hora.
        horas = [action for action in player_possible_actions if action["type"] == MjMove.hora.value]
        if len(horas) > 0:
            return horas[0]
            
        # start agent think
        on_tsumo = False
        if player_observation.dahai_observation is not None:
            on_tsumo = True

        if on_tsumo:
            if player_observation.reach_observation is not None:
                # predict reach
                pred = self.reach_agent.policy(player_observation.reach_observation)
                if pred == 1:
                    reach_action = [action for action in player_possible_actions if action["type"]==MjMove.reach.value][0]
                    return reach_action
            else:
                # predict dahai
                pred = self.dahai_agent.policy(player_observation.dahai_observation)
                dahai_candidates = [action for action in player_possible_actions if action["type"]==MjMove.dahai.value]
                
                highest_dahai_p = 0.0
                highest_dahai = None
                for candidate in dahai_candidates:
                    prob = pred[Pai.str_to_id(candidate["pai"])]
                    if prob >= highest_dahai_p:
                        highest_dahai_p = prob
                        highest_dahai = candidate
                
                return highest_dahai
        else:
            highest_prob = 0.0
            highest_prob_action = None

            # predict chi
            for candidate_action, candidate_observe in player_observation.chi_observations:
                action_prob = self.chi_agent.policy(candidate_observe)[1] # [0] is not action prob, [1] is action prob 
                if highest_prob < action_prob:
                    highest_prob = action_prob
                    highest_prob_action = candidate_action

            # predict pon
            for candidate_action, candidate_observe in player_observation.pon_observations:
                action_prob = self.pon_agent.policy(candidate_observe)[1]
                if highest_prob < action_prob:
                    highest_prob = action_prob
                    highest_prob_action = candidate_action
            
            # predict kan
            for candidate_action, candidate_observe in player_observation.kan_observations:
                action_prob = self.kan_agent.policy(candidate_observe)[1]
                if highest_prob < action_prob:
                    highest_prob = action_prob
                    highest_prob_action = candidate_action

            # check probability threshold
            if highest_prob >= self.binary_action_prob_thresh:
                return highest_prob_action
            else:
                return {"type":"none"}

        assert Exception("not intended path")


    def think_all_player(self, state, possible_actions, board_state):
        result = {}
        for player_id in range(4):
            player_possible_actions = possible_actions[player_id]
            player_observation = state[player_id]
            result[player_id] = self.think_one_player(player_observation, player_possible_actions, player_id, board_state)
        
        return result
    

    def evaluate(self, experiences:typing.List[Experience]):
        evaluate_result = {}
        stats = self.dahai_agent.evaluate(experiences)
        if stats is not None:
            loss, acc = stats
            evaluate_result["dahai_loss_test"] = loss
            evaluate_result["dahai_acc_test"] = acc

        stats = self.reach_agent.evaluate(experiences)
        if stats is not None:
            loss, acc = stats
            evaluate_result["reach_loss_test"] = loss
            evaluate_result["reach_acc_test"] = acc

        stats = self.chi_agent.evaluate(experiences)
        if stats is not None:
            loss, acc = stats
            evaluate_result["chi_loss_test"] = loss
            evaluate_result["chi_acc_test"] = acc

        stats = self.pon_agent.evaluate(experiences)
        if stats is not None:
            loss, acc = stats
            evaluate_result["pon_loss_test"] = loss
            evaluate_result["pon_acc_test"] = acc

        stats = self.kan_agent.evaluate(experiences)
        if stats is not None:
            loss, acc = stats
            evaluate_result["kan_loss_test"] = loss
            evaluate_result["kan_acc_test"] = acc

        return evaluate_result

    def update(self, experiences:typing.List[Experience]):
        # make dataset for each agents
        update_result = {}
        stats = self.dahai_agent.update(experiences)
        if stats is not None:
            update_result.update(stats)

        stats = self.reach_agent.update(experiences)
        if stats is not None:
            update_result.update(stats)

        stats = self.chi_agent.update(experiences)
        if stats is not None:
            update_result.update(stats)

        stats = self.pon_agent.update(experiences)
        if stats is not None:
            update_result.update(stats)

        stats = self.kan_agent.update(experiences)
        if stats is not None:
            update_result.update(stats)

        return update_result

class InnerAgent():
    """内部エージェント
    Nクラス分類問題を解くための機能を提供する
    """
    def __init__(self, actions, model_class:Model, model_config, epsilon=0.0):
        self.epsilon = epsilon
        self.actions = actions
        self.actions_length = actions.n
        self.model_class = model_class
        self.model = None
        self.model_config = model_config
        self.initialized = False
        
    def policy(self, observation):
        if not self.initialized:
            self.initialize(observation)

        if np.random.rand() < self.epsilon:
            return np.random.rand(self.actions_length)
        
        return self.model.policy([observation])[0]

    def estimate(self, observation):
        if not self.initialized:
            self.initialize(observation)
        return self.model.estimate([observation])[0]


    def initialize(self, observation):
        self.model = self.model_class(
            in_channels=observation.shape[0],
            mid_channels=self.model_config.mid_channels,
            blocks_num=self.model_config.resnet_repeat,
            learning_rate=self.model_config.learning_rate,
            batch_size=self.model_config.batch_size,
        )
        self.initialized = True

    def update(self, experiences:typing.List[Experience]):
        raise NotImplementedError()
    def evaluate(self, experiences:typing.List[Experience]):
        raise NotImplementedError()


    def save(self, save_path):
        if self.model is not None:
            self.model.save(save_path)
    
    def load(self, load_path):
        if self.model is not None:
            self.initialize(self.observation)

        self.model.load(load_path)




class DahaiTrainableAgent(InnerAgent):
    """ニューラルネットを使った打牌モデル用の内部モデルエージェント"""
    def __init__(self, actions, model_class, model_config, epsilon=0.0):
        super(DahaiTrainableAgent, self).__init__(actions, model_class, model_config, epsilon)
        self.update_buffer = None
        self.model_config = model_config
        
    def update(self, experiences:typing.List[Experience]):
        """
        打牌用特徴量を抽出する
        """
        state_action_rewards = []
        for experience in experiences:
            for i in range(4):
                player_state = experience.state[i]
                if player_state.dahai_observation is not None\
                    and not experience.board_state.reach[i]\
                    and experience.action["type"] == MjMove.dahai.value:
                    
                    label = Pai.str_to_id(experience.action["pai"])
                    state_action_rewards.append(tuple((
                        player_state.dahai_observation,
                        label,
                        experience.reward,
                    )))
        
        # print("dahai update s_a_r",len(state_action_rewards))
        
        if not self.initialized:
            self.initialize(state_action_rewards[0][0])


        return self.model.update(state_action_rewards)

    def evaluate(self, experiences:typing.List[Experience]):
        lgs.logger_main.info("start dahai model evaluate")
        state_action_rewards = []
        for experience in experiences:
            for i in range(4):
                player_state = experience.state[i]
                
                
                if player_state.dahai_observation is not None\
                    and not experience.board_state.reach[i]\
                    and experience.action["type"] == MjMove.dahai.value:
                    
                    label = Pai.str_to_id(experience.action["pai"])
                    state_action_rewards.append(tuple((
                        player_state.dahai_observation,
                        label,
                        experience.reward,
                    )))
        batch_num = len(state_action_rewards) // self.model_config.batch_size
        sampled_state_action_rewards = random.choices(state_action_rewards, k=batch_num*self.model_config.batch_size)
        
        loss, acc = self.model.evaluate(sampled_state_action_rewards)
        lgs.logger_main.info(f"end dahai model evaluate")
        
        sampled_state_action_rewards.clear()
        state_action_rewards.clear()
        return loss, acc


class FixPolicyAgent(InnerAgent):
    """ルールベースの内部エージェント。
    出力確率が固定。
    """
    def __init__(self, fix_policy_probs):
        self.fix_policy_probs = fix_policy_probs
    def policy(self, observation):
        return self.fix_policy_probs
    def update(self, experiences):
        pass
    def evaluate(self, experiences):
        pass
    def save(self, save_path):
        pass





class MaxUkeireMjAgent(MjAgent):
    """ 外側のエージェント
    内部モデルは持たずルールベースで判断する。
    打牌は受け入れ最大、副露は確率でルールベースで行う。
    """
    def __init__(self, id, name):
        self.client = MaxUkeireClient(id=id, name=name)

    def think_one_player(self, player_observation, player_possible_actions, player_id, board_state):
        return self.client.think(board_state)

    def update(self):
        pass



class DahaiActorCriticAgent(InnerAgent):
    """ActorCritic法を用いた打牌モデル用の内部モデルエージェント"""
    def __init__(self, actions, model_class, model_config, epsilon=0.0):
        super(DahaiActorCriticAgent, self).__init__(actions, model_class, model_config, epsilon)
        self.update_buffer = None
        self.model_config = model_config
        
    def update(self, experiences:typing.List[Experience]):
        """
        打牌用特徴量を抽出する
        """
        state_action_rewards = []
        for experience in experiences:
            for i in range(4):
                player_state = experience.state[i]
                if player_state.dahai_observation is not None\
                    and not experience.board_state.reach[i]\
                    and experience.action["type"] == MjMove.dahai.value:
                    
                    label = Pai.str_to_id(experience.action["pai"])
                    state_action_rewards.append(tuple((
                        player_state.dahai_observation,
                        label,
                        experience.reward,
                    )))
        
        # print("dahai update s_a_r",len(state_action_rewards))
        
        if not self.initialized:
            self.initialize(state_action_rewards[0][0])

        return self.model.update(state_action_rewards)

    def evaluate(self, experiences:typing.List[Experience]):
        lgs.logger_main.info("start dahai model evaluate")
        state_action_rewards = []
        for experience in experiences:
            for i in range(4):
                player_state = experience.state[i]
                
                if player_state.dahai_observation is not None\
                    and not experience.board_state.reach[i]\
                    and experience.action["type"] == MjMove.dahai.value:
                    
                    label = Pai.str_to_id(experience.action["pai"])
                    state_action_rewards.append(tuple((
                        player_state.dahai_observation,
                        label,
                        experience.reward,
                    )))
        batch_num = len(state_action_rewards) // self.model_config.batch_size
        sampled_state_action_rewards = random.choices(state_action_rewards, k=batch_num*self.model_config.batch_size)
        
        loss, acc = self.model.evaluate(sampled_state_action_rewards)
        lgs.logger_main.info(f"end dahai model evaluate")
        
        sampled_state_action_rewards.clear()
        state_action_rewards.clear()
        return loss, acc
