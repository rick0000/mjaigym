import os
import re
from abc import ABCMeta, abstractmethod
import typing
import copy
import random
import datetime
import uuid
import json

from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
from gym import spaces
import numpy as np
from collections import deque, namedtuple
from tqdm import tqdm

import torch
import multiprocessing

if torch.cuda.is_available():
    from torch import multiprocessing
    from torch.multiprocessing import Pool, Process, set_start_method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
else:
    from multiprocessing import Pool, Process, set_start_method

from mjaigym.board import Board, BoardState
from mjaigym.board.function.pai import Pai
from mjaigym.board.mj_move import MjMove
from ml.model import Model
from mjaigym.board.function.efficient_dfs import Dfs

@dataclass
class SceneObservation:
    """ represents one player's feature observation for possible_actions.
    if there is no possible actions, fields filled None or [].

    chi, pon, kan use List[np.array] because they may have multiple furo candidates and
    each observation has each candidates infomation.
        ex) last action is dahai 3s, generate 4 observations
        [1s,2s], [2s,3s], [4s,5s], [4s,5sr].


    ある局面に対してあるプレーヤーが行動可能なアクションを特徴量化したものを保持する。
    行動可能なアクションがない場合はNoneか空のリストが入る。

    チー、ポン、カンについては同じ局面に対して複数の行動候補を取りうるため
    候補アクションと特徴量のタプルのリストが入る。
    """
    dahai_observation: np.array
    reach_observation: np.array
    chi_observations: typing.List[typing.Tuple[typing.Dict, np.array]]
    pon_observations: typing.List[typing.Tuple[typing.Dict, np.array]]
    kan_observations: typing.List[typing.Tuple[typing.Dict, np.array]]
    
    @classmethod
    def create_empty(cls):
        """ default value is as follows.
            dahai, reach: None,
            chi, pon, kan: []
        """
        return SceneObservation(
            dahai_observation=None,
            reach_observation=None,
            chi_observations=[],
            pon_observations=[],
            kan_observations=[],
        )


@dataclass
class Experience:
    """
    (state, action, rward) set.
    state: all players SceneObservation.
    action: executed action
    reward: this step reward


    """
    state:typing.Dict[int, SceneObservation] # key is 0~3, value is observation for player0~3
    action:typing.Dict
    reward:float
    board_state:BoardState


""" Observers
"""
class MjObserver(metaclass=ABCMeta):
    """ convert board state to feature.

    麻雀ロジックを担当するBoardクラスをopenaigymのインターフェースに一致させるためのクラス。
    """

    def __init__(self, board, reward_calclator_cls, oracle_rate:float=0.0):
        self._board = board
        self.reward_calclator = reward_calclator_cls()
        self.oracle_rate = oracle_rate
        self._dfs = Dfs()

    @abstractmethod
    def get_tsumo_observe_channels_num(self):
        """ return dahai and reach observation channel num.
        打牌、立直の特徴量チャネル数定義を返す
        """
        raise NotImplementedError()
        
    @abstractmethod
    def get_otherdahai_observe_channels_num(self):
        """ return chi, pon and kan observation channel num.
        チー、ポン、カンの特徴量チャネル数定義を返す
        """
        raise NotImplementedError()

    @abstractmethod
    def trainsform_dahai(self, state:BoardState, id:int, oracle_enable_flag:bool):
        """打牌モデル用特徴量を生成する
        """
        raise NotImplementedError("")

    @abstractmethod
    def trainsform_reach(self, state:BoardState, id:int, oracle_enable_flag:bool):
        """立直モデル用特徴量を生成する
        """
        raise NotImplementedError("")

    @abstractmethod
    def trainsform_chi(self, state:BoardState, id:int, candidate_action:typing.Dict, oracle_enable_flag:bool):
        """チーモデル用特徴量を生成する
        """
        raise NotImplementedError("")

    @abstractmethod
    def trainsform_pon(self, state:BoardState, id:int, candidate_action:typing.Dict, oracle_enable_flag:bool):
        """ポンモデル用特徴量を生成する
        """
        raise NotImplementedError("")

    @abstractmethod
    def trainsform_kan(self, state:BoardState, id:int, candidate_action:typing.Dict, oracle_enable_flag:bool):
        """カンモデル用特徴量を生成する
        """
        raise NotImplementedError("")

    @property
    def action_space(self):
        """ action spaces for each InnerAgent.
        """
        return spaces.Dict({
            "dahai_agent":spaces.Discrete(34),
            "reach_agent":spaces.Discrete(2),
            "chi_agent":spaces.Discrete(2),
            "pon_agent":spaces.Discrete(2),
            "kan_agent":spaces.Discrete(2),
        })
    
    @property
    def observation_space(self):
        """ observation spaces for each InnerAgent.
        """
        return spaces.Dict({
            "dahai_agent":spaces.Box(low=0, high=1, shape=(self.get_tsumo_observe_channels_num(), 34, 1), dtype=np.int),
            "reach_agent":spaces.Box(low=0, high=1, shape=(self.get_tsumo_observe_channels_num(), 34, 1), dtype=np.int),
            "chi_agent":spaces.Box(low=0, high=1, shape=(self.get_otherdahai_observe_channels_num(), 34, 1), dtype=np.int),
            "pon_agent":spaces.Box(low=0, high=1, shape=(self.get_otherdahai_observe_channels_num(), 34, 1), dtype=np.int),
            "kan_agent":spaces.Box(low=0, high=1, shape=(self.get_otherdahai_observe_channels_num(), 34, 1), dtype=np.int),
        })

    def reset(self):
        """ initialize environment.

        Returns:
            (observation, reward, done, None)
        """
        self._board.reset()
        return self._make_response()

    def step(self, actions):
        """returns states for each agent, reward, done, None.
        NOTE:input variable actions class depends on self.board class.
        self.boardのクラスによってactionsのクラスが変わる。
        
        
        Args:
            actions (Dict[int, Dict] or Dict): board step action.
                for Board, need 4 player actions.
                for ArchiveBoard, need only 1 action.
        Returns:
            (observation, reward, done, None)

        """
        self._board.step(actions)
        return self._make_response()
        
    def _make_response(self):
        board_state = self._board.get_state()
        self.reward_calclator.step(board_state.previous_action)
        reward = self.reward_calclator.calc()
        done = self._board.is_end
        possible_actions = board_state.possible_actions
        info = {
            "possible_actions":board_state.possible_actions,
            "board_state":board_state,
            }

        oracle_enable_flag = random.random() < self.oracle_rate
        return self._transform(board_state, possible_actions, oracle_enable_flag), reward, done, info


    def render(self, mode="console"):
        """ render board state to console.
        """
        print(self._board.get_state())

    def _transform(self, state:BoardState, possible_actions:typing.Dict[int,typing.Dict], oracle_enable_flag:bool=False):
        """ convert board state to feature.
        
        BoardStateクラスと候補手をもとに特徴量を生成する。

        Args:
            state (BoardState):
            possible_actions (typing.List[int, typing.Dict]): [description]

        Returns:
            SceneObservation: converted features for InnerAgents.
        """
        result = {}
        for player_id in range(4):
            # initialize
            player_observation = SceneObservation.create_empty()
            
            player_possible_actions = possible_actions[player_id]
            
            dahais = [a for a in player_possible_actions if a["type"]==MjMove.dahai.value]
            reachs = [a for a in player_possible_actions if a["type"]==MjMove.reach.value]
            chis = [a for a in player_possible_actions if a["type"]==MjMove.chi.value]
            pons = [a for a in player_possible_actions if a["type"]==MjMove.pon.value]
            kans = [a for a in player_possible_actions if a["type"] in [MjMove.ankan.value, MjMove.kakan.value, MjMove.daiminkan.value]]

            # transform with candidate actions
            if len(dahais) > 0:
                player_observation.dahai_observation = self.trainsform_dahai(state, player_id, oracle_enable_flag)
            elif len(reachs) > 0:
                player_observation.reach_observation = self.trainsform_reach(state, player_id, oracle_enable_flag)
            elif len(chis) > 0:
                for chi in chis:
                    converted = self.trainsform_chi(state, player_id, chi, oracle_enable_flag)
                    player_observation.chi_observations.append((chi, converted))
            elif len(pons) > 0:
                for pon in pons:
                    converted = self.trainsform_pon(state, player_id, pon, oracle_enable_flag)
                    player_observation.pon_observations.append((pon, converted))
            elif len(kans) > 0:
                for kan in kans:
                    converted = self.trainsform_kan(state, player_id, kan, oracle_enable_flag)
                    player_observation.kan_observations.append((kan, converted))
            
            result[player_id] = player_observation

        return result

    def dump(self, dir_path):    
        os.makedirs(dir_path, exist_ok=True)
        fname = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + "_" + str(uuid.uuid4()) + '.mjson'
        fpath = os.path.join(dir_path, fname)
        history = self._board.dealer_history
        with open(fpath,'wt') as f:
            formatted_lines = [json.dumps(l)+os.linesep for l in self._board.dealer_history]
            f.writelines(formatted_lines)
        return fname

class TensorBoardLogger():
    """Tensorboardに実行結果を書き込む
    """
    def __init__(self, log_dir="", session_name=""):
        self.log_dir = log_dir
        if not log_dir:
            self.log_dir = os.path.join(os.path.dirname(__file__), "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        
        if session_name:
            self.log_dir = os.path.join(self.log_dir, session_name)
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir, exist_ok=True)
        

    def write(self, tag, value, i):
        with SummaryWriter(log_dir=self.log_dir) as writer:
            writer.add_scalar(tag, value, i)




    