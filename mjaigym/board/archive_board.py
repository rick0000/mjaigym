import copy
import os
import pprint
from typing import List

import numpy as np
from mjaigym.board.function.hora import Hora
from mjaigym.board.function.mj_move import MjMove
from mjaigym.board.function.pai import Pai
from mjaigym.board.function.player import Player
from mjaigym.board.function.yama import Yama

from .board import Board


class ArchiveBoard(Board):
    def __init__(self, scene_mjsons=None, **args):
        super().__init__()
        if scene_mjsons is None:
            scene_mjsons = []

        self._current_seed = self._raw_seed
        self.rest_nums = np.array([4] * 34)

        self.dealer_history = []
        self.renponse_history = []
        self.chicha = None
        self.last = False

        self.valid_move_history = []

        for mjson in scene_mjsons:
            self._paifu_step(mjson)
            self.dealer_history.append(mjson)
            self.valid_move_history.append(self.possible_actions)
            self.restnum_update(mjson)

    def step(self, action):
        self._paifu_step(action)
        self.dealer_history.append(action)
        self.valid_move_history.append(self.possible_actions)
        self.restnum_update(action)

    def _paifu_step(self, action):
        if "actor" in action:
            self.actor = action["actor"]
        action_type = action["type"]

        if "scores" in action:
            self.scores = action["scores"]

        if action_type == MjMove.start_game.value:
            if "seed" in action:
                self._current_seed = action["seed"]
        if action_type == MjMove.start_kyoku.value:
            self.yama = Yama(self._current_seed)
            self.bakaze = action["bakaze"]
            self.kyoku = action["kyoku"]
            self.honba = action["honba"]
            if "kyotaku" in action:
                self.kyotaku = action["kyotaku"]
            self.oya = action["oya"]

            if self.chicha == None:
                self.chicha = self.oya

            for tehai in action["tehais"]:
                for p in tehai:
                    self.yama.paifu_tsumo(p)

            self.dora_markers = [action["dora_marker"]]
            self.yama.paifu_open_doramarker(action["dora_marker"])

            self.first_turn = True
            self.next_reach_pending = False
            self.reach_pending = False

        elif action_type == MjMove.tsumo.value:
            self.yama.paifu_tsumo(action["pai"])

            if self.yama.get_tsumoed_num() > 4:
                self.first_turn = False

        elif action_type in [
            MjMove.chi.value,
            MjMove.pon.value,
            MjMove.daiminkan.value,
            MjMove.kakan.value,
            MjMove.ankan.value,
        ]:
            self.first_turn = False

        elif action_type == MjMove.dora.value:
            self.yama.paifu_open_doramarker(action["dora_marker"])
            self.dora_markers.append(action["dora_marker"])

        elif action_type == MjMove.reach_accepted.value:
            self.kyotaku += 1
        elif action_type == MjMove.hora.value:
            self.kyotaku = 0

        for i in range(4):
            self.players[i].update_state(action)
