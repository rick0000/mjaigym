import copy
import os
import pprint
from typing import List

import numpy as np
from mjaigym.board.board import Board
from mjaigym.board.function.client_yama import ClientYama
from mjaigym.board.function.hora import Hora
from mjaigym.board.function.pai import Pai
from mjaigym.board.function.player import Player
from mjaigym.board.mj_move import MjMove


class ClientBoard(Board):
    def __init__(self):
        super().__init__()
        self._current_seed = self._raw_seed
        self.rest_nums = np.array([4] * 34)

        self.dealer_history = []
        self.renponse_history = []
        self.chicha = None
        self.last = False
        self.id = None

        self.valid_move_history = []

    def step(self, action):
        if action["type"] in [
            MjMove.join.value,
            MjMove.error.value,
            MjMove.hello.value,
        ]:
            return

        self._paifu_step(action)
        self.dealer_history.append(action)

        base_space = self._get_base_space()
        if "possible_actions" in action:
            base_space[self.id] = action["possible_actions"]

        self.valid_move_history.append(base_space)

    def _get_base_space(self):
        return {
            0: [{"type": MjMove.none.value}],
            1: [{"type": MjMove.none.value}],
            2: [{"type": MjMove.none.value}],
            3: [{"type": MjMove.none.value}],
        }

    @property
    def possible_actions(self):
        if len(self.valid_move_history) > 0:
            return self.valid_move_history[-1]
        else:
            return self._get_base_space()

    def _paifu_step(self, action):
        if "actor" in action:
            self.actor = action["actor"]
        action_type = action["type"]

        if "scores" in action:
            self.scores = action["scores"]

        if action_type == MjMove.start_game.value:
            if "seed" in action:
                self._current_seed = action["seed"]
            if "id" in action:
                self.id = action["id"]

        if action_type == MjMove.start_kyoku.value:
            # self.yama = Yama(self._current_seed)
            self.yama = ClientYama()
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
                    self.yama.tsumo(p)

            self.dora_markers = [action["dora_marker"]]
            self.yama.open_doramarker(action["dora_marker"])

            self.first_turn = True
            self.next_reach_pending = False
            self.reach_pending = False

        elif action_type == MjMove.tsumo.value:
            self.yama.tsumo(action["pai"])

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
            self.yama.open_doramarker(action["dora_marker"])
            self.dora_markers.append(action["dora_marker"])

        elif action_type == MjMove.reach_accepted.value:
            self.kyotaku += 1
        elif action_type == MjMove.hora.value:
            self.kyotaku = 0

        for i in range(4):
            if "id" in action:
                action = copy.copy(action)
                del action["id"]
            self.players[i].update_state(action)
