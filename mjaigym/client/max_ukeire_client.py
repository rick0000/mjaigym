import copy
import datetime
import itertools
import pprint
import random
from typing import Dict, List, Tuple

import numpy as np
from mjaigym.board import BoardState
from mjaigym.board.function.mj_move import MjMove
from mjaigym.board.function.pai import Pai
from mjaigym.board.function.player import Player
from mjaigym.board.function.rs_shanten_analysis import RsShantenAnalysis
from mjaigym.client import Client


class MaxUkeireClient(Client):
    ShantenAnalyser = RsShantenAnalysis()

    def __init__(self, id=None, name=None):
        if name is None:
            name = f"MaxUkeire{id}"
        super(MaxUkeireClient, self).__init__(id, name)

    def clone(self):
        return self

    def think(self, board_state: BoardState):
        message = board_state.previous_action

        if message["type"] == MjMove.start_game.value and "id" in message:
            self.id = message["id"]
        if len(board_state.possible_actions[self.id]) == 1:
            return board_state.possible_actions[self.id][-1]

        candidates = board_state.possible_actions[self.id]
        if (
            "actor" in board_state.previous_action
            and board_state.previous_action["actor"] == self.id
        ):
            return self.think_on_tsumo(board_state, candidates)
        else:
            return self.think_on_other_dahai(board_state, candidates)

    def think_on_tsumo(self, board_state: BoardState, candidates: List[Dict]):

        my_tehais = board_state.tehais[self.id]
        rest_in_view = board_state.restpai_in_view[self.id]

        # if can hora, always do hora
        hora_candidates = [c for c in candidates if c["type"] == MjMove.hora.value]
        if len(hora_candidates) == 1:
            return hora_candidates[0]

        # if can reach, always do reach
        reach_candiadtes = [c for c in candidates if c["type"] == MjMove.reach.value]
        if len(reach_candiadtes) == 1:
            return reach_candiadtes[0]

        # dahai think
        dahai_candiadtes = [c for c in candidates if c["type"] == MjMove.dahai.value]

        # calclate the number of shanten reducing tsumo with each dahai
        node = Node(my_tehais)
        valid_nums = node.calc_ukeire_num(rest_num=rest_in_view)

        max_valid_dahai_action = None
        max_valid_num = -1

        for candidate in dahai_candiadtes:
            valid_dahai_id = Pai.str_to_id(candidate["pai"])
            valid_num = valid_nums[valid_dahai_id]

            if max_valid_num < valid_num:
                max_valid_dahai_action = candidate
                max_valid_num = valid_num

        return max_valid_dahai_action

    def think_on_other_dahai(self, board_state: BoardState, candidates: List[Dict]):

        my_tehais = board_state.tehais[self.id]

        # if there is only one candidate, return that.
        if len(candidates) == 1:
            return candidates[-1]

        # if can hora, always do hora
        hora_candidates = [c for c in candidates if c["type"] == MjMove.hora.value]
        if len(hora_candidates) == 1:
            return hora_candidates[0]

        # if there is other player reach, don't furo.
        if any(board_state.reach):
            none_candidates = [c for c in candidates if c["type"] == MjMove.none.value]
            if len(none_candidates) == 0:
                raise Exception("not intended path, none candidate not found.")
            else:
                return none_candidates[0]

        # this path assumes there is no other player reach

        # shanten reduceable yakuhai pon is executed.
        furo_types = [MjMove.daiminkan.value, MjMove.pon.value, MjMove.chi.value]
        furo_candidates = [c for c in candidates if c["type"] in furo_types]

        for candidate in furo_candidates:
            if candidate["type"] == MjMove.pon.value:
                candidate_pai = Pai.from_str(candidate["pai"])
                is_dragon = candidate_pai.is_sangenpai()
                is_jikaze = candidate_pai.str == board_state.jikaze
                is_bakaze = candidate_pai.str == board_state.bakaze

                # if is_yakuhai, 20% do pon
                if is_dragon or is_jikaze or is_bakaze:
                    before_node = Node(board_state.tehais[self.id])
                    executed_node = Node(board_state.tehais[self.id], candidate_pai)
                    if (
                        before_node.shanten > executed_node.shanten
                        and random.random() < 0.2
                    ):
                        return candidate

            # if already furo, and shanten reduceable, 60% furo.
            if len(board_state.furos[self.id]) > 0:
                candidate_pai = Pai.from_str(candidate["pai"])
                before_node = Node(board_state.tehais[self.id])
                executed_node = Node(board_state.tehais[self.id], candidate_pai)
                if (
                    before_node.shanten > executed_node.shanten
                    and random.random() < 0.6
                ):
                    return candidate

        none_candidates = [c for c in candidates if c["type"] == MjMove.none.value]
        if len(none_candidates) == 0:
            raise Exception("not intended path, none candidate not found.")
        else:
            return none_candidates[0]

    @classmethod
    def calclate_valid_nums(cls, tehais):
        tehai = np.zeros(34, dtype=int)
        for t in tehais:
            tehai[t.id] += 1

        # assert all([t >= 0 and t <= 4 for t in tehai])

        start = datetime.datetime.now()

        # calclate 2 times change
        moves = {}
        for i in range(34):  # remove
            for j in range(34):  # add
                if i == j:
                    continue
                for k in range(34):  # add
                    if i == k:
                        continue

                    tmp_tehai = tehai.copy()
                    tmp_tehai[i] -= 1
                    tmp_tehai[j] += 1
                    tmp_tehai[k] += 1
                    if tmp_tehai[i] >= 0 and tmp_tehai[j] <= 4 and tmp_tehai[k] <= 4:
                        moves[(i, j, k)] = cls.ShantenAnalyser.calc_all_shanten(
                            tmp_tehai, 0
                        )

        end = datetime.datetime.now()
        return tehai


class Node:
    shanten_analysis = RsShantenAnalysis()

    def __init__(self, tehais, taken=None):

        # copy tehai state
        self.tehai = [0] * 34
        for t in tehais:
            self.tehai[t.id] += 1
        self.furo_num = (14 - sum(self.tehai)) // 3

        if taken:
            self.tehai[taken.id] += 1

    def valid_change(self, sub_id, add_id):
        return not (self.tehai[add_id] >= 4 or self.tehai[sub_id] <= 0)

    def valid_sub(self, sub_id):
        return self.tehai[sub_id] > 0

    def valid_add(self, add_id):
        return self.tehai[add_id] < 4

    def change(self, sub_id, add_id):
        # assert self.valid_change(sub_id, add_id)
        self.tehai[sub_id] -= 1
        self.tehai[add_id] += 1

    def sub(self, sub_id):
        # assert self.valid_sub(sub_id)
        self.tehai[sub_id] -= 1

    def add(self, add_id):
        # assert self.valid_add(add_id)
        self.tehai[add_id] += 1

    def calc_ukeire_num(self, rest_num=None):
        if rest_num is None:
            rest_num = [4] * 34
            for i in range(34):
                rest_num[i] -= self.tehai[i]

        target_shanten = self.shanten
        target_cache = {}
        node = copy.deepcopy(self)

        for i in range(34):  # da loop
            if self.tehai[i] == 0:
                continue
            for j in range(34):  # tsumo loop
                if rest_num[j] == 0:
                    continue
                if i == j:
                    continue
                if node.valid_change(sub_id=i, add_id=j):
                    node.change(i, j)
                    changed_shanten = node.shanten
                    node.change(j, i)

                    if changed_shanten < target_shanten:
                        target_cache = {}
                        target_shanten = changed_shanten
                    if changed_shanten == target_shanten:
                        target_cache[(i, j)] = changed_shanten

        valid_nums = [0] * 34
        for n in target_cache:
            da_index = n[0]
            tsumo_index = n[1]
            valid_nums[da_index] += rest_num[tsumo_index]

        return valid_nums

    @property
    def shanten(self):
        normal, chitoitsu, kokushi = self.shanten_analysis.calc_all_shanten(
            self.tehai, self.furo_num
        )
        if chitoitsu < 1 and chitoitsu < normal and chitoitsu < kokushi:
            return chitoitsu
        if kokushi < 3 and kokushi < normal and kokushi < chitoitsu:
            return kokushi
        else:
            return normal


if __name__ == "__main__":
    pais = [
        "1m",
        "2m",
        "3m",
        "4m",
        "8m",
        "2p",
        "3p",
        "8p",
        "8p",
        "9p",
        "1s",
        "4s",
        "5s",
        "7p",
    ]
    pais = [Pai.from_str(p) for p in pais]
    node = Node(pais)
    print(pais)
    print(node.calc_ukeire_num())

    start = datetime.datetime.now()
    print(start)

    loopnum = 1000
    for i in range(loopnum):
        node.calc_ukeire_num()

    end = datetime.datetime.now()
    print(datetime.datetime.now())
    print(loopnum, end - start)
