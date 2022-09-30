import copy
import sys
from collections import OrderedDict

from mjaigym.board.function.pai import Pai

# ryanpen = ryanmen or penchan
MENTSU_TYPES = ["kotsu", "shuntsu", "toitsu", "ryanpen", "kanta", "single"]
MENTSU_CATEGORIES = {
    "kotsu": "complete",
    "shuntsu": "complete",
    "toitsu": "toitsu",
    "ryanpen": "tatsu",
    "kanta": "tatsu",
    "single": "single",
}
MENTSU_SIZES = {
    "complete": 3,
    "toitsu": 2,
    "tatsu": 2,
    "single": 1,
}
ALL_TYPES = ["normal", "chitoitsu", "kokushimuso"]


class ShantenAnalysis:
    @classmethod
    def benchmark(cls):
        import datetime

        from mjaigym.board.function.yama import Yama

        iter_num = 100
        tests = []
        for _ in range(iter_num):
            y = Yama()
            pais = y.all_yama[0:13]
            pais = sorted(pais)
            tests.append(pais)

        start = datetime.datetime.now()
        for test in tests:
            shanten = ShantenAnalysis(test, max_shanten=100)
            print(f"bench:{test} => shanten:{shanten.shanten}")
        end = datetime.datetime.now()
        print(f"need {(end-start).microseconds / 1000} msec for {iter_num} iteration")

    def __init__(
        self,
        pais,
        max_shanten=None,  # 返り値の解析結果に含める分割候補のシャンテン数上限値
        shanten_types=None,
        num_used_pais=None,
        need_all_combinations=True,
    ):
        if num_used_pais is None:
            num_used_pais = len(pais)
        if shanten_types is None:
            shanten_types = ALL_TYPES
        self.pais = sorted(pais)
        self.max_shanten = max_shanten
        self.num_used_pais = num_used_pais
        self.need_all_combinations = need_all_combinations

        if self.num_used_pais % 3 == 0:
            raise Exception("invalid number of pais")

        # self.pai_set keep sorted
        self.pai_set = OrderedDict()

        for p in self.pais:
            p_symbol = p.remove_red()
            if p_symbol not in self.pai_set:
                self.pai_set[p_symbol] = 1
            else:
                self.pai_set[p_symbol] += 1

        self.cache = {}

        results = []
        if "normal" in shanten_types:
            results.append(self.count_normal(self.pai_set, []))
        if "chitoitsu":
            results.append(self.count_chitoi(self.pai_set))
        if "kokushimuso":
            results.append(self.count_kokushi(self.pai_set))

        self.shanten = sys.maxsize
        self.combinations = []
        for r in results:
            shanten = r[0]
            combinations = r[1]
            if self.max_shanten and shanten > self.max_shanten:
                continue

            if shanten < self.shanten:
                self.shanten = shanten
                self.combinations = combinations
            elif shanten == self.shanten:
                self.combinations += combinations

    def count_normal(self, pai_set, mentsus):
        key = self.get_key(pai_set, mentsus)
        min_combinations = []
        # print(key)
        if key not in self.cache:
            if len(pai_set) == 0:
                min_shanten = self.get_min_shanten_for_mentsus(mentsus)
                min_combinations = [mentsus]
            else:
                if self.max_shanten:
                    shanten_lowerbound = self.get_min_shanten_for_mentsus(mentsus)
                else:
                    shanten_lowerbound = sys.maxsize

                if (self.max_shanten) and (shanten_lowerbound > self.max_shanten):
                    min_shanten = sys.maxsize
                    min_combinations = []
                else:

                    min_shanten = sys.maxsize
                    first_pai = sorted(pai_set.keys(), key=Pai.sort)[0]
                    for type in MENTSU_TYPES:

                        if self.max_shanten == -1:
                            if type in ["ryanpen", "kanta"]:
                                continue
                            if (
                                any([m[0] == "toitsu" for m in mentsus])
                                and type == "toitsu"
                            ):
                                continue

                        (removed_pais, remains_set) = self.remove(
                            pai_set, type, first_pai
                        )
                        if remains_set is not None:
                            (shanten, combinations) = self.count_normal(
                                remains_set, mentsus + [[type, removed_pais]]
                            )
                            if shanten < min_shanten:
                                min_shanten = shanten
                                min_combinations = combinations
                                if (
                                    self.need_all_combinations == False
                                    and min_shanten == -1
                                ):
                                    break
                            elif shanten == min_shanten and shanten < sys.maxsize:
                                min_combinations += combinations

            self.cache[key] = [min_shanten, min_combinations]
        return self.cache[key]

    def count_chitoi(self, pai_set):
        num_toitsu = len([p for p in pai_set if pai_set[p] >= 2])
        num_singles = len([p for p in pai_set if pai_set[p] == 1])
        if num_toitsu == 6 and num_singles == 0:
            shanten = 1
        else:
            shanten = -1 + max(7 - num_toitsu, 0)

        return [shanten, ["chitoitsu"]]

    def count_kokushi(self, pai_set):
        yaochus = [p for p in pai_set if p.is_yaochu() or p.is_jihai()]
        has_yaochu_toitsu = any([pai_set[y] >= 2 for y in yaochus])

        shanten = 13 - len(yaochus) - (1 if has_yaochu_toitsu else 0)
        return [shanten, ["kokushimuso"]]

    def get_min_shanten_for_mentsus(self, mentsus):
        mentsu_categories = [MENTSU_CATEGORIES[m[0]] for m in mentsus]
        num_current_pais = sum([MENTSU_SIZES[m] for m in mentsu_categories])
        num_remain_pais = len(self.pais) - num_current_pais

        min_shantens = []

        if "toitsu" in mentsu_categories:
            mentsu_categories.remove("toitsu")
            min_shantens.append(
                self.get_min_shanten_without_janto(mentsu_categories, num_remain_pais)
            )
        else:
            min_shantens.append(
                self.get_min_shanten_without_janto(mentsu_categories, num_remain_pais)
                + 1
            )
            if num_remain_pais >= 2:
                min_shantens.append(
                    self.get_min_shanten_without_janto(
                        mentsu_categories, num_remain_pais - 2
                    )
                )
        return min(min_shantens)

    def get_min_shanten_without_janto(self, mentsu_categories, num_remain_pais):
        # Assume remain is complete
        mentsu_categories += ["complete"] * (num_remain_pais // 3)

        rest_mod = num_remain_pais % 3
        if rest_mod == 1:
            mentsu_categories.append("single")
        elif rest_mod == 2:
            mentsu_categories.append("toitsu")

        sizes = sorted([MENTSU_SIZES[s] for s in mentsu_categories], reverse=True)
        num_required_mentus = self.num_used_pais // 3
        return -1 + sum([(3 - m) for m in sizes[:num_required_mentus]])

    def remove(self, pai_set, type, first_pai):
        if type == "kotsu":
            removed_pais = [first_pai] * 3
        elif type == "shuntsu":
            removed_pais = self.shuntsu_piece(first_pai, [0, 1, 2])
        elif type == "toitsu":
            removed_pais = [first_pai] * 2
        elif type == "ryanpen":
            removed_pais = self.shuntsu_piece(first_pai, [0, 1])
        elif type == "kanta":
            removed_pais = self.shuntsu_piece(first_pai, [0, 2])
        elif type == "single":
            removed_pais = [first_pai]
        else:
            raise Exception("not intended path")

        if removed_pais is None:  # shuntsu_piece() return none
            return [None, None]

        result_set = copy.copy(pai_set)
        for pai in removed_pais:
            if pai in result_set and result_set[pai] > 0:
                result_set[pai] -= 1
                if result_set[pai] == 0:
                    result_set.pop(pai)
            else:
                return [None, None]
        return [removed_pais, result_set]

    def shuntsu_piece(self, first_pai, relative_numbers):
        if first_pai.type == "z":
            return None

        added = [first_pai.number + r for r in relative_numbers]
        if any([(1 <= a <= 9) == False for a in added]):
            return None
        else:
            return [
                Pai(
                    type=first_pai.type,
                    number=first_pai.number + n,
                    is_red=False,
                    pai_str=f"{first_pai.number+n}{first_pai.type}",  # this can only for shuntsu
                )
                for n in relative_numbers
            ]

    def get_key(self, pai_set, mentsus):

        pai_set_keys = [f"{item[0]}:{item[1]}" for item in pai_set.items()]

        mentsu_keys = [f"{item[0]}:{item[1]}" for item in mentsus]
        key = (",".join(pai_set_keys), ",".join(mentsu_keys))
        return key
