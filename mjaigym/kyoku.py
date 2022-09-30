import numpy as np
from mjaigym import loggers as lgs


class Kyoku:
    def __init__(self, kyoku_mjsons, initial_scores):
        self.initial_scores = initial_scores
        self.kyoku_id = None
        self.bakaze = None
        self.honba = None
        self.oya = None
        self.result_scores = None
        self.kyoku_end_label = None
        self.one_kyoku_sample = None
        self.kyoku_mjsons = kyoku_mjsons

        start_lines = list(filter(lambda x: x["type"] == "start_kyoku", kyoku_mjsons))
        if len(start_lines) != 1:
            lgs.logger_main.warn("start_kyoku num not equal 1")
            return

        start_line = start_lines[0]
        self.kyoku_id = start_line["kyoku"]
        self.bakaze = start_line["bakaze"]
        self.honba = start_line["honba"]
        self.oya = start_line["oya"]

        self.end_lines = list(
            filter(
                lambda x: x["type"] == "hora" or x["type"] == "ryukyoku", kyoku_mjsons
            )
        )
        if len(self.end_lines) == 0:
            lgs.logger_main.warn("end_kyoku num is 0")
            return

        self.result_scores = self.end_lines[-1]["scores"]

    @classmethod
    def convert_bakaze_to_id(cls, bakaze):
        if bakaze == "E":
            return 0
        elif bakaze == "S":
            return 1
        elif bakaze == "W":
            return 2
        elif bakaze == "N":
            return 3

    @classmethod
    def calc_result_label(cls, result_line):
        result_type = result_line["type"]
        if result_type == "hora":
            actor = result_line["actor"]
            target = result_line["target"]
            return Kyoku.get_hora_label(actor, target)
        elif result_type == "ryukyoku":
            tenpais = result_line["tenpais"]
            return Kyoku.get_tenpai_label(tenpais)
        else:
            raise Exception("invalid type")

    @classmethod
    def get_hora_label(cls, actor, target):
        return cls._horas.index([actor, target])

    @classmethod
    def get_tenpai_label(cls, tenpais):
        hora_result_label_offset = len(cls._horas)
        return cls._tenpais.index(tenpais) + hora_result_label_offset

    _horas = [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 0],
        [1, 1],
        [1, 2],
        [1, 3],
        [2, 0],
        [2, 1],
        [2, 2],
        [2, 3],
        [3, 0],
        [3, 1],
        [3, 2],
        [3, 3],
    ]
    _tenpais = [
        [
            False,
            False,
            False,
            False,
        ],
        [
            False,
            False,
            False,
            True,
        ],
        [
            False,
            False,
            True,
            False,
        ],
        [
            False,
            False,
            True,
            True,
        ],
        [
            False,
            True,
            False,
            False,
        ],
        [
            False,
            True,
            False,
            True,
        ],
        [
            False,
            True,
            True,
            False,
        ],
        [
            False,
            True,
            True,
            True,
        ],
        [
            True,
            False,
            False,
            False,
        ],
        [
            True,
            False,
            False,
            True,
        ],
        [
            True,
            False,
            True,
            False,
        ],
        [
            True,
            False,
            True,
            True,
        ],
        [
            True,
            True,
            False,
            False,
        ],
        [
            True,
            True,
            False,
            True,
        ],
        [
            True,
            True,
            True,
            False,
        ],
        [
            True,
            True,
            True,
            True,
        ],
    ]
