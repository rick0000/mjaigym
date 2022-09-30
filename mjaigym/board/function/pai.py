from typing import List


class Pai:
    """class for mahjong pai\n
    type :str [m,p,s,z] pai type\n
    number :int [1 ~ 9] pai number\n
    is_red  :bool           pai is red pai\n
    str     :str            pai string\n
    id      :int  [0 ~ 33]  pai id in 34 types\n
    """

    def __init__(self, type: str, number: int, is_red: bool, pai_str: str):
        self.type = type
        self.number = number
        self.is_red = is_red
        self.str = pai_str
        self.id = Pai.str_to_id(pai_str)

    def __eq__(self, other):
        # if not isinstance(other, Pai):
        #     return False
        return (
            self.number == other.number
            and self.type == other.type
            and self.is_red == other.is_red
        )

    def __hash__(self):
        return self.id

    def __repr__(self):
        return self.str

    def __lt__(self, other):
        return Pai.SORT_ORDER[self.str] < Pai.SORT_ORDER[other.str]

    @classmethod
    def from_idlist(cls, pais: List[str]):
        return [Pai.from_id(p) for p in pais]

    @classmethod
    def from_list(cls, pais: List[str]):
        return [Pai.from_str(p) for p in pais]

    @classmethod
    def from_str(cls, pai_str: str):
        type = cls.str_to_type(pai_str)
        number = cls.str_to_num(pai_str)
        is_red = pai_str[-1] == "r"
        return Pai(type, number, is_red, pai_str)

    @classmethod
    def from_id(cls, pai_id: int):
        pai_str = cls.id_to_str(pai_id, False)
        type = cls.str_to_type(pai_str)
        number = cls.str_to_num(pai_str)
        is_red = pai_str[-1] == "r"
        return Pai(type, number, is_red, pai_str)

    @classmethod
    def from_number_type(cls, number: int, type: str):
        if type == "z":
            # assert 1 <= number and number <= 7
            pai_str = ["E", "S", "W", "N", "P", "F", "C"][number - 1]
        else:
            pai_str = f"{number}{type}"

        return Pai.from_str(pai_str)

    def remove_red(self):
        """returns same symbol (=same number, same type) pai removed red.
        example)
            3m -> 3m
            5mr -> 5m
            5pr -> 5p
            5sr -> 5s

        Returns:
            Pai: same symbol pai
        """
        if self.is_red:
            new_str = self.str[:-1]
        else:
            new_str = self.str

        return Pai(self.type, self.number, False, new_str)

    def is_sangenpai(self) -> bool:
        """returns pai is in [P, F, C]

        Returns:
            bool: self is sangenpai
        """
        return self.str in ["P", "F", "C"]

    def is_wind(self) -> bool:
        """returns pai is in [E, S, W, N]

        Returns:
            bool: self is wind
        """
        return self.str in ["E", "S", "W", "N"]

    def is_jihai(self) -> bool:
        """returns pai is in [E, S, W, N, P, F, C]

        Returns:
            bool: self is jihai
        """
        return self.str in ["E", "S", "W", "N", "P", "F", "C"]

    def is_number(self) -> bool:
        """returns pai type is in [manzu, pinzu, souzu]

        Returns:
            bool: self is number
        """
        return self.type in ["m", "p", "s"]

    def is_yaochu(self) -> bool:
        """returns pai number is in [1, 9] and is_number() or
            pai number is in [1, 7] and is_jihai()

        Returns:
            bool: self is yaochu
        """
        return self.number in [1, 9] or self.type == "z"

    def is_same_symbol(self, other) -> bool:
        """returns self and other number and type is same.

        Returns:
            bool: is same symbole
        """

        if type(other) != type(self):
            return False
        return self.number == other.number and self.type == other.type

    @property
    def succ(self):
        """returns next pai with dora calclate rule, this is used for calclate dora from doramarker
        example)
            1m -> 2m
            9m -> 1m
            P -> F
            C -> P
            N -> E

        Returns:
            Pai: next Pai with dora calclate rule
        """
        if self.type != "z" and self.number == 9:
            number = 1
        elif self.type == "z" and self.number == 4:
            number = 1
        elif self.type == "z" and self.number == 7:
            number = 5
        else:
            number = self.number + 1
        return Pai.from_number_type(number, self.type)

    @classmethod
    def sort(cls, pai):
        return cls.SORT_ORDER[pai.str]

    SORT_ORDER = {
        "1m": 110,
        "2m": 120,
        "3m": 130,
        "4m": 140,
        "5m": 150,
        "5mr": 151,
        "6m": 160,
        "7m": 170,
        "8m": 180,
        "9m": 190,
        "1p": 210,
        "2p": 220,
        "3p": 230,
        "4p": 240,
        "5p": 250,
        "5pr": 251,
        "6p": 260,
        "7p": 270,
        "8p": 280,
        "9p": 290,
        "1s": 310,
        "2s": 320,
        "3s": 330,
        "4s": 340,
        "5s": 350,
        "5sr": 350,
        "6s": 360,
        "7s": 370,
        "8s": 380,
        "9s": 390,
        "E": 410,
        "S": 420,
        "W": 430,
        "N": 440,
        "P": 450,
        "F": 460,
        "C": 470,
        "?": 500,
    }

    @classmethod
    def str_to_num(cls, pai_str):
        if pai_str == "?":
            return None

        # assert pai_str in cls.STR_ID
        return (cls.STR_ID[pai_str] % 9) + 1

    @classmethod
    def str_to_type(cls, pai_str):
        if pai_str == "?":
            return None
        # assert pai_str in cls.STR_ID
        type_id = cls.STR_ID[pai_str] // 9
        return ["m", "p", "s", "z"][type_id]

    @classmethod
    def str_to_id(cls, pai_str):
        # assert pai_str in cls.STR_ID
        return cls.STR_ID[pai_str]

    STR_ID = {
        "1m": 0,
        "2m": 1,
        "3m": 2,
        "4m": 3,
        "5m": 4,
        "5mr": 4,
        "6m": 5,
        "7m": 6,
        "8m": 7,
        "9m": 8,
        "1p": 9,
        "2p": 10,
        "3p": 11,
        "4p": 12,
        "5p": 13,
        "5pr": 13,
        "6p": 14,
        "7p": 15,
        "8p": 16,
        "9p": 17,
        "1s": 18,
        "2s": 19,
        "3s": 20,
        "4s": 21,
        "5s": 22,
        "5sr": 22,
        "6s": 23,
        "7s": 24,
        "8s": 25,
        "9s": 26,
        "E": 27,
        "S": 28,
        "W": 29,
        "N": 30,
        "P": 31,
        "F": 32,
        "C": 33,
        "?": 100,
    }

    @classmethod
    def id_to_str(cls, pai_id, is_red):
        # assert pai_id in cls.ID_STR
        if is_red:
            return cls.ID_STR[pai_id] + "r"
        else:
            return cls.ID_STR[pai_id]

    ID_STR = {
        0: "1m",
        1: "2m",
        2: "3m",
        3: "4m",
        4: "5m",
        5: "6m",
        6: "7m",
        7: "8m",
        8: "9m",
        9: "1p",
        10: "2p",
        11: "3p",
        12: "4p",
        13: "5p",
        14: "6p",
        15: "7p",
        16: "8p",
        17: "9p",
        18: "1s",
        19: "2s",
        20: "3s",
        21: "4s",
        22: "5s",
        23: "6s",
        24: "7s",
        25: "8s",
        26: "9s",
        27: "E",
        28: "S",
        29: "W",
        30: "N",
        31: "P",
        32: "F",
        33: "C",
        100: "?",
    }


UNKNOWN_PAI_STR = "?"
UNKNOWN_PAI = Pai("?", Pai.STR_ID["?"], False, "?")


if __name__ == "__main__":
    Pai.str_to_id("5m")
