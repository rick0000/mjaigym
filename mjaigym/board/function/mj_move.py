from enum import Enum


class MjMove(Enum):
    start_game = "start_game"
    start_kyoku = "start_kyoku"
    tsumo = "tsumo"
    dahai = "dahai"
    chi = "chi"
    pon = "pon"
    ankan = "ankan"
    daiminkan = "daiminkan"
    kakan = "kakan"
    reach = "reach"
    reach_accepted = "reach_accepted"
    dora = "dora"
    hora = "hora"
    ryukyoku = "ryukyoku"
    end_kyoku = "end_kyoku"
    end_game = "end_game"
    none = "none"

    join = "join"
    error = "error"
    hello = "hello"
