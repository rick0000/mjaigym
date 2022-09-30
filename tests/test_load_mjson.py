import conftest
from mjaigym.mjson import Mjson


def test_load_mjson():
    mjson = Mjson.load("sample.mjson")

    for kyoku in mjson.game.kyokus:
        for line in kyoku.kyoku_mjsons:
            if line["type"] == "reach_accepted":
                assert "scores" in line


def test_load_normal_mjson():
    mjson = Mjson.load("normal_sample.mjson")
