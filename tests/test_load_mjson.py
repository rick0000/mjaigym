import conftest

from mjaigym.mjson import Mjson


def test_load_mjson():
    mjson = Mjson.load("sample.mjson")

    