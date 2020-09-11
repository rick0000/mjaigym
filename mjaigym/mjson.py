import json
from pathlib import Path
from mjaigym.game import Game

class Mjson():
    def __init__(self, path:Path, mjsons):
        self.path = path
        self.mjsons = mjsons
        self.game = Game(mjsons)

    @classmethod
    def load(cls, path:Path):
        try:
            with open(path, 'rt') as f:
                mjsons = [json.loads(l) for l in f.read().splitlines() if l]
            return Mjson(path, mjsons)
        except:
            print(f"error @ {path}")
            raise
