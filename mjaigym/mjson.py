import json
from pathlib import Path

from mjaigym.game import Game


class Mjson:
    def __init__(self, path: Path, mjsons):
        self.path = path
        self.mjsons = mjsons
        self.game = Game(mjsons)

    @classmethod
    def load(cls, path: Path):
        try:
            with open(path, "rt") as f:
                mjsons = [json.loads(l) for l in f.read().splitlines() if l]

            scores = [25000, 25000, 25000, 25000]
            pai = "?"
            for line in mjsons:

                # update scores
                if "scores" in line:
                    scores = line["scores"]
                elif "deltas" in line:
                    for i in range(4):
                        scores[i] += line["deltas"][i]
                elif line["type"] == "reach_accepted":
                    scores[line["actor"]] -= 1000

                if "pai" in line:
                    pai = line["pai"]

                # add scores to line
                if line["type"] == "hora":
                    if "scores" not in line:
                        line["scores"] = scores
                    if "pai" not in line:
                        line["pai"] = pai

                if line["type"] in [
                    "start_kyoku",
                    "reach_accepted",
                    "ryukyoku",
                    "end_game",
                ]:
                    if "scores" not in line:
                        line["scores"] = scores

            return Mjson(path, mjsons)
        except:
            print(f"error @ {path}")
            raise
