import conftest
from mjaigym.kyoku import Kyoku
from mjaigym.reward import HoraActionReward, KyokuScoreReward


def test_hora_action_reward():

    hora_actor = 3

    actions = []
    actions.append(
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 3,
            "dora_marker": "2s",
            "tehais": [
                [
                    "9p",
                    "1m",
                    "P",
                    "S",
                    "3s",
                    "3m",
                    "E",
                    "7p",
                    "3m",
                    "S",
                    "3m",
                    "3p",
                    "8m",
                ],
                [
                    "6m",
                    "3m",
                    "W",
                    "E",
                    "4p",
                    "C",
                    "E",
                    "5s",
                    "3p",
                    "3s",
                    "2m",
                    "2m",
                    "C",
                ],
                [
                    "5s",
                    "W",
                    "8p",
                    "3p",
                    "4p",
                    "N",
                    "F",
                    "N",
                    "9p",
                    "1s",
                    "5pr",
                    "2s",
                    "9p",
                ],
                [
                    "S",
                    "5mr",
                    "7s",
                    "6p",
                    "4p",
                    "7p",
                    "9p",
                    "4s",
                    "8p",
                    "3s",
                    "9m",
                    "7m",
                    "2p",
                ],
            ],
        }
    )
    actions += [{"type": "none"}] * 5
    actions.append(
        {"type": "hora", "actor": hora_actor, "scores": [20000, 20000, 20000, 20000]}
    )
    actions.append({"type": "end_kyoku"})

    kyoku = Kyoku(actions, [0, 0, 0, 0])

    rewards = HoraActionReward.calc_from_kyoku(kyoku)

    assert len(rewards) == len(actions)
    assert len(rewards[0]) == 4

    # hora line
    assert rewards[-2][0] == 0
    assert rewards[-2][1] == 0
    assert rewards[-2][2] == 0
    assert rewards[-2][hora_actor] == 0

    # end kyoku line
    assert rewards[-1][0] == 0
    assert rewards[-1][1] == 0
    assert rewards[-1][2] == 0
    assert rewards[-1][hora_actor] == 1


def test_kyoku_score_reward():
    initial_score = [
        25000,
        25000,
        25000,
        25000,
    ]
    end_score = [26000, 25000, 48000, -10000]  # [+, +-0, +, -]

    actions = []
    actions.append(
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 3,
            "dora_marker": "2s",
            "tehais": [
                [
                    "9p",
                    "1m",
                    "P",
                    "S",
                    "3s",
                    "3m",
                    "E",
                    "7p",
                    "3m",
                    "S",
                    "3m",
                    "3p",
                    "8m",
                ],
                [
                    "6m",
                    "3m",
                    "W",
                    "E",
                    "4p",
                    "C",
                    "E",
                    "5s",
                    "3p",
                    "3s",
                    "2m",
                    "2m",
                    "C",
                ],
                [
                    "5s",
                    "W",
                    "8p",
                    "3p",
                    "4p",
                    "N",
                    "F",
                    "N",
                    "9p",
                    "1s",
                    "5pr",
                    "2s",
                    "9p",
                ],
                [
                    "S",
                    "5mr",
                    "7s",
                    "6p",
                    "4p",
                    "7p",
                    "9p",
                    "4s",
                    "8p",
                    "3s",
                    "9m",
                    "7m",
                    "2p",
                ],
            ],
        }
    )
    actions += [{"type": "none"}] * 5
    actions.append({"type": "hora", "actor": 0, "scores": end_score})
    actions.append({"type": "end_kyoku"})

    kyoku = Kyoku(actions, initial_score)

    rewards = KyokuScoreReward.calc_from_kyoku(kyoku)

    assert len(rewards) == len(actions)
    assert len(rewards[0]) == 4

    # hora line
    assert rewards[-2][0] == 0
    assert rewards[-2][1] == 0
    assert rewards[-2][2] == 0
    assert rewards[-2][3] == 0

    # end kyoku line
    assert rewards[-1][0] == end_score[0] - initial_score[0]
    assert rewards[-1][1] == end_score[1] - initial_score[1]
    assert rewards[-1][2] == end_score[2] - initial_score[2]
    assert rewards[-1][3] == end_score[3] - initial_score[3]

    reward_calclator = KyokuScoreReward()
    for action in actions:
        reward_calclator.step(action)

    assert reward_calclator.calc() == [
        end_score[0] - initial_score[0],
        end_score[1] - initial_score[1],
        end_score[2] - initial_score[2],
        end_score[3] - initial_score[3],
    ]
