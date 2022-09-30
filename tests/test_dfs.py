import conftest
from mjaigym import shanten
from mjaigym.board.function.efficient_dfs import Dfs
from mjaigym.board.function.pai import Pai


def test_dfs_pattern_chitoitsu():
    dfs = Dfs()

    # chitoitsu pattern
    nums = [
        0,
        2,
        0,
        0,
        2,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        2,
        0,
        0,
        1,
    ]
    dora_ids = [1, 5, 33]
    doras = [p for p in Pai.from_idlist(dora_ids)]
    depth = 3
    _, _, chitoitsu_shanten = shanten.get_shanten_all(nums, 0)
    results = dfs.dfs_with_score_chitoitsu(
        nums,
        furos=[],
        depth=depth,
        shanten_chitoitsu=chitoitsu_shanten,
        doras=doras,
    )
    result = sorted(results, key=lambda x: x.point_info.points)[-1]

    print(f"{nums} \n-> {result}, depth:{depth}, {[d.id for d in doras]}")

    dora_id_contains = False
    for toitsu in result.combination:
        dora_id_contains |= toitsu[0] in dora_ids

    assert dora_id_contains


def test_dfs_pattern_chitoitsu_toitsu_tanki():
    dfs = Dfs()
    # can add new dora and dora tanki test
    nums = [
        0,
        0,
        2,
        0,
        2,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        2,
        0,
        0,
        1,
    ]
    dora_ids = [1, 5, 20, 33]
    doras = [p for p in Pai.from_idlist(dora_ids)]
    depth = 3

    _, _, chitoitsu_shanten = shanten.get_shanten_all(nums, 0)
    results = dfs.dfs_with_score_chitoitsu(
        nums,
        furos=[],
        depth=depth,
        shanten_chitoitsu=chitoitsu_shanten,
        doras=doras,
    )
    result = sorted(results, key=lambda x: x.point_info.points)[-1]

    print(f"{nums} \n-> {result}, depth:{depth}, {[d.id for d in doras]}")

    contains_33 = False
    contains_1_5_20 = False

    for toitsu in result.combination:
        if toitsu[0] in [1, 5, 20]:
            contains_1_5_20 = True
        if toitsu[0] == 33:
            contains_33 = True

    assert contains_1_5_20 & contains_33


def test_dfs_pattern_chitoitsu_tanki():
    dfs = Dfs()
    # can add new dora and dora tanki test
    nums = [
        0,
        0,
        2,
        0,
        2,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        2,
        0,
        0,
        1,
    ]
    dora_ids = [1, 5, 20, 33]
    doras = [p for p in Pai.from_idlist(dora_ids)]
    depth = 1

    _, _, chitoitsu_shanten = shanten.get_shanten_all(nums, 0)
    results = dfs.dfs_with_score_chitoitsu(
        nums,
        furos=[],
        depth=depth,
        shanten_chitoitsu=chitoitsu_shanten,
        doras=doras,
    )
    result = sorted(results, key=lambda x: x.point_info.points)[-1]

    print(f"{nums} \n-> {result}, depth:{depth}, {[d.id for d in doras]}")

    contains_33 = False
    contains_1_5_20 = False

    for toitsu in result.combination:
        if toitsu[0] in [1, 5, 20]:
            contains_1_5_20 = True
        if toitsu[0] == 33:
            contains_33 = True

    assert (not contains_1_5_20) & contains_33


def test_dfs_pattern_hitoitsu_learge_depth():
    dfs = Dfs()
    # can add new dora and dora tanki test
    nums = [
        0,
        2,
        0,
        2,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        2,
        0,
        0,
        1,
    ]
    dora_ids = [1, 5, 27, 33]
    doras = [p for p in Pai.from_idlist(dora_ids)]
    depth = 9

    _, _, chitoitsu_shanten = shanten.get_shanten_all(nums, 0)

    results = dfs.dfs_with_score_chitoitsu(
        nums,
        furos=[],
        depth=3,
        shanten_chitoitsu=chitoitsu_shanten,
        doras=doras,
    )
    result = sorted(results, key=lambda x: x.point_info.points)[-1]
    print(f"{nums} \n-> {result}, depth:{depth}, {[d.id for d in doras]}")

    contains_1 = False
    contains_5 = False
    contains_27 = False
    contains_33 = False

    for toitsu in result.combination:
        if toitsu[0] == 1:
            contains_1 = True
        if toitsu[0] == 5:
            contains_5 = True
        if toitsu[0] == 27:
            contains_27 = True
        if toitsu[0] == 33:
            contains_33 = True

    assert contains_1 & contains_5 & contains_27 & contains_33


def test_dfs_score():
    dfs = Dfs()
    # can add new dora and dora tanki test
    nums = [
        0,
        0,
        2,
        0,
        2,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        2,
        0,
        0,
        1,
    ]
    dora_ids = [1, 5, 27, 33]
    doras = [p for p in Pai.from_idlist(dora_ids)]
    depth = 9

    _, _, chitoitsu_shanten = shanten.get_shanten_all(nums, 0)

    result = dfs.dfs_with_score_chitoitsu(
        nums,
        furos=[],
        depth=3,
        shanten_chitoitsu=chitoitsu_shanten,
        doras=doras,
    )

    print(f"{nums} \n-> {result}, depth:{depth}, {[d.id for d in doras]}")
    assert len(result) > 0


def test_dfs_pattern_kokushi():
    dfs = Dfs()
    # can add new dora and dora tanki test
    nums = [
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
    ]
    dora_ids = [1, 5, 27, 33]
    doras = [p for p in Pai.from_idlist(dora_ids)]
    depth = 3
    oya = True

    _, shanten_kokushi, _ = shanten.get_shanten_all(nums, 0)

    results = dfs.dfs_with_score_kokushi(
        nums,
        furos=[],
        depth=3,
        oya=oya,
        shanten_kokushi=shanten_kokushi,
    )

    print(f"{nums} \n-> {results}, depth:{depth}, {[d.id for d in doras]}")
    assert len(results) > 0

    kokushi_exists = False
    for result in results:
        yakus = result.point_info.yakus
        if any([yaku[0] == "kokushimuso" for yaku in yakus]):
            kokushi_exists = True
    assert kokushi_exists


# def test_bench_dahaied_dfs():
#     import datetime
#     dfs = Dfs()
#     # can add new dora and dora tanki test
#     nums = [
#         0,0,1,3,0,0,1,0,0,
#         0,0,0,1,0,0,0,0,0,
#         0,0,0,1,1,0,0,3,0,
#         0,0,0,3,0,0,0,
#     ]
#     dora_ids = [1, 5, 27, 33]
#     doras = [p for p in Pai.from_idlist(dora_ids)]
#     depth = 3
#     shanten_cache = {}
#     dahaied_results = {}
#     start = datetime.datetime.now()
#     for _ in range(10):
#         for i in range(len(nums)):
#             if nums[i]==0:
#                 continue
#             nums[i] -= 1
#             normal_shanten, _, _ = shanten.get_shanten_all(nums, 0)

#             results = dfs.dfs_with_score_normal(
#                 tehai=nums,
#                 furos=[],
#                 depth=depth,
#                 shanten_normal=normal_shanten,
#                 oya=False,
#                 bakaze="E",
#                 jikaze="S",
#                 doras=doras,
#                 uradoras=[],
#                 num_akadoras=0,
#             )
#             # print(f"{nums} \n-> {results}, depth:{depth}, {[d.id for d in doras]}")
#             dahaied_results[i] = results
#             nums[i] += 1

#             # print(i, "shanten", normal_shanten, "results len", len(results))
#             # for result in results:
#             #     print(i, result)


#     end = datetime.datetime.now()
#     print(f"need time for dfs {end - start}")
#     assert len(dahaied_results) > 0
