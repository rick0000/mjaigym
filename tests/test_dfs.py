import conftest
from mjaigym.board.function.efficient_dfs import Dfs
from mjaigym.board.function.pai import Pai
from mjaigym import shanten

def test_dfs_pattern_chitoitsu():
    dfs = Dfs()
    
    # chitoitsu pattern
    nums = [
        0,2,0,0,2,0,2,0,0,
        0,0,0,0,0,2,0,0,0,
        0,0,0,0,2,0,0,0,1,
        0,0,0,2,0,0,1,
    ]
    dora_ids = [1,5,33]
    doras = [p for p in Pai.from_idlist(dora_ids)]
    depth = 3
    _, _, chitoitsu_shanten = shanten.get_shanten_all(nums, 0)
    results = dfs.dfs_chitoitsu(
            nums, 
            depth, 
            doras,
            chitoitsu_shanten,
        )
    max_result = sorted(results, key=lambda x:x[1])[-1]
    
    print(f"{nums} \n-> {max_result}, depth:{depth}, {[d.id for d in doras]}")
    assert len(max_result) > 0
    
    dora_id_contains = False
    for toitsu in max_result[0]:
        dora_id_contains |= toitsu[0] in dora_ids
        
    assert dora_id_contains

def test_dfs_pattern_chitoitsu_toitsu_tanki():
    dfs = Dfs()
    # can add new dora and dora tanki test
    nums = [
        0,0,2,0,2,0,2,0,0,
        0,0,0,0,0,2,0,0,0,
        0,0,0,0,2,0,0,0,1,
        0,0,0,2,0,0,1,
    ]
    dora_ids = [1, 5, 20, 33]
    doras = [p for p in Pai.from_idlist(dora_ids)]
    depth = 3

    _, _, chitoitsu_shanten = shanten.get_shanten_all(nums, 0)
    results = dfs.dfs_chitoitsu(
            nums, 
            depth, 
            doras,
            chitoitsu_shanten,
        )
    max_result = sorted(results, key=lambda x:x[1])[-1]
    
    print(f"{nums} \n-> {max_result}, depth:{depth}, {[d.id for d in doras]}")
    assert len(max_result) > 0
    
    contains_33 = False
    contains_1_5_20 = False
    for toitsu in max_result[0]:
        if toitsu[0] in [1,5,20]:
            contains_1_5_20 = True
        if toitsu[0] == 33:
            contains_33 = True
        
    assert contains_1_5_20 & contains_33

def test_dfs_pattern_chitoitsu_tanki():
    dfs = Dfs()
    # can add new dora and dora tanki test
    nums = [
        0,0,2,0,2,0,2,0,0,
        0,0,0,0,0,2,0,0,0,
        0,0,0,0,2,0,0,0,1,
        0,0,0,2,0,0,1,
    ]
    dora_ids = [1, 5, 20, 33]
    doras = [p for p in Pai.from_idlist(dora_ids)]
    depth = 1

    _, _, chitoitsu_shanten = shanten.get_shanten_all(nums, 0)
    results = dfs.dfs_chitoitsu(
            nums, 
            depth, 
            doras,
            chitoitsu_shanten,
        )
    max_result = sorted(results, key=lambda x:x[1])[-1]

    print(f"{nums} \n-> {max_result}, depth:{depth}, {[d.id for d in doras]}")
    assert len(max_result) > 0
    
    contains_33 = False
    contains_1_5_20 = False
    for toitsu in max_result[0]:
        if toitsu[0] in [1,5,20]:
            contains_1_5_20 = True
        if toitsu[0] == 33:
            contains_33 = True
        
    assert (not contains_1_5_20) & contains_33


def test_dfs_pattern_hitoitsu_learge_depth():
    dfs = Dfs()
    # can add new dora and dora tanki test
    nums = [
        0,0,2,0,2,0,2,0,0,
        0,0,0,0,0,2,0,0,0,
        0,0,0,0,2,0,0,0,1,
        0,0,0,2,0,0,1,
    ]
    dora_ids = [1, 5, 27, 33]
    doras = [p for p in Pai.from_idlist(dora_ids)]
    depth = 9

    _, _, chitoitsu_shanten = shanten.get_shanten_all(nums, 0)
    results = dfs.dfs_chitoitsu(
            nums, 
            depth, 
            doras,
            chitoitsu_shanten,
        )
    
    max_result = sorted(results, key=lambda x:x[1])[-1]

    print(f"{nums} \n-> {max_result}, depth:{depth}, {[d.id for d in doras]}")
    assert len(max_result) > 0
    
    
    contains_1 = False
    contains_5 = False
    contains_27 = False
    contains_33 = False
    
    for toitsu in max_result[0]:
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
        0,0,2,0,2,0,2,0,0,
        0,0,0,0,0,2,0,0,0,
        0,0,0,0,2,0,0,0,1,
        0,0,0,2,0,0,1,
    ]
    dora_ids = [1, 5, 27, 33]
    doras = [p for p in Pai.from_idlist(dora_ids)]
    depth = 9

    _, _, chitoitsu_shanten = shanten.get_shanten_all(nums, 0)

    result = dfs.dfs_with_score_chitoitsu(
            nums, 
            furos=[],
            depth=3,
            reach=True,
            shanten_chitoitsu=chitoitsu_shanten,
            doras=doras,
        )
    
    print(f"{nums} \n-> {result}, depth:{depth}, {[d.id for d in doras]}")
    assert len(result) > 0

def test_dfs_pattern_kokushi():
    dfs = Dfs()
    # can add new dora and dora tanki test
    nums = [
        1,0,0,0,0,0,0,0,1,
        1,0,0,0,0,0,0,0,1,
        1,0,0,0,0,0,0,0,1,
        1,1,1,1,1,1,2,
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
            oya = oya,
            shanten_kokushi=shanten_kokushi,
        )
    
    print(f"{nums} \n-> {results}, depth:{depth}, {[d.id for d in doras]}")
    assert len(results) > 0
    
    result = results[0]
    hora_pattern = result[0]
    hora_dist = result[1]
    assert "kokushimuso" in hora_pattern["yakus"]


