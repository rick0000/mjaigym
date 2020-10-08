import itertools
import datetime
from collections import deque

import joblib

from mjaigym.board.function.efficient_dfs import Dfs
from mjaigym.board.function.rs_shanten_analysis import RsShantenAnalysis


def create_all_patterns():
    nums = range(0,5)
    nums_9 = [nums] * 9
    items = itertools.product(*nums_9)
    valid_items = [item for item in items if sum(item)<= 14]
    
    return list(valid_items)

def filter_candidates(candidate):
    mentsu_num = sum(candidate) // 3
    mentsus = cut_mentsu(candidate, mentsu_num, 0)
    return mentsus
    

def create_num_valid_candidate(pattern, change):
    pai_num = sum(pattern) + sum(change)
    if pai_num % 3 != 0:
        return []
    if pai_num < 0:
        return []
    return apply_change(pattern, change)

def build_change_cache(depth=3):
    cache = {}
    cache_with_head = {}

    subs = [-3,-2,-1,0]
    adds = [0,1,2,3]

    changes = list(itertools.product(subs, adds))
    patterns = create_all_patterns()

    print(len(changes), len(patterns))
    for i,pattern in enumerate(patterns):
        print(datetime.datetime.now(), f"pattern:{i}/{len(patterns)}")
        pattern = list(pattern)
        key = get_key(pattern)
        cache[key] = {}
        for change in changes:
            
            
            """without head"""
            """crate valid pattern
            """
            changed_candidates = create_num_valid_candidate(pattern, change)
            
            mentsu_patterns = deque()

            for candidate in changed_candidates:
                filterd_candidates = filter_candidates(candidate)
                # print(pattern, change, len(filterd_candidates))
                mentsu_patterns.extend(filterd_candidates)
            if len(mentsu_patterns) > 0:
                
                cache[key][change] = mentsu_patterns
            
            # print(datetime.datetime.now() ,pattern, change, len(mentsu_patterns))
            # mentsus.extend()
                
            # print(mentsus)

            """search mentsus
            """
            # for mentu in mentsus:
            #     mentsu_valid_candidates = filter_candidates(mentu)
            #     cache[key][change] = mentsu_valid_candidates
            
            
            """with head"""
            """crate valid pattern
            """
            # candidates = apply_change(pattern, change)

            """search mentsus
            """
        
        import sys
        import json
        import copy
        if i % 1000 == 0:
            joblib.dump(cache, f"_{i}.pkl", compress=0)

            
    # joblib.dump([cache]*4000,"tehai_changed_cache.pkl")



def load_test():
    start = datetime.datetime.now()
    # a = joblib.load("_4000.pkl")
    # b = joblib.load("_3000.pkl")
    # c = joblib.load("_2000.pkl")
    d = joblib.load("_7000.pkl")

    end = datetime.datetime.now()
    print(end-start, "need load")


def main():
    dfs = Dfs()
    shanten_analysis = RsShantenAnalysis()
    tehai = [
        0,2,2,2,2,0,0,0,0,
        0,0,2,0,0,2,0,0,0,
        0,0,1,0,0,1,0,0,0,
        0,0,0,0,0,0,0,
    ]
    tehai = [0, 2, 0, 1, 1, 1, 2, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]

    shanten_noraml, _, shanten_chitoitsu = shanten_analysis.calc_all_shanten(tehai, 0)
    # result = dfs.dfs_with_score_chitoitsu(tehai, [], 3, shanten_chitoitsu)
    # print(len(result))
    # print(result)
    for i in range(1):
        result = dfs.dfs_with_score_normal(tehai, [], 2, shanten_noraml)
    print(len(result))
    for r in result:
        print(r)


if __name__ == "__main__":
    main()
