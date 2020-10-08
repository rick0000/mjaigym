import itertools
from collections import deque
from enum import Enum
import copy

class Syu(Enum):
    Manzu=0
    Pinzu=1
    Souzu=2
    Ji=3

apply_cache = {}

def create_change(depth, change_sum):
    m_sub = range(4)
    m_add = range(4)
    result = deque()
    add_ranges = list(range(depth+1))
    minus_ranges = list(range(-(depth),1))
    for (m_sub, m_add) in itertools.product(minus_ranges, add_ranges):
        for (p_sub, p_add) in itertools.product(minus_ranges, add_ranges):
            for (s_sub, s_add) in itertools.product(minus_ranges, add_ranges):
                for (j_sub, j_add) in itertools.product(minus_ranges, add_ranges):
                    if not (m_sub + m_add + p_sub + p_add + s_sub + s_add + j_sub + j_add) == change_sum:
                        continue
                    if (abs(m_sub) + m_add + abs(p_sub) + p_add + abs(s_sub) + s_add + abs(j_sub) + j_add) > depth * 2 - change_sum:
                        continue
                    result.append(
                        ((m_sub,m_add), (p_sub, p_add), (s_sub, s_add), (j_sub, j_add))
                    )

    return result


def normal_search(tehai, depth):
    manzu = tehai[0:9]
    pinzu = tehai[9:18]
    souzu = tehai[18:27]
    ji = tehai[27:34]

    # ありえる全changeパターン列挙。
    changes = create_change(depth, 0)


    # マイナスの場合→Noneパターン
    # 移動0枚の場合→0パターン
    # マイナスが同じでプラスが3枚増えた場合→元のパターン　*（34+27）パターン

    manzu_sum = sum(manzu)
    pinzu_sum = sum(pinzu)
    souzu_sum = sum(souzu)
    ji_sum = sum(ji)

    datas = {
        Syu.Manzu:{"num":manzu,"sum":manzu_sum},
        Syu.Pinzu:{"num":pinzu,"sum":pinzu_sum},
        Syu.Souzu:{"num":souzu,"sum":souzu_sum},
        Syu.Ji:{"num":ji,"sum":ji_sum},
    }
    mpsz_combinations = set()
    changed_buffer_with_head = {Syu.Manzu:{}, Syu.Pinzu:{}, Syu.Souzu:{}, Syu.Ji:{}}
    changed_buffer_nohead = {Syu.Manzu:{}, Syu.Pinzu:{}, Syu.Souzu:{}, Syu.Ji:{}}
        
    for change in changes:
        before_len = len(mpsz_combinations)
        for mpsz in [Syu.Manzu, Syu.Pinzu, Syu.Souzu, Syu.Ji]:
            extract(
                mpsz_combinations,
                changed_buffer_with_head,
                changed_buffer_nohead,
                change,
                datas,
                mpsz,
            )
        print(change, f"{before_len}->{len(mpsz_combinations)}")
    
    return sorted(mpsz_combinations)

    # (-3,0)
    
    # (-3,1)
    # (-2,0)

    # (-3,2)
    # (-2,1)
    # (-1,0)

    # (-3,3) *
    # (-2,2)
    # (-1,1)
    # (0,0)

    # (-2,3) *
    # (-1,2) *
    # (0,1) *

    # changeを適用。
    # マイナス3のものがある場合は*(34+27)できる。


def extract(
        mpsz_combinations, 
        changed_buffer_with_head, 
        changed_buffer_nohead,
        change,
        datas,
        target
    ):

    head_candidate = None
    mentsu_candidates = []
    
    for mpsj in [Syu.Manzu, Syu.Pinzu, Syu.Souzu, Syu.Ji]:
        num = datas[mpsj]["num"]
        num_sum = datas[mpsj]["sum"]
        syu_change = change[mpsj.value]

        if mpsj==Syu.Ji:
            num_key = get_key_ji(num)
        else:
            num_key = get_key(num)


        
        if num_key not in changed_buffer_with_head[mpsj]:
            changed_buffer_with_head[mpsj][num_key] = {}
        if num_key not in changed_buffer_nohead:
            changed_buffer_nohead[mpsj][num_key] = {}

        if target == mpsj:
            # use -3 cache
            if syu_change not in changed_buffer_with_head[mpsj][num_key]:
                # create cache
                if mpsj == Syu.Ji:
                    head_combinations = get_with_head_combination_ji(num, num_sum, syu_change, mpsj)
                else:
                    head_combinations = get_with_head_combination(num, num_sum, syu_change, mpsj)
                changed_buffer_with_head[mpsj][num_key][syu_change] = head_combinations
            head_candidate = changed_buffer_with_head[mpsj][num_key][syu_change]
        else:
            if syu_change not in changed_buffer_nohead[mpsj][num_key]:
                # create cache
                if mpsj == Syu.Ji:
                    combination = get_combination_ji(num, num_sum, syu_change, mpsj)
                else:
                    combination = get_combination(num, num_sum, syu_change, mpsj)
                changed_buffer_nohead[mpsj][num_key][syu_change] = combination
            mentsu_candidates.append(changed_buffer_nohead[mpsj][num_key][syu_change])
            
    
    if head_candidate is None:
        return
    
    if any([c is None for c in mentsu_candidates]):
        return

    change_dist_sum = sum([sum([abs(sc) for sc in syu_change]) for syu_change in change])

    for head in head_candidate:
        head_type_candidates = head_candidate[head]
        
        targets = []
        if len(head_type_candidates) > 0:
            targets.append(head_type_candidates)
        for i in range(len(mentsu_candidates)):
            if len(mentsu_candidates[i]) > 0:
                targets.append(mentsu_candidates[i])
        
        for c in itertools.product(*targets):
            flatten = tuple(sorted(itertools.chain.from_iterable(c)))
            mpsz_combinations.add(((head, flatten), change_dist_sum))



def get_combination(tehai, tehai_sum, change, pai_syu):
    return _get_combination(tehai, tehai_sum, change, cut_mentsu, pai_syu)

def get_combination_ji(tehai, tehai_sum, change, pai_syu):
    return _get_combination(tehai, tehai_sum, change, cut_mentsu_ji, pai_syu)

def _get_combination(tehai, tehai_sum, change, cut_func, pai_syu):
    if (tehai_sum + sum(change)) % 3 != 0:
        return None
    if (tehai_sum + sum(change)) == 0:
        return []

    mentsu_num = (tehai_sum + sum(change)) // 3
    
    result = []
    changed_tehais = apply_change(tehai, change)
    # print(f"pai_syu:{pai_syu}, tehai:{tehai}, change:{change}, changed_tehais len:{len(changed_tehais)}")
    for changed_tehai in changed_tehais:
        combinations = cut_func(changed_tehai, mentsu_num, pai_syu)
        result.extend(combinations)
    if len(result) == 0:
        return None
    else:
        return result


def get_with_head_combination(tehai, tehai_sum, change, pai_syu):
    return _get_with_head_combination(tehai, tehai_sum, change, cut_mentsu, pai_syu)

def get_with_head_combination_ji(tehai, tehai_sum, change, pai_syu):
    return _get_with_head_combination(tehai, tehai_sum, change, cut_mentsu_ji, pai_syu)

def _get_with_head_combination(tehai, tehai_sum, change, cut_func, pai_syu):
    if (tehai_sum + sum(change)) % 3 != 2:
        return None
    
    mentsu_num = ((tehai_sum + sum(change)) - 2) // 3
    result = {}

    if mentsu_num == 0:
        # search only head pattern
        for head in range(len(tehai)):
            result_head = head + 9*pai_syu.value
            if tehai[head] + change[1] == 2:
                result[result_head] = []
        return result

    for head in range(len(tehai)):
        result_head = head + 9*pai_syu.value

        if tehai[head] >= 2 and tehai[head] + change[1] <= 4:
            tehai[head] -= 2
            changed_tehais = apply_change(tehai, change)
            # print(f"head:{head}, change:{change}, changed_tehais len:{len(changed_tehais)}")
            for changed_tehai in changed_tehais:
                combinations = cut_func(changed_tehai, mentsu_num, pai_syu)
                
                for combination in combinations:
                    same_head_num = sum([len([m for m in mentsu if m == result_head]) for mentsu in combination])
                    if same_head_num <= 2:
                        if result_head not in result:
                            result[result_head] = []
                        result[result_head].append(combination)
            tehai[head] += 2
        
        if tehai[head] >= 1 and change[1] >= 1 and  tehai[head] + change[1] <= 4:
            new_change = (change[0], change[1]-1)
            tehai[head] -= 1
            changed_tehais = apply_change(tehai, new_change)
            for changed_tehai in changed_tehais:
                combinations = cut_func(changed_tehai, mentsu_num, pai_syu)
                for combination in combinations:
                    same_head_num = sum([len([m for m in mentsu if m == result_head]) for mentsu in combination])
                    if same_head_num <= 2:
                        if result_head not in result:
                            result[result_head] = []
                        result[result_head].append(combination)
            tehai[head] += 1
        
        if tehai[head] == 0 and change[1] >= 2 and tehai[head] + change[1] <= 4:
            new_change = (change[0], change[1]-2)
            changed_tehais = apply_change(tehai, new_change)
            for changed_tehai in changed_tehais:
                combinations = cut_func(changed_tehai, mentsu_num, pai_syu)
                for combination in combinations:
                    same_head_num = sum([len([m for m in mentsu if m == result_head]) for mentsu in combination])
                    if same_head_num <= 2:
                        if result_head not in result:
                            result[result_head] = []
                        result[result_head].append(combination)
            
            
    return result

def apply_change(tehai, change):
    global apply_cache
    key = (tuple(tehai), change)
    if key not in apply_cache:
        result = _apply_change(tehai, change)
        apply_cache[key] = result
        return result
    else:
        return apply_cache[key]

def _apply_change(tehai, change):
    
    if change == (0,0):
        return [tehai]
    if len(tehai) == 7:
        get_key_func = get_key_ji
    else:
        get_key_func = get_key
    result = {}
    queue = deque()
    queue.append((tehai, change))
    while len(queue) > 0:
        record_tehai, rest_change = queue.pop()

        if rest_change == (0,0):
            key = get_key_func(record_tehai)
            result[key] = record_tehai

        if abs(rest_change[0]) > 0:
            new_rest_change = (rest_change[0]+1, rest_change[1])
            for i, num in enumerate(record_tehai):
                if num == 0:
                    continue
                record_tehai[i] -= 1
                queue.append((copy.copy(record_tehai), new_rest_change))
                record_tehai[i] += 1

        elif rest_change[1] > 0:
            new_rest_change = (rest_change[0], rest_change[1]-1)
            for i, num in enumerate(record_tehai):
                if num == 4:
                    continue
                record_tehai[i] += 1
                queue.append((copy.copy(record_tehai), new_rest_change))
                record_tehai[i] -= 1
    
    return list(result.values())

def cut_mentsu(tehai, mentsu_num, pai_syu):
    result = []
    queue = deque()
    history = []
    start_index = 0
    current_depth = 0
    queue.append((tehai, current_depth, history, start_index))
    
    while len(queue):
        (record_tehai, current_depth, history, start_index) = queue.pop()

        if len(history) > 0 and current_depth == mentsu_num and mentsu_num == len(history):
            history_sorted = sorted(history)
            if history_sorted not in result:
                result.append(history_sorted)

        # cut syuntsu
        for i in range(start_index, 7):
            if record_tehai[i] >= 1\
                and record_tehai[i+1] >= 1\
                and record_tehai[i+2] >= 1:
                
                record_tehai[i] -= 1
                record_tehai[i+1] -= 1
                record_tehai[i+2] -= 1
                history.append(tuple([p + 9*pai_syu.value for p in (i,i+1,i+2)]))
                queue.append((copy.copy(record_tehai), current_depth+1, copy.copy(history), i))
                history.pop()
                record_tehai[i] += 1
                record_tehai[i+1] += 1
                record_tehai[i+2] += 1
        
        # kotsu
        for i in range(start_index, 9):
            if record_tehai[i] >= 3:
                record_tehai[i] -= 3
                history.append(tuple([p + 9*pai_syu.value for p in (i,i,i)]))
                queue.append((copy.copy(record_tehai), current_depth+1, copy.copy(history), i))
                history.pop()
                record_tehai[i] += 3
    return result

def cut_mentsu_ji(tehai, mentsu_num, pai_syu):
    result = []
    queue = deque()
    history = []
    start_index = 0
    current_depth = 0
    queue.append((tehai, current_depth, history, start_index))
    
    while len(queue):
        (record_tehai, current_depth, history, start_index) = queue.pop()

        if len(history) > 0 and current_depth == mentsu_num and mentsu_num == len(history):
            result.append(history)
        
        # kotsu
        for i in range(start_index, 7):
            if record_tehai[i] >= 3:
                record_tehai[i] -= 3
                history.append(tuple([p + 9*pai_syu.value for p in (i,i,i)]))
                queue.append((copy.copy(record_tehai), current_depth+1, copy.copy(history), i))
                history.pop()
                record_tehai[i] += 3
    return result


def get_key(tehai):
    return tuple(tehai)
    return (tehai[0] << 24)\
        + (tehai[1] << 21)\
        + (tehai[2] << 18)\
        + (tehai[3] << 15)\
        + (tehai[4] << 12)\
        + (tehai[5] << 9)\
        + (tehai[6] << 6)\
        + (tehai[7] << 3)\
        + tehai[8] 
def get_key_ji(tehai):
    return tuple(tehai)
    return (tehai[0] << 24)\
        + (tehai[1] << 21)\
        + (tehai[2] << 18)\
        + (tehai[3] << 15)\
        + (tehai[4] << 12)\
        + (tehai[5] << 9)\
        + (tehai[6] << 6)

def main():
    tehai = [
        3, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 1, 0, 1, 2, 0, 0, 0,
        0, 0, 0, 1, 2, 0, 0, 1, 0,
        2, 0, 0, 0, 0, 0, 0
    ]
    depth = 3
    for i in range(10):
        result = normal_search(tehai, depth)
    # print(len(result))

if __name__ == "__main__":
    main()

