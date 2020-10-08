import os
from collections import deque, namedtuple
import itertools
import copy
import joblib
from typing import List
from enum import Enum

from mjaigym.board.function.rs_shanten_analysis import RsShantenAnalysis
from mjaigym.board.function.hora import Hora, Candidate
from mjaigym import shanten
from mjaigym.board.function.pai import Pai
from mjaigym.board.function.furo import Furo
from mjaigym.board.function.dfs_result import DfsResult, DfsResultType
from mjaigym.board.function.hora_rs import HoraRs

CHANGE_CACHE = "change_cache.pkl"

KokushiHora = namedtuple("KokushiHora", "fu fan yakus points oya_payment ko_payment")

class Syu(Enum):
    Manzu=0
    Pinzu=1
    Souzu=2
    Ji=3

class Dfs():
    def __init__(self):
        self.changed_buffer_with_head = {Syu.Manzu:{}, Syu.Pinzu:{}, Syu.Souzu:{}, Syu.Ji:{}}
        self.changed_buffer_nohead = {Syu.Manzu:{}, Syu.Pinzu:{}, Syu.Souzu:{}, Syu.Ji:{}}
        self.hora_cash = {}
        self.apply_change_cache = {}
        self.cut_mentsu_cache = {}

    def create_change(self, depth, mode):
        if mode % 3 == 2:
            return self._create_change14(depth)
        elif mode % 3 == 1:
            # return self._create_change13(depth)
            raise Exception("not intended path")


    # def _create_change13(self, depth):
    #     cache_file_name = f"cache/{depth}_13_{CHANGE_CACHE}"
    #     try:
    #         return joblib.load(cache_file_name)
    #         pass
    #     except:
    #         pass
        
    #     result = self._create_change(depth, change_sum=1)

    #     try:
    #         os.makedirs(os.path.dirname(cache_file_name), exist_ok=True)
    #         joblib.dump(result, cache_file_name)
    #     except:
    #         print("fail to save change cache")
    #     return result

        
    def _create_change14(self, depth):
        cache_file_name = f"cache/{depth}_14_{CHANGE_CACHE}"
        try:            
            return joblib.load(cache_file_name)
            pass
        except:
            pass

        result = self._create_change(depth)

        try:
            os.makedirs(os.path.dirname(cache_file_name), exist_ok=True)
            joblib.dump(result, cache_file_name)
        except:
            print("fail to save change cache")

        return result

    def _create_change(self, depth):
        m_sub = range(4)
        m_add = range(4)
        result = deque()
        add_ranges = list(range(depth+1))
        minus_ranges = list(range(-(depth),1))
        for (m_sub, m_add) in itertools.product(minus_ranges, add_ranges):
            for (p_sub, p_add) in itertools.product(minus_ranges, add_ranges):
                for (s_sub, s_add) in itertools.product(minus_ranges, add_ranges):
                    for (j_sub, j_add) in itertools.product(minus_ranges, add_ranges):
                        items = [m_sub, m_add, p_sub, p_add, s_sub, s_add, j_sub, j_add]
                        
                        change_dist_sum = sum([abs(sc) for sc in items])
                        if change_dist_sum > depth*2:
                            continue
                        minus_sum_abs = sum([abs(sc) for sc in items if sc<0])
                        if minus_sum_abs > depth:
                            continue
                        plus_sum_abs = sum([abs(sc) for sc in items if sc>0])
                        if plus_sum_abs > depth:
                            continue

                        result.append(
                            ((m_sub, m_add), (p_sub, p_add), (s_sub, s_add), (j_sub, j_add))
                        )

        return result


    def dfs_chitoitsu(self, tehai, depth, doras, chitoitsu_shanten):
        """returns max point dfs patten, not all pattern, by considering doras.
        """
        tehai_sum = sum(tehai)
        
        if tehai_sum == 13:
            depth_offset = 1
        elif tehai_sum == 14:
            depth_offset = 0
        else:
            return []

        results = set()
        for target_depth in range(depth+1):
            target_depth_dfs_result = self._dfs_chitoitsu(tehai, target_depth, doras, chitoitsu_shanten)
            
            if target_depth_dfs_result is not None:
                results.add(target_depth_dfs_result)
        return results


    def _dfs_chitoitsu(self, tehai, depth, doras, chitoitsu_shanten):
        if chitoitsu_shanten - depth > -1:
            return None

        new_toitsu_ids = []

        toitsu_ids = []
        addable_dora_toitsu_candidates = []
        addable_dora_tanki_candidates = []
        addable_no_dora_tanki_candidates = []
        addable_no_dora_toitsu_candidates = []
        no_dora_toitsu_ids = []
        dora_ids = [d.id for d in doras]
        
        for i,num in enumerate(tehai):
            if num >= 2:
                toitsu_ids.append(i)
                if i not in dora_ids:
                    no_dora_toitsu_ids.append(i)
            elif num == 1:
                if i in dora_ids:
                    addable_dora_tanki_candidates.append(i)
                else:
                    addable_no_dora_tanki_candidates.append(i)
            elif num == 0:
                if i in dora_ids:
                    addable_dora_toitsu_candidates.append(i)
                else:
                    addable_no_dora_toitsu_candidates.append(i)
        
        new_toitsu_num = (chitoitsu_shanten+1) - (7-len(toitsu_ids))
        use_tanki_num = 7-len(toitsu_ids)-new_toitsu_num
        # print("new_toitsu_num,use_tanki_num",new_toitsu_num,use_tanki_num)

        rest_depth = depth

        # add up to 7 toitsu
        while new_toitsu_num > 0\
                and len(toitsu_ids) + len(new_toitsu_ids) < 7\
                and len(addable_dora_toitsu_candidates) > 0\
                and rest_depth >= 2:
            dora_toitsu_id = addable_dora_toitsu_candidates.pop()
            new_toitsu_ids.append(dora_toitsu_id)
            # print("dora_toitsu_id",dora_toitsu_id)
            rest_depth -= 2

        while len(toitsu_ids) + len(new_toitsu_ids) < 7\
                and len(addable_dora_tanki_candidates) > 0\
                and rest_depth >= 1:
            dora_tanki_id = addable_dora_tanki_candidates.pop()
            new_toitsu_ids.append(dora_tanki_id)
            # print("dora_tanki_id",dora_tanki_id)
            rest_depth -= 1
        
        while len(toitsu_ids) + len(new_toitsu_ids) < 7\
                and len(addable_no_dora_tanki_candidates) > 0\
                and rest_depth >= 1:
            nodora_tanki = addable_no_dora_tanki_candidates.pop()
            new_toitsu_ids.append(nodora_tanki)
            # print("nodora_tanki", nodora_tanki)
            rest_depth -= 1
        
        while len(toitsu_ids) + len(new_toitsu_ids) < 7\
                and len(addable_no_dora_toitsu_candidates) > 0\
                and rest_depth >= 2:
            nodora_toitsu = addable_no_dora_toitsu_candidates.pop()
            new_toitsu_ids.append(nodora_toitsu)
            # print("nodora_toitsu", nodora_toitsu)
            rest_depth -= 2

        # change already have toitsu
        while len(addable_dora_toitsu_candidates) > 0\
                and len(no_dora_toitsu_ids) > 0\
                and rest_depth >= 2:
            toitsu_ids.remove(no_dora_toitsu_ids.pop())
            new_toitsu_ids.append(addable_dora_toitsu_candidates.pop())
            rest_depth -= 2

        while len(addable_dora_tanki_candidates) > 0\
                and len(no_dora_toitsu_ids) > 0\
                and rest_depth >= 1:
            toitsu_ids.remove(no_dora_toitsu_ids.pop())
            new_toitsu_ids.append(addable_dora_tanki_candidates.pop())
            rest_depth -= 1
    
        max_combination = []
        for toitsu_id in toitsu_ids:
            max_combination.append((toitsu_id, toitsu_id))

        for new_toitsu_id in new_toitsu_ids:
            max_combination.append((new_toitsu_id, new_toitsu_id))
        
        # if len(max_combination) < 7:
        #     print(max_combination, rest_depth, depth)

        return tuple(sorted(max_combination))

    

    def dfs_normal(self, tehai, furo_num, depth, current_shanten):
        """returns all dfs patten
        """
        mpsz_combinations = set()
        
        if current_shanten - depth > -1:
            # print("finish because current_shanten - depth > 1.", current_shanten,  depth)
            return mpsz_combinations
        
        changes = self.create_change(depth, sum(tehai))

        # for change in changes:
        #     c = list(itertools.chain.from_iterable(change))
            
        #     assert sum([d for d in c if d > 0]) <= depth, print(c)
        #     assert sum([abs(d) for d in c if d < 0]) <= depth, print(c)
        #     assert sum([abs(d) for d in c]) <= depth*2, print(c)
        
        # print("changes",len(changes))
        manzu = tehai[0:9]
        pinzu = tehai[9:18]
        souzu = tehai[18:27]
        ji = tehai[27:34]
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
        
        for change in changes:
            for mpsz in [Syu.Manzu, Syu.Pinzu, Syu.Souzu, Syu.Ji]:
                self.extract(
                    tehai, # debug
                    mpsz_combinations,
                    self.changed_buffer_with_head,
                    self.changed_buffer_nohead,
                    change,
                    datas,
                    mpsz,
                )
        
        return sorted(mpsz_combinations)



    def extract(
            self, 
            tehai, # debug
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
            
            # num_key = tuple(num)
            
            # if num_key not in changed_buffer_with_head[mpsj]:
            #     changed_buffer_with_head[mpsj][num_key] = {}
            # if num_key not in changed_buffer_nohead:
            #     changed_buffer_nohead[mpsj][num_key] = {}

            if target == mpsj:
                if mpsj == Syu.Ji:
                    head_candidate = self.get_with_head_combination_ji(num, num_sum, syu_change, mpsj)
                else:
                    head_candidate = self.get_with_head_combination(num, num_sum, syu_change, mpsj)
                
                # if head_candidate is not None:
                #     print(num, change, head_candidate)
            else:
                if mpsj == Syu.Ji:
                    combination = self.get_combination_ji(num, num_sum, syu_change, mpsj)
                else:
                    combination = self.get_combination(num, num_sum, syu_change, mpsj)
                mentsu_candidates.append(combination)
                 
        # assert len(mentsu_candidates) == 3
        
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


                changed_tehai_num = [0] * 34
                changed_tehai_num[head] += 2
                for mentsu in flatten:
                    for pai in mentsu:
                        changed_tehai_num[pai] += 1

                # diff = [changed_tehai_num[i] - num for i,num in enumerate(tehai)]
                # assert sum([d for d in diff if d>0]) <= 2, f"{head},{tehai}, {changed_tehai_num}, {diff}, {change}"






    def get_combination(self, tehai, tehai_sum, change, pai_syu):
        return self._get_combination(tehai, tehai_sum, change, self.cut_mentsu, pai_syu)

    def get_combination_ji(self, tehai, tehai_sum, change, pai_syu):
        return self._get_combination(tehai, tehai_sum, change, self.cut_mentsu_ji, pai_syu)

    def _get_combination(self, tehai, tehai_sum, change, cut_func, pai_syu):
        if (tehai_sum + sum(change)) % 3 != 0:
            return None
        if (tehai_sum + sum(change)) == 0:
            return []

        mentsu_num = (tehai_sum + sum(change)) // 3
        
        result = []
        
        changed_tehais = self.apply_change(tehai, change)
        
        # print(f"pai_syu:{pai_syu}, tehai:{tehai}, change:{change}, changed_tehais len:{len(changed_tehais)}")
        for changed_tehai in changed_tehais:
            combinations = cut_func(changed_tehai, mentsu_num, pai_syu)
            result.extend(combinations)
        if len(result) == 0:
            return None
        else:
            return result


    def get_with_head_combination(self, tehai, tehai_sum, change, pai_syu):
        return self._get_with_head_combination(tehai, tehai_sum, change, self.cut_mentsu, pai_syu)

    def get_with_head_combination_ji(self, tehai, tehai_sum, change, pai_syu):
        return self._get_with_head_combination(tehai, tehai_sum, change, self.cut_mentsu_ji, pai_syu)

    def _get_with_head_combination(self, tehai, tehai_sum, change, cut_func, pai_syu):
        if (tehai_sum + sum(change)) % 3 != 2:
            return None
        
        mentsu_num = ((tehai_sum + sum(change)) - 2) // 3
        result = {}

        if mentsu_num == 0 and tehai_sum + sum(change) == 2:
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
                changed_tehais = self.apply_change(tehai, change)
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
            
            if tehai[head] >= 1 and change[1] >= 1 and tehai[head] + change[1] <= 4:
                new_change = (change[0], change[1]-1)
                tehai[head] -= 1
                changed_tehais = self.apply_change(tehai, new_change)
                for changed_tehai in changed_tehais:
                    combinations = cut_func(changed_tehai, mentsu_num, pai_syu)
                    # print(new_change, changed_tehais, combinations)
                    for combination in combinations:
                        same_head_num = sum([len([m for m in mentsu if m == result_head]) for mentsu in combination])
                        if same_head_num <= 2:
                            if result_head not in result:
                                result[result_head] = []
                            result[result_head].append(combination)
                tehai[head] += 1
                
            
            if tehai[head] == 0 and change[1] >= 2 and tehai[head] + change[1] <= 4:
                new_change = (change[0], change[1]-2)
                changed_tehais = self.apply_change(tehai, new_change)
                for changed_tehai in changed_tehais:
                    combinations = cut_func(changed_tehai, mentsu_num, pai_syu)
                    for combination in combinations:
                        same_head_num = sum([len([m for m in mentsu if m == result_head]) for mentsu in combination])
                        if same_head_num <= 2:
                            if result_head not in result:
                                result[result_head] = []
                            result[result_head].append(combination)
                
                
        return result

    def apply_change(self, tehai, change):
        key = (tuple(tehai), change)
        if key not in self.apply_change_cache:
            result = self._apply_change(tehai, change)
            self.apply_change_cache[key] = result
            return result
        else:
            return self.apply_change_cache[key]

    def _apply_change(self, tehai, change):
        if change == (0,0):
            return [list(tehai)]
        if len(tehai) == 7:
            get_key_func = self.get_key_ji
        else:
            get_key_func = self.get_key
        result = set()
        queue = deque()
        queue.append((tehai, change))
        while len(queue) > 0:
            record_tehai, rest_change = queue.pop()

            if rest_change == (0,0):
                # key = get_key_func(record_tehai)
                result.add(tuple(record_tehai))

            if rest_change[0] < 0:
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
        
        return list([list(r) for r in result])



    def get_key(self, tehai):
        return (tehai[0] << 24)\
            + (tehai[1] << 21)\
            + (tehai[2] << 18)\
            + (tehai[3] << 15)\
            + (tehai[4] << 12)\
            + (tehai[5] << 9)\
            + (tehai[6] << 6)\
            + (tehai[7] << 3)\
            + tehai[8] 
    def get_key_ji(self, tehai):
        return (tehai[0] << 24)\
            + (tehai[1] << 21)\
            + (tehai[2] << 18)\
            + (tehai[3] << 15)\
            + (tehai[4] << 12)\
            + (tehai[5] << 9)\
            + (tehai[6] << 6)

    def cut_mentsu(self, tehai, mentsu_num, pai_syu):
        key = (tuple(tehai), mentsu_num, pai_syu)
        if key not in self.cut_mentsu_cache:
            self.cut_mentsu_cache[key] = self._cut_mentsu(tehai, mentsu_num, pai_syu)
        return self.cut_mentsu_cache[key]

    def _cut_mentsu(self, tehai, mentsu_num, pai_syu):
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

    def cut_mentsu_ji(self, tehai, mentsu_num, pai_syu):
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

    def dfs_with_score_kokushi(
        self,
        tehai:List[int],
        furos:List[Furo],
        depth:int,
        oya:bool,
        shanten_kokushi:int,
    ):
        if len(furos) > 0:
            return []
        
        if shanten_kokushi - depth > -1:
            return []
        
        
        if oya:
            hora = KokushiHora(
                fu=30,
                fan=100,
                yakus=[["kokushimuso",100]],
                points=48000,
                oya_payment=0,
                ko_payment=16000,
            )
        else:
            hora = KokushiHora(
                fu=30,
                fan=100,
                yakus=[["kokushimuso",100]],
                points=32000,
                oya_payment=16000,
                ko_payment=8000,
            )

        kokushi_tenpai_nums = [
            1,0,0,0,0,0,0,0,1,
            1,0,0,0,0,0,0,0,1,
            1,0,0,0,0,0,0,0,1,
            1,1,1,1,1,1,1,
        ]
        diff = [kokushi_tenpai_nums[i]-tehai[i] for i in range(34)]

        return [DfsResult(DfsResultType.Kokushimuso, kokushi_tenpai_nums, hora, diff)]

    def dfs_with_score_chitoitsu(
        self,
        tehai:List[int],
        furos:List[Furo],
        depth:int, 
        shanten_chitoitsu:int,
        oya:bool=False, 
        bakaze:str="E", 
        jikaze:str="S", 
        doras:List[Pai]=None, 
        uradoras:List[Pai]=None,
        num_akadoras:int=0,
    ):
        if doras is None:
            doras = []
        if uradoras is None:
            uradoras = []
        
        horas = []
        
        results = self.dfs_chitoitsu(tehai, depth, doras, shanten_chitoitsu)
        
        
        for result in results:
            tehais = []
            toitsus = result
            
            for i, toitsu in enumerate(toitsus):
                tehais.append(Pai.from_id(toitsu[0]))
                tehais.append(Pai.from_id(toitsu[1]))


            if len(tehais) > 0:
                taken = tehais.pop()
            else:
                taken = None
            
            # print(horra_tehai, taken, taken_id)
            """
            hora = Candidate.from_already_splited_chitoitsu(
                tehais=tehais,
                furos=furos, # List[Furo]
                taken=taken,
                oya=oya,
                bakaze=bakaze,
                jikaze=jikaze,
                doras=doras,
                uradoras=uradoras,
                num_akadoras=num_akadoras,
                )
            """
            hora = HoraRs(
                tehais=tehais,
                furos=furos,
                taken=taken,
                hora_type='tsumo',
                oya=oya,
                bakaze=bakaze,
                jikaze=jikaze,
                doras=doras,
                uradoras=uradoras,
                reach=len(furos)==0,
                double_reach=False,
                ippatsu=False,
                rinshan=False,
                haitei=False,
                first_turn=False,
                chankan=False,
                num_akadoras=num_akadoras
            )
            nums = [0] * 34
            nums[taken.id] += 1
            for t in tehais:
                nums[t.id] += 1
            diff = [nums[i] - tehai[i] for i in range(34)]
            horas.append(DfsResult(DfsResultType.Chitoitsu, toitsus, hora, diff))
            
        return horas

    def dfs_with_score_normal(
        self,
        tehai:List[int], 
        furos:List[Furo], 
        depth:int, 
        shanten_normal:int,
        oya:bool=False,
        bakaze:str="E", 
        jikaze:str="S", 
        doras:List[Pai]=None, 
        uradoras:List[Pai]=None,
        num_akadoras:int=0,
    ):
        if doras is None:
            doras = []
        if uradoras is None:
            uradoras = []
        
        # print(tehai, len(furos), depth, shanten_normal)
        results = self.dfs_normal(tehai, len(furos), depth, shanten_normal)
        

        # union to min distance
        unioned_results = {}
        
        for result in results:
            result_tehai = result[0]
            distance = result[1]
            # if distance >= 5:
            #     print("distance",distance)
            
            if result_tehai in unioned_results:
                unioned_results[result_tehai] = min(distance, unioned_results[result_tehai])
            else:
                unioned_results[result_tehai] = distance
        
        horas = []
        dahai_horas = {}
        for result_tehai, distance in unioned_results.items():
            (head, mentsus) = result_tehai
            # try to use cache
            changed_tehai_num = [0] * 34
            changed_tehai_num[head] += 2
            for mentsu in mentsus:
                for pai in mentsu:
                    changed_tehai_num[pai] += 1

            diff = [changed_tehai_num[i] - num for i,num in enumerate(tehai)]
            # assert sum([d for d in diff if d>0]) <= depth, f"{tehai}, {result_tehai}, {diff}, {distance}"

            taken_candidate_ids = [i for (i,p) in enumerate(diff) if p > 0]
            hora_key = (
                result_tehai, 
                tuple(sorted(furos)),
                oya,
                bakaze,
                jikaze,
                tuple(sorted(doras)),
                tuple(sorted(uradoras)),
                num_akadoras,
                tuple(sorted(taken_candidate_ids)),
                )
            if hora_key in self.hora_cash:
                (max_result, diff) = self.hora_cash[hora_key]
                horas.append(DfsResult(DfsResultType.Normal, result_tehai, max_result, diff))
                # print("hora cache found")
                continue

            if len(taken_candidate_ids) == 0:
                taken_candidate_ids = [i for (i,p) in enumerate(changed_tehai_num) if p > 0]
            
            changed_tehais = []
            for i, value in enumerate(changed_tehai_num):
                for _ in range(value):
                    changed_tehais.append(Pai.from_id(i))
            
            
            

            taken_changed_results = []
            for taken_id in taken_candidate_ids:
                horra_tehai = copy.copy(changed_tehais)
                taken_index = horra_tehai.index(Pai.from_id(taken_id))
                taken = horra_tehai.pop(taken_index)
                
                """
                # print(horra_tehai, taken, taken_id)
                hora = Candidate.from_already_spliteds(
                    head=head, # 9
                    mentsus=mentsus, # ((1,2,3), (4,5,6), (20,20,20), (32,32,32))
                    furos=furos, # List[Furo]
                    taken=taken_id, # 20
                    oya=oya,
                    bakaze=bakaze,
                    jikaze=jikaze,
                    doras=doras,
                    uradoras=uradoras,
                    num_akadoras=num_akadoras,
                    )
                """
                
                hora = HoraRs(
                    tehais=horra_tehai,
                    furos=furos,
                    taken=taken,
                    hora_type='tsumo',
                    oya=oya,
                    bakaze=bakaze,
                    jikaze=jikaze,
                    doras=doras,
                    uradoras=uradoras,
                    reach=len(furos)==0,
                    double_reach=False,
                    ippatsu=False,
                    rinshan=False,
                    haitei=False,
                    first_turn=False,
                    chankan=False,
                    num_akadoras=num_akadoras
                )
                
                # print(hora)
                taken_changed_results.append(hora)
        
            # max_result = max(taken_changed_results, key= lambda x:{x["points"]*1000+x["fan"]*1000+x["fu"]})    
            max_result = max(taken_changed_results, key= lambda x:{x.points*1000+x.fan*1000+x.fu})    
            self.hora_cash[hora_key] = (max_result, diff)
            horas.append(DfsResult(DfsResultType.Normal, result_tehai, max_result, diff))
            
            # print(result_tehai, max_result)

        return horas


def compare():
    import datetime

    depth = 3

    dfs = Dfs()
    print("start dfs a")
    start = datetime.datetime.now()

    manzu = [0,0,0,0,0,0,0,0,0]
    pinzu = [0,0,0,0,0,0,0,0,0]
    souzu = [0,0,0,0,0,0,0,0,0]
    ji = [1,0,0,0,0,3,1]
    tehai_a = manzu + pinzu + souzu + ji
    
    depth = 3
    furo_num = (14-sum(tehai_a))//3

    from mjaigym.board.function.furo import Furo
    w_pai = Pai.from_str("W")
    pai_3m =Pai.from_str("3m")
    furos = [
        Furo({"type":"pon", "actor":0, "target":2, "taken":w_pai, "consumed":[w_pai,w_pai]}),
        Furo({"type":"pon", "actor":0, "target":2, "taken":w_pai, "consumed":[w_pai,w_pai]}),
        Furo({"type":"pon", "actor":0, "target":2, "taken":pai_3m, "consumed":[pai_3m,pai_3m]}),
    ]
    assert 13 <= sum(tehai_a) + 3 * len(furos) <= 14

    shanten_normal, shanten_kokushi, shanten_chitoitsu = shanten.get_shanten_all(tehai_a, len(furos))
    result_a = dfs.dfs_with_score_normal(tehai_a, furos, depth, shanten_normal=shanten_normal)

    # a_changes = set()
    # for ra in result_a:
    #     a_changes.add(tuple(ra[0]))
    print("start dfs b")
    dfs = Dfs()
    manzu = [0,0,0,0,0,0,0,0,0]
    pinzu = [0,0,0,0,0,0,0,0,0]
    souzu = [0,0,0,0,0,0,0,0,0]
    ji = [1,0,0,0,0,1,3]
    tehai_b = manzu + pinzu + souzu + ji
    shanten_normal, shanten_kokushi, shanten_chitoitsu = shanten.get_shanten_all(tehai_b, len(furos))
    result_b = dfs.dfs_with_score_normal(tehai_b, furos, depth, shanten_normal=shanten_normal)

    result_a = sorted(result_a, key=lambda x:x[0])
    result_b = sorted(result_b, key=lambda x:x[0])

    print(len(result_a))
    print(len(result_b))

    for i in range(max(len(result_a),len(result_b))):
        if i >= len(result_b):
            print(result_a[i][0])
        else:
            print(result_a[i][0], result_b[i][0])

    


def main():
    import datetime
    
    manzu = [1,0,0,1,0,0,0,0,0]
    pinzu = [0,0,0,0,0,2,2,0,0]
    souzu = [0,0,0,2,0,0,2,0,0]
    ji = [0,3,0,0,0,0,0]
    """
    manzu = [1,0,0,0,0,0,0,0,1]
    pinzu = [1,0,0,0,0,0,0,0,1]
    souzu = [1,0,3,0,0,0,0,0,1]
    ji = [1,1,1,1,0,0,1]
    """
    tehai = manzu + pinzu + souzu + ji
    
    tehai = [
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 3, 0, 0, 0,
        0, 0, 0, 3, 2, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0
    ]
    
    
    depth = 3
    furo_num = (14-sum(tehai))//3

    from mjaigym.board.function.furo import Furo
    w_pai = Pai.from_str("W")
    pai_3m =Pai.from_str("3m")
    furos = [
        Furo({"type":"pon", "actor":0, "target":2, "taken":pai_3m, "consumed":[pai_3m,pai_3m]}),
        # Furo({"type":"pon", "actor":0, "target":2, "taken":w_pai, "consumed":[w_pai,w_pai]}),
        # Furo({"type":"pon", "actor":0, "target":2, "taken":pai_3m, "consumed":[pai_3m,pai_3m]}),
    ]
    assert 13 <= sum(tehai) + 3 * len(furos) <= 14

    dfs = Dfs()
    shanten_normal, shanten_kokushi, shanten_chitoitsu = shanten.get_shanten_all(tehai, len(furos))
    start = datetime.datetime.now()
    for i in range(40):
        
        tehai = get_random_tehai()
        # if 0 <= shanten_normal < depth-1:
        result = dfs.dfs_with_score_normal(tehai, furos, depth, oya=True, shanten_normal=shanten_normal)
        # result = dfs.dfs_with_score_chitoitsu(tehai, furos, depth, doras=Pai.from_list(["1m","3m"]), shanten_chitoitsu=shanten_chitoitsu)
        # result = dfs.dfs_with_score_kokushi(tehai, furos, depth, oya=True, shanten_kokushi=shanten_kokushi)
    
    # result = dfs.dfs(tehai, furo_num, depth)
    end = datetime.datetime.now()
    
    print(tehai)
    for r in result:
        print(r)
        pass
    print(len(result))
    print(end - start)

def get_random_tehai():
    alls = [i//4 for i in range(34*4)]
    import random
    nums = [0] * 34
    for t in random.sample(alls, 14):
        nums[t] += 1
    return nums


def debug_head_cut():
    dfs = Dfs()
    tehai = [0, 0, 0, 1, 1, 1, 1, 0, 0]
    tehai_sum = 4
    change = (0, 1)
    pai_syu = Syu.Souzu
    r = dfs._get_with_head_combination(tehai, tehai_sum, change, dfs.cut_mentsu, pai_syu)
    print(r)

    
if __name__ == "__main__":
    debug_head_cut()
    
















