from collections import deque
import itertools
import copy
import joblib
from typing import List

from mjaigym.board.function.rs_shanten_analysis import RsShantenAnalysis
from mjaigym.board.function.hora import Hora, Candidate
from mjaigym import shanten
from mjaigym.board.function.pai import Pai
from mjaigym.board.function.furo import Furo

MANZU=0
PINZU=1
SOUZU=2
JI=3

CHANGE_CACHE = "change_cache.pkl"

class Dfs():
    def __init__(self):
        self.changed_buffer_with_head = {MANZU:{}, PINZU:{}, SOUZU:{}, JI:{}}
        self.changed_buffer_nohead = {MANZU:{}, PINZU:{}, SOUZU:{}, JI:{}}
        self.tehai_change_cache = {}
        self.head_tehai_change_cache = {}
        self.hora_cash = {}

    def create_change(self, depth):
        try:
            return joblib.load(str(depth)+CHANGE_CACHE)
        except:
            pass

        m_sub = range(4)
        m_add = range(4)
        result = deque()
        ranges = list(range(depth+1))
        minus_ranges = list(range(-depth,1))
        for (m_sub, m_add) in itertools.product(minus_ranges, ranges):
            for (p_sub, p_add) in itertools.product(minus_ranges, ranges):
                for (s_sub, s_add) in itertools.product(minus_ranges, ranges):
                    for (j_sub, j_add) in itertools.product(minus_ranges, ranges):
                        if not (m_sub + m_add + p_sub + p_add + s_sub + s_add + j_sub + j_add) == 0:
                            continue
                        if (abs(m_sub) + m_add + abs(p_sub) + p_add + abs(s_sub) + s_add + abs(j_sub) + j_add) > depth * 2:
                            continue
                        result.append(
                            ((m_sub,m_add), (p_sub, p_add), (s_sub, s_add), (j_sub, j_add))
                        )
        try:
            joblib.dump(result, str(depth)+CHANGE_CACHE)
        except:
            print("fail to save change cache")

        return result


    def dfs_chitoitsu(self, tehai, depth, doras, chitoitsu_shanten):
        """returns max point dfs patten, not all pattern, by considering doras.
        """
        
        results = []
        for target_depth in range(depth+1):
            target_depth_dfs_result = self._dfs_chitoitsu(tehai, target_depth, doras, chitoitsu_shanten)
            if target_depth_dfs_result is not None:
                results.append((target_depth_dfs_result, target_depth))
        return results

    def _dfs_chitoitsu(self, tehai, depth, doras, chitoitsu_shanten):
        
        if chitoitsu_shanten - depth > -1:
            return None

        new_toitsu_ids = []
        
        toitsu_ids = []
        addable_dora_toitsu_candidates = []
        addable_dora_tanki_candidates = []
        addable_no_dora_tanki_candidates = []
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
        
        
        rest_depth = depth

        # add up to 7 toitsu
        while len(toitsu_ids) + len(new_toitsu_ids) < 7\
                and len(addable_dora_toitsu_candidates) > 0\
                and rest_depth >= 2:
            dora_toitsu_id = addable_dora_toitsu_candidates.pop()
            new_toitsu_ids.append(dora_toitsu_id)
            rest_depth -= 2

        while len(toitsu_ids) + len(new_toitsu_ids) < 7\
                and len(addable_dora_tanki_candidates) > 0\
                and rest_depth >= 1:
            dora_tanki_id = addable_dora_tanki_candidates.pop()
            new_toitsu_ids.append(dora_tanki_id)
            rest_depth -= 1

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
        
        return tuple(sorted(max_combination))

    

    def dfs_normal(self, tehai, furo_num, depth, current_shanten):
        """returns all dfs patten
        """
        mpsz_combinations = set()
        
        if current_shanten - depth > -1:
            # print("finish because current_shanten - depth > 1.", current_shanten,  depth)
            return mpsz_combinations
        
        changes = self.create_change(depth)
        
        manzu = tehai[0:9]
        pinzu = tehai[9:18]
        souzu = tehai[18:27]
        ji = tehai[27:34]
        manzu_sum = sum(manzu)
        pinzu_sum = sum(pinzu)
        souzu_sum = sum(souzu)
        ji_sum = sum(ji)
    
        datas = {
            MANZU:{"num":manzu,"sum":manzu_sum},
            PINZU:{"num":pinzu,"sum":pinzu_sum},
            SOUZU:{"num":souzu,"sum":souzu_sum},
            JI:{"num":ji,"sum":ji_sum},
        }
        
        for change in changes:
            for mpsz in [MANZU, PINZU, SOUZU, JI]:
                self.extract(
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
            mpsz_combinations, 
            changed_buffer_with_head, 
            changed_buffer_nohead,
            change,
            datas,
            target
        ):
        
        head_candidate = None
        mentsu_candidates = []
        for mpsj in [MANZU, PINZU, SOUZU, JI]:
            num = datas[mpsj]["num"]
            num_sum = datas[mpsj]["sum"]
            syu_change = change[mpsj]

            if target == mpsj:
                if syu_change not in changed_buffer_with_head[mpsj]:
                    # create cache
                    if mpsj == JI:
                        head_combinations = self.get_with_head_combination_ji(num, num_sum, syu_change, mpsj)
                    else:
                        head_combinations = self.get_with_head_combination(num, num_sum, syu_change, mpsj)
                    changed_buffer_with_head[mpsj][syu_change] = head_combinations
                head_candidate = changed_buffer_with_head[mpsj][syu_change]
            else:
                if syu_change not in changed_buffer_nohead[mpsj]:
                    # create cache
                    if mpsj == JI:
                        combination = self.get_combination_ji(num, num_sum, syu_change, mpsj)
                    else:
                        combination = self.get_combination(num, num_sum, syu_change, mpsj)
                    
                    changed_buffer_nohead[mpsj][syu_change] = combination
                mentsu_candidates.append(changed_buffer_nohead[mpsj][syu_change])
        
        if head_candidate is None:
            print("head_candidate is none")
            return
        
        if any([c is None for c in mentsu_candidates]):
            return

        have_mentsu_candidates = [c for c in mentsu_candidates if len(c) > 0]
        change_dist_sum = sum([sum([abs(sc) for sc in syu_change]) for syu_change in change])
        added = tuple([c[1] for c in change])

        for head in head_candidate:
            head_type_candidates = head_candidate[head]
            
            for c in itertools.product(head_type_candidates, *have_mentsu_candidates):
                flatten = tuple(sorted(itertools.chain.from_iterable(c)))
                
                mpsz_combinations.add(((head, flatten), change_dist_sum))



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
            return []
        
        mentsu_num = ((tehai_sum + sum(change)) - 2) // 3
        
        result = {}
        
        for head in range(len(tehai)):
            result_head = head + 9*pai_syu
            
            if tehai[head] >= 2:
                tehai[head] -= 2
                changed_tehais = self.apply_change(tehai, change)
                # print(f"head:{head}, change:{change}, changed_tehais len:{len(changed_tehais)}")
                for changed_tehai in changed_tehais:
                    combinations = cut_func(changed_tehai, mentsu_num, pai_syu)
                    for combination in combinations:
                        if head not in result:
                            result[result_head] = []
                        result[result_head].append(combination)
                
                if mentsu_num == 0:
                    result[result_head] = []
                    result[result_head].append([])
                tehai[head] += 2
            
            elif tehai[head] == 1 and change[1] >= 1:
                new_change = (change[0], change[1]-1)
                tehai[head] -= 1
                changed_tehais = self.apply_change(tehai, new_change)
                for changed_tehai in changed_tehais:
                    combinations = cut_func(changed_tehai, mentsu_num, pai_syu)
                    for combination in combinations:
                        if head not in result:
                            result[result_head] = []
                        result[result_head].append(combination)
                tehai[head] += 1
            
            elif tehai[head] == 0 and change[1] >= 2:
                new_change = (change[0], change[1]-2)
                
                changed_tehais = self.apply_change(tehai, new_change)
                for changed_tehai in changed_tehais:
                    combinations = cut_func(changed_tehai, mentsu_num, pai_syu)
                    for combination in combinations:
                        if head not in result:
                            result[result_head] = []
                        result[result_head].append(combination)
                
        return result



    def apply_change(self, tehai, change):
        if change == (0,0):
            return [tehai]
        if len(tehai) == 7:
            get_key_func = self.get_key_ji
        else:
            get_key_func = self.get_key
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


    MULTIPLES = list([5**i for i in range(9)])
    def _get_key(self, tehai):
        key = 0
        for i, n in enumerate(tehai):
            key += n * self.MULTIPLES[i]
        return key

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
                if record_tehai[i] > 0\
                    and record_tehai[i+1] > 0\
                    and record_tehai[i+2] > 0:
                    
                    record_tehai[i] -= 1
                    record_tehai[i+1] -= 1
                    record_tehai[i+2] -= 1
                    history.append(tuple([p + 9*pai_syu for p in (i,i+1,i+2)]))
                    queue.append((copy.copy(record_tehai), current_depth+1, copy.copy(history), i))
                    history.pop()
                    record_tehai[i] += 1
                    record_tehai[i+1] += 1
                    record_tehai[i+2] += 1
            
            # kotsu
            for i in range(start_index, 9):
                if record_tehai[i] >= 3:
                    record_tehai[i] -= 3
                    history.append(tuple([p + 9*pai_syu for p in (i,i,i)]))
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
                    history.append(tuple([p + 9*pai_syu for p in (i,i,i)]))
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
            hora = {
                "fu":30,
                "fan":100,
                "yakus":["kokushimuso",100],
                "points":48000,
                "oya_payment":0,
                "ko_payment":16000,
            }
        else:
            hora = {
                "fu":30,
                "fan":100,
                "yakus":["kokushimuso",100],
                "points":32000,
                "oya_payment":16000,
                "ko_payment":8000,
            }

        return [(hora, shanten_kokushi+1)]

    def dfs_with_score_chitoitsu(
        self,
        tehai:List[int], 
        furos:List[Furo], 
        depth:int, 
        reach:bool,
        shanten_chitoitsu:int,
        oya:bool=False, 
        bakaze:str="E", 
        jikaze:str="S", 
        doras:List[Pai]=None, 
        uradoras:List[Pai]=None,
        double_reach:bool=False,
        ippatsu:bool=False,
        rinshan:bool=False,
        haitei:bool=False,
        first_turn:bool=False,
        chankan:bool=False,
        num_akadoras:int=0,
    ):
        if doras is None:
            doras = []
        if uradoras is None:
            uradoras = []
        
        horas = []
        
        
        results = self.dfs_chitoitsu(tehai, depth, doras, shanten_chitoitsu)
    
        tehais = []
        
        for result in results:
            (toitsus, target_depth) = result
            for i, toitsu in enumerate(toitsus):
                tehais.append(Pai.from_id(toitsu[0]))
                tehais.append(Pai.from_id(toitsu[0]))

            if len(tehais) > 0:
                taken = tehais.pop()
            else:
                taken = None
            
            # print(horra_tehai, taken, taken_id)
            hora = Candidate.from_already_splited_chitoitsu(
                tehais=tehais,
                furos=furos, # List[Furo]
                taken=taken,
                oya=oya,
                bakaze=bakaze,
                jikaze=jikaze,
                doras=doras,
                uradoras=uradoras,
                reach=reach,
                double_reach=double_reach,
                ippatsu=ippatsu,
                rinshan=rinshan,
                haitei=haitei,
                first_turn=first_turn,
                chankan=chankan,
                num_akadoras=num_akadoras,
                )
            horas.append((hora, target_depth))
            
        return horas

    def dfs_with_score_normal(
        self,
        tehai:List[int], 
        furos:List[Furo], 
        depth:int, 
        reach:bool,
        shanten_normal:int,
        oya:bool=False, 
        bakaze:str="E", 
        jikaze:str="S", 
        doras:List[Pai]=None, 
        uradoras:List[Pai]=None,
        double_reach:bool=False,
        ippatsu:bool=False,
        rinshan:bool=False,
        haitei:bool=False,
        first_turn:bool=False,
        chankan:bool=False,
        num_akadoras:int=0,
    ):
        if doras is None:
            doras = []
        if uradoras is None:
            uradoras = []
        
        results = self.dfs_normal(tehai, len(furos), depth, shanten_normal)
        
        # union to min distance
        unioned_results = {}
        for result in results:
            result_tehai = result[0]
            distance = result[1]
            if result_tehai in unioned_results:
                unioned_results[result_tehai] = min(distance, unioned_results[result_tehai])
            else:
                unioned_results[result_tehai] = distance
        
        horas = []
        for result_tehai, distance in unioned_results.items():
            (head, mentsus) = result_tehai
            # try to use cache
            hora_key = (
                result_tehai, 
                tuple(sorted(furos)),
                oya,
                bakaze,
                jikaze,
                tuple(sorted(doras)),
                tuple(sorted(uradoras)),
                reach,
                ippatsu,
                rinshan,
                haitei,
                first_turn,
                chankan,
                num_akadoras
                )
            if hora_key in self.hora_cash:
                max_result = self.hora_cash[hora_key]
                horas.append((result_tehai, furos, max_result))
                continue


            changed_tehai_num = [0] * 34
            changed_tehai_num[head] += 2
            for mentsu in mentsus:
                for pai in mentsu:
                    changed_tehai_num[pai] += 1

            diff = [changed_tehai_num[i] - num for i,num in enumerate(tehai)]
            
            taken_candidate_ids = [i for (i,p) in enumerate(diff) if p > 0]
                
            # tehais.append()
            changed_tehais = []
            for i, value in enumerate(changed_tehai_num):
                for _ in range(value):
                    changed_tehais.append(Pai.from_id(i))

            taken_changed_results = []
            for taken_id in taken_candidate_ids:
            
                horra_tehai = copy.copy(changed_tehais)
                taken_index = horra_tehai.index(Pai.from_id(taken_id))
                taken = horra_tehai.pop(taken_index)

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
                    reach=reach,
                    double_reach=double_reach,
                    ippatsu=ippatsu,
                    rinshan=rinshan,
                    haitei=haitei,
                    first_turn=first_turn,
                    chankan=chankan,
                    num_akadoras=num_akadoras,
                    )
                # print(hora)
                taken_changed_results.append(hora)
            max_result = max(taken_changed_results, key= lambda x:{x["points"]*1000+x["fan"]*1000+x["fu"]})
            self.hora_cash[hora_key] = max_result
            horas.append((result_tehai, furos, max_result))
            
            # print(result_tehai, max_result)

        return horas






def main():
    import datetime
    # """
    manzu = [0,0,0,3,0,0,0,0,0]
    pinzu = [0,0,0,0,0,0,0,0,0]
    souzu = [0,1,1,1,0,0,0,0,0]
    ji = [1,0,0,2,0,0,2]
    tehai = manzu + pinzu + souzu + ji
    # """
    """
    tehai = [
        0,4,4,4,2,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
    ]
    """
    """
    tehai = [
        0,2,0,2,0,0,2,0,0,
        0,0,0,0,0,0,2,0,1,
        0,0,0,2,0,0,0,0,0,
        0,0,2,0,0,0,1,
    ]
    """
    
    depth = 3
    furo_num = (14-sum(tehai))//3

    from mjaigym.board.function.furo import Furo
    w_pai = Pai.from_str("W")
    furos = [
        Furo({"type":"pon", "actor":0, "target":2, "taken":w_pai, "consumed":[w_pai,w_pai]}),
    ]
    

    dfs = Dfs()
    print(shanten.get_shanten(tehai, 0))
    start = datetime.datetime.now()
    result = dfs.dfs_with_score_normal(tehai, furos, depth, reach=False)
    # result = dfs.dfs(tehai, furo_num, depth)
    end = datetime.datetime.now()
    
    for r in result:
        # print(r)
        pass
    print(len(result))
    print(end - start)


    



    
if __name__ == "__main__":
    main()
    