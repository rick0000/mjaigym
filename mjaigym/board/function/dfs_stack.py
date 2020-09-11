from collections import deque
import copy
import numpy as np

from mjaigym.board.function.rs_shanten_analysis import RsShantenAnalysis
from mjaigym.board.function.pai import Pai
from mjaigym.board.function.hora import Hora
from mjaigym.board.function.furo import Furo

from .shanten_score_cache import ShantenScoreCache
CACHE = ShantenScoreCache()


class Dfs():
    def __init__(self):
        self.score_cache = {}
        self.shanten_analysis = RsShantenAnalysis()


    def dfs_hora(self, depth, tehais, furos, points, rest=None):
        """return pai change combinations which achive hora and hora score >= points.

        Args:
            depth ([type]): [description]
            tehais ([type]): [description]
            furos ([type]): [description]
            points ([type]): [description]
            rest ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        
        if rest is None:
            rest = [4] * 34
            for i in range(34):
                rest[i] -= tehais[i]

        result = {}
        # initialize
        for point in points:
            result[point] = [0] * 34
    
        if sum(tehais) % 3 == 2:
            for point in points:
                dfs_results = self._dfs_hora(depth, tehais, furos, point, rest)
                result[point] = np.array(dfs_results, dtype="int8")

        elif sum(tehais) % 3 == 1:
            for point in points:
                result[point] = np.zeros((34,), dtype="int8")
        else:
            raise Exception("not intended path")

        return result


    def _dfs_hora(self, depth, tehais, furos, point, rest, added=None):
        """returns [bool] * 34
        """
        
        stack = deque()
        furo_num = len(furos)
        base_shanten = self.shanten_analysis.calc_shanten(tehais, furo_num)

        if added:
            initial_history = [tuple((None, added))]
        else:
            initial_history = []
            
        stack.append((base_shanten, depth, tehais, furos, initial_history))
        
        i_found = [False] * 34
        

        while len(stack) > 0:
            target_shanten, target_depth, target_tehais, target_furos, target_history = stack.pop()
            
            if target_depth <= 0:
                continue
            
            if len(target_history) > 0 and i_found[target_history[0][0]]:
                continue

            # apply history for rest
            for h in target_history:
                rest[h[1]] -= 1

            # sub
            for i in range(34):
                temp_i_found = False
                if target_tehais[i] == 0:
                    continue
                
                target_tehais[i]-=1
                shanten = self.shanten_analysis.calc_shanten(target_tehais,furo_num)
                if shanten - depth > -1:
                    # cannot reach hora
                    target_tehais[i]+=1
                    continue
                if target_shanten < shanten:
                    target_tehais[i]+=1
                    continue
                
                rest_depth = target_depth - 1
                # add
                for j in range(34):
                    if temp_i_found:
                        break
                    if target_tehais[j] == 4 or rest[j] == 0:
                        continue

                    target_tehais[j]+=1
                    target_history.append(tuple((i,j)))

                    changed_shanten = self.shanten_analysis.calc_shanten(target_tehais, furo_num)
                    
                    if changed_shanten - (rest_depth) <= -1:
                        if changed_shanten == -1:
                            # if hora, treat as leaf
                            key = get_key(target_tehais, target_furos, j)
                            if key not in self.score_cache:
                                hora = get_score(target_tehais, target_furos, target_history[-1][1])
                                self.score_cache[key] = hora.points
                            score = self.score_cache[key]
                            if score >= point:
                                i_found[target_history[0][0]] = True
                                temp_i_found = True
                                
                        else:
                            stack.append((changed_shanten, rest_depth, copy.copy(target_tehais), copy.copy(target_furos), copy.copy(target_history)))
                    
                    target_history.pop()
                    target_tehais[j] -= 1
                
                target_tehais[i]+=1
            
            # restore rest
            for h in target_history:
                rest[h[1]] += 1
        
        return i_found



KEY_MUL = [5**i for i in range(9)]

def get_key(tehais, furos, taken):
    
    key1 = (tehais[0] * KEY_MUL[0] +\
        tehais[1] * KEY_MUL[1] +\
        tehais[2] * KEY_MUL[2] +\
        tehais[3] * KEY_MUL[3] +\
        tehais[4] * KEY_MUL[4] +\
        tehais[5] * KEY_MUL[5] +\
        tehais[6] * KEY_MUL[6] +\
        tehais[7] * KEY_MUL[7] +\
        tehais[8] * KEY_MUL[8])

    key2 = (tehais[9] * KEY_MUL[0] +\
        tehais[10] * KEY_MUL[1] +\
        tehais[11] * KEY_MUL[2] +\
        tehais[12] * KEY_MUL[3] +\
        tehais[13] * KEY_MUL[4] +\
        tehais[14] * KEY_MUL[5] +\
        tehais[15] * KEY_MUL[6] +\
        tehais[16] * KEY_MUL[7] +\
        tehais[17] * KEY_MUL[8])

    key3 = (tehais[18] * KEY_MUL[0] +\
        tehais[19] * KEY_MUL[1] +\
        tehais[20] * KEY_MUL[2] +\
        tehais[21] * KEY_MUL[3] +\
        tehais[22] * KEY_MUL[4] +\
        tehais[23] * KEY_MUL[5] +\
        tehais[24] * KEY_MUL[6] +\
        tehais[25] * KEY_MUL[7] +\
        tehais[26] * KEY_MUL[8])

    key4 = (tehais[27] * KEY_MUL[0] +\
        tehais[28] * KEY_MUL[1] +\
        tehais[29] * KEY_MUL[2] +\
        tehais[30] * KEY_MUL[3] +\
        tehais[31] * KEY_MUL[4] +\
        tehais[32] * KEY_MUL[5] +\
        tehais[33] * KEY_MUL[6])
    return (key1,key2,key3,key4,tuple([hash(f) for f in furos]), taken)

def get_score(tehais_num, furos, taken):
    tehais = []
    for i in range(len(tehais_num)):
        for _ in range(tehais_num[i]):
            tehais.append(Pai.from_id(i))

    taken_index = tehais.index(Pai.from_id(taken))
    taken = tehais.pop(taken_index)

    hora = Hora(
            tehais=tehais,
            furos=furos,
            taken=taken,
            hora_type='tsumo',
            oya=False,
            bakaze='E',
            jikaze='S',
            doras=[],
            uradoras=[],
            reach=len(furos)==0,
            double_reach=False,
            ippatsu=False,
            rinshan=False,
            haitei=False,
            first_turn=False,
            chankan=False,
        )
    return hora




