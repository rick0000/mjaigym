from collections import deque
import random
import copy
from mjaigym.board.function.pai import Pai


INITIAL_YAMA = [
    "1m","2m","3m","4m","5m","6m","7m","8m","9m",
    "1p","2p","3p","4p","5p","6p","7p","8p","9p",
    "1s","2s","3s","4s","5s","6s","7s","8s","9s",
    "E","S","W","N","P","F","C",
] * 4
INITIAL_YAMA.remove('5m')
INITIAL_YAMA.remove('5p')
INITIAL_YAMA.remove('5s')

INITIAL_YAMA.append('5mr')
INITIAL_YAMA.append('5pr')
INITIAL_YAMA.append('5sr')
INITIAL_YAMA = sorted(INITIAL_YAMA)

INITIAL_YAMA = sorted(Pai.from_list(INITIAL_YAMA))

class Yama():
    def __init__(self, seed=None, shuffle=True):
        if not seed:
            random.seed(seed)
        shuffled = copy.copy(INITIAL_YAMA)
        if shuffle:
            random.shuffle(shuffled)
        self._yama = shuffled
        self._pointer = 0
        self._doramarker = []


    def paifu_tsumo(self, pai):
        rest_yama = [y.str for y in self._yama[self._pointer:]]
        pai_index = rest_yama.index(pai)
        self._yama[self._pointer], self._yama[self._pointer+pai_index] = self._yama[self._pointer+pai_index], self._yama[self._pointer]
        return self.tsumo()

    def paifu_open_doramarker(self, pai):
        rest_yama = [y.str for y in self._yama[self._pointer:]]
        pai_index = rest_yama.index(pai)
        self._yama[-1], self._yama[self._pointer+pai_index] = self._yama[self._pointer+pai_index], self._yama[-1]
        return self.open_doramarker()


    

    def tsumo(self):
        tsumo_pai = self._yama[self._pointer]
        self._pointer += 1
        return tsumo_pai

    def open_doramarker(self):
        doramarker = self._yama.pop(-1)
        self._doramarker.append(doramarker)
        return doramarker

    def get_rest_num(self):
        # 14 is wanpai

        # when open_dora, _yama is poped, so add get_doramarker_num.
        return len(self._yama) - self._pointer - 14 + self.get_doramarker_num() 

    def get_tsumoed_num(self):
        """ when first player first tsumo, returns 0
        """
        return self._pointer - 13*4

    def get_consumed_num(self):
        """ returns get_tsumoed_num + haipai_num (=13*4)
        """
        return self._pointer


    def get_doramarker_num(self):
        return len(self._doramarker)

    def shuffle_rest(self):
        fix = copy.copy(self._yama[:self._pointer])
        shuffled = copy.copy(self._yama[self._pointer:])
        random.shuffle(shuffled)
        self._yama = fix + shuffled

    @property
    def all_yama(self):
        return list(copy.copy(self._yama))

    @property
    def rest_yama(self):
        return list(copy.copy(self._yama[self._pointer:]))

    
if __name__ == "__main__":    
    y = Yama(0)
    for _ in range(10):
        y.tsumo()

    print(y.get_rest_num())
    for _ in range(10):
        y.open_doramarker()

    print(y.get_rest_num())

