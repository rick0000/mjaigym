import os
import itertools
import time
import numpy as np
import array
from mjaigym import shanten
class RsShantenAnalysis:

    def calc_shanten(self, tehai, furonum):
        return shanten.get_shanten(tehai, furonum)


    def calc_all_shanten(self, tehai, furonum):
        """returns [normal, kokushi, chitoi]

        Args:
            tehai ([type]): [description]
            furonum ([type]): [description]

        Returns:
            []
        """
        return shanten.get_shanten_all(tehai, furonum)

    @classmethod
    def benchmark(cls):
        from mjaigym.board.function.yama import Yama
        start = time.time()
        print("start prepare:",start)
        iter_num = 10000
        tests = [None] * iter_num
        for i in range(iter_num):
            y = Yama()
            pais = y.all_yama[0:14]
            pais = sorted(pais)

            tehai = [0] * 34
            for p in pais:
                tehai[p.id] += 1

            tests[i] = tehai

        fsa = RsShantenAnalysis()
        fsa.calc_shanten(tests[0],0)
        end = time.time()
        print("end prepare:",end)
        print(f"need {(end-start):6f} seconds for {iter_num} prepare")

        
        start = time.time()
        print("start:",start)

        # for test in tests:
        #     shanten = fsa.calc_shanten(test, 0)
        multiple = 100
        for _ in range(multiple):
            [fsa.calc_shanten(t,0) for t in tests]

        end = time.time()
        print("end:",end)
        print(f"need {(end-start):6f} seconds for {iter_num}*{multiple} iteration")



if __name__ == "__main__":
    import datetime
    rsa = RsShantenAnalysis()
    rsa.benchmark()
    
    sample = [3, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,]
    sample = sample
    print(datetime.datetime.now())
    for i in range(10000):
        rsa.calc_shanten(sample,0)
    print(datetime.datetime.now())
    