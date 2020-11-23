from mjaigym.board.function.pai  import Pai
from mjaigym.board.function.shanten_analysis import ShantenAnalysis
from mjaigym.board.function.rs_shanten_analysis import RsShantenAnalysis
from collections import OrderedDict

class TenpaiAnalysis:
    ALL_YAOCHUS = Pai.from_list([
        "1m","9m","1p","9p","1s","9s","E","S","W","N","P","F","C",
    ])

    def __init__(self, pais):
        self.pais = pais
        self.shanten_analysis = RsShantenAnalysis()
        self.tehai = [0]*34
        for p in pais:
            self.tehai[p.id] += 1
        
        self.furo_num = (14 - len(pais)) // 3
        
        self.shanten = self.shanten_analysis.calc_shanten(self.tehai, self.furo_num)

        self.waiting = []
        if self.shanten != 0:
            return

        for waiting_id in range(34):
            if self.tehai[waiting_id] == 4:
                # already use all pais
                continue

            self.tehai[waiting_id] += 1
            if self.shanten_analysis.calc_shanten(self.tehai, self.furo_num) == -1:
                self.waiting.append(Pai.from_id(waiting_id))
            self.tehai[waiting_id] -= 1
        

        
        
    
    @property
    def tenpai(self):
        if self.shanten != 0:
            return False
        
        # assert self.shanten.shanten == 0
        return (len(self.pais) % 3 != 1) or (len(self.waited_pais) > 0)
        
    @property
    def waited_pais(self):
        # assert len(self.pais) % 3 == 1, "invalid number of pais"
        # assert self.shanten.shanten == 0, "not tenpai"
        return self.waiting



if __name__ == "__main__":
    pais = Pai.from_list([
        "1m","1m","1m","1p","5m","6m","7m","5p","6p","7p","9m","9m","9m",
    ])
    ana = TenpaiAnalysis(pais)
    print(ana.tenpai)
    print(ana.waited_pais)