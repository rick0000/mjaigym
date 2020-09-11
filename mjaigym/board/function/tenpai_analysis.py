from mjaigym.board.function.pai  import Pai
from mjaigym.board.function.shanten_analysis import ShantenAnalysis
from collections import OrderedDict

class TenpaiAnalysis:
    ALL_YAOCHUS = Pai.from_list([
        "1m","9m","1p","9p","1s","9s","E","S","W","N","P","F","C",
    ])

    def __init__(self, pais):
        self.pais = pais
        self.shanten = ShantenAnalysis(pais, 0)
    
    @property
    def tenpai(self):
        if self.shanten.shanten != 0:
            return False
        
        
        # remove already use all pai case.
        any_not_use_4 = False
        for wp in self.waited_pais:
            if len([p for p in self.pais if p.is_same_symbol(wp)]) < 4:
                any_not_use_4 = True
                break
        
        # assert self.shanten.shanten == 0
        return len(self.pais) % 3 != 1 or any_not_use_4
        
    @property
    def waited_pais(self):
        # assert len(self.pais) % 3 == 1, "invalid number of pais"
        # assert self.shanten.shanten == 0, "not tenpai"

        pai_set = OrderedDict()
        for pai in self.pais:
            k = pai.remove_red()
            if k not in pai_set:
                pai_set[k] = 0
            pai_set[k] += 1
        
        result = []

        for mentsus in self.shanten.combinations:
            if mentsus == "chitoitsu":
                waiting = [k for (k,v) in pai_set.items() if v==1]
                result += waiting
            elif mentsus == "kokushimuso":
                missing = list(self.ALL_YAOCHUS - pai_set.keys())
                if len(missing) == 0:
                    result += self.ALL_YAOCHUS
                else:
                    result.append(missing[0])
            else:
                toitsu_num = len([m for m in mentsus if m[0]=='toitsu'])
                if toitsu_num == 0:
                    (type, pais) = [m for m in mentsus if m[0] == 'single'][0]
                    result.append(pais[0])
                if toitsu_num == 1:
                    (type, pais) = [m for m in mentsus if m[0] in ['ryanpen', 'kanta']][0]
                    if type == 'kanta':
                        machi_nums = [n + pais[0].number for n in [1]]
                    else:
                        machi_nums = [n + pais[0].number for n in [-1, 2]]
                    machi_nums = [n for n in machi_nums if 1 <= n and n <= 9]
                    result += [Pai.from_number_type(n, pais[0].type) for n in machi_nums]
                if toitsu_num == 2:
                    result += [m[1][0].remove_red() for m in mentsus if m[0] == 'toitsu']
        return sorted(set(result))



if __name__ == "__main__":
    pais = Pai.from_list([
        "1p","1p","1m","2m","3m","4m","5m","6m","7m","8m","9m","9m","9m",
    ])
    ana = TenpaiAnalysis(pais)
    print(ana.tenpai)
    print(ana.waited_pais)