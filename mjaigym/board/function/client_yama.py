from mjaigym.board.function.yama import INITIAL_YAMA, Yama
from mjaigym.board.function.pai import Pai, UNKNOWN_PAI_STR

class ClientYama(Yama):

    def __init__(self):
        self.rest = INITIAL_YAMA.copy()
        self._tsumoed_num = 0
        self._doramarker = []
    
    def tsumo(self, pai:str):
        self._tsumoed_num += 1
        if pai == UNKNOWN_PAI_STR:
            return

        self.rest.remove(Pai.from_str(pai))

    

    def get_rest_num(self):
        return len(INITIAL_YAMA) - self.get_tsumoed_num() - 14
    
    def get_tsumoed_num(self):
        """ when first player first tsumo, returns 0
        """
        return self._tsumoed_num - 13 * 4

    def get_consumed_num(self):
        """ returns get_tsumoed_num + haipai_num (=13*4)
        """
        return self._tsumoed_num

    def get_doramarker_num(self):
        return len(self._doramarker)
        
    def open_doramarker(self, pai:str):
        self._doramarker.append(pai)
