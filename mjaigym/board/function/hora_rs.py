import copy
import math

from mjaigym import shanten

from mjaigym.board.function.pai import Pai
from mjaigym.board.function.mentsu import Mentsu
from mjaigym.board.function.shanten_analysis import ShantenAnalysis
from mjaigym.board.function.mj_move import MjMove

        
class HoraRs():

    def __init__(
        self,
        tehais,
        furos,
        taken,
        hora_type,
        oya,
        bakaze,
        jikaze,
        doras,
        uradoras,
        reach,
        double_reach,
        ippatsu,
        rinshan,
        haitei,
        first_turn,
        chankan,
        num_akadoras=None
    ):
        self.tehais = copy.copy(tehais)
        self.furos = furos
        self.taken = taken # last tsumo
        self.hora_type = hora_type
        self.oya = oya
        self.bakaze = bakaze
        self.jikaze = jikaze
        self.doras = doras
        self.uradoras = uradoras
        self.reach = reach
        self.double_reach = double_reach
        self.ippatsu = ippatsu
        self.rinshan = rinshan
        self.haitei = haitei
        self.first_turn = first_turn
        self.chankan = chankan


        self.free_pais = self.tehais + [self.taken]
        furos_flatten = []
        for f in self.furos:
            furos_flatten.extend(f.pais)

        furo_input = []
        for f in self.furos:
            furo_input.append(f.to_rs_hora_furo())
            
        
        self.all_pais = self.free_pais + furos_flatten

        self.num_doras = self.count_doras(self.doras)
        self.num_uradoras = self.count_doras(self.uradoras)
        if num_akadoras is None:
            self.num_akadoras = len([p for p in self.all_pais if p.is_red])
        else:
            self.num_akadoras = num_akadoras

        doras_str = [d.str for d in doras]
        uradoras_str = [d.str for d in uradoras]
        show = False
        tehai_str = [t.str for t in tehais]
        taken_str = taken.str
        try:
            result = shanten.get_hora(
                tehai_str,
                furo_input,
                taken_str,
                oya,
                hora_type,
                first_turn,
                doras_str,
                uradoras_str,
                reach,
                double_reach,
                ippatsu,
                rinshan,
                chankan,
                haitei,
                bakaze,
                jikaze,
                show,
            )
        except:
            import pdb; pdb.set_trace(); import time; time.sleep(1)
            result = ([0,0,0,0,0], [])
        self._points = result[0]
        self._yakus = result[1]
    
    @property
    def fu(self):
        return self._points[0]

    @property
    def fan(self):
        return self._points[1]
    
    
    @property
    def points(self):
        return self._points[2]
    
    @property
    def yakus(self):
        return self._yakus

    @property
    def valid(self):
        return len([y for y in self._yakus if y[0] not in ['dora', 'uradora', 'akadora']]) > 0

    @property
    def oya_payment(self):
        return self._points[3]

    @property
    def ko_payment(self):
        return self._points[4]

    def count_doras(self, target_doras):
        dora_sum = 0
        all_pais = self.all_pais
        for td in target_doras:
            dora_sum += sum([1 for p in all_pais if Pai.is_same_symbol(td,p)])
        return dora_sum


    def __str__(self):
        return self.yakus.__str__() + str(self.all_pais)


