import copy
import math
from typing import List
from dataclasses import dataclass

from mjaigym.board.function.pai import Pai
from mjaigym.board.function.mentsu import Mentsu
from mjaigym.board.function.furo import Furo
from mjaigym.board.function.shanten_analysis import ShantenAnalysis
from mjaigym.board.function.mj_move import MjMove


FUROTYPE_TO_MENTSU_TYPE = {
    MjMove.chi.value:'shuntsu',
    MjMove.pon.value:'kotsu',
    MjMove.daiminkan.value:'kantsu',
    MjMove.kakan.value:'kantsu',
    MjMove.ankan.value:'kantsu',
}

BASE_FU_MAP = {
    'shuntsu':0,
    'kotsu':2,
    'kantsu':8,
}



GREEN_PAIS = ['2s','3s', '4s', '6s', '8s', 'F']
CHUREN_NUMBERS = [1,1,1,2,3,4,5,6,7,8,9,9,9]
CHUREN_NUMBERS_ALL_VALIATION = []
for i in range(1,10): # 9 pattern
    CHUREN_NUMBERS_ALL_VALIATION.append(sorted(CHUREN_NUMBERS+[i]))

YAKUMAN_FAN = 100


class PointDatam():
    def __init__(
        self,
        fu,
        fan,
        oya,
        hora_type
        ):
        self.fu = fu
        self.fan = fan
        self.oya = oya
        
        if self.fan >= YAKUMAN_FAN:
            self.base_points = 8000 * (self.fan / YAKUMAN_FAN)
        elif self.fan >= 13:
            self.base_points = 8000
        elif self.fan >= 11:
            self.base_points = 6000
        elif self.fan >= 8:
            self.base_points = 4000
        elif self.fan >= 6:
            self.base_points = 3000
        elif self.fan >= 5 or (self.fan >= 4 and self.fu >= 40) or (self.fan >= 3 and self.fu >= 70):
            self.base_points = 2000
        else:
            self.base_points = self.fu * (2 ** (self.fan+2))

        if hora_type == 'ron':
            if self.oya:
                multiple = 6
            else:
                multiple = 4
            points = self.ceil_points(self.base_points * multiple)
            self.points = points
            self.oya_payment = points
            self.ko_payment = points
        else:
            if self.oya:
                self.ko_payment = self.ceil_points(self.base_points * 2)
                self.oya_payment = 0
                self.points = self.ko_payment * 3
            else:
                self.oya_payment = self.ceil_points(self.base_points * 2)
                self.ko_payment = self.ceil_points(self.base_points)
                self.points = self.oya_payment + self.ko_payment * 2

    def ceil_points(self, points):
        return math.ceil(points / 100.0) * 100


    
@dataclass
class HoraYakuInformation():
    taken:Pai
    all_pais:List[Pai]
    hora_type:str
    oya:bool
    first_turn:bool
    num_doras:int
    num_uradoras:int
    num_akadoras:int
    reach:bool
    ippatsu:bool
    rinshan:bool
    chankan:bool
    haitei:bool
    double_reach:bool
    furos: List[Furo]
    jikaze:str # str
    bakaze:str # str

class Candidate():
    @classmethod
    def from_already_splited_chitoitsu(
        cls,
        tehais,
        furos, # List[Furo]
        taken, # 20
        oya,
        bakaze,
        jikaze,
        doras,
        uradoras,
        num_akadoras,
    ):
        if len(furos) > 0:
            return {
                "fu":0,
                "fan":0,
                "yakus":[],
                "points":0,
                "oya_payment":0,
                "ko_payment":0,
            }

        all_pais = tehais

        num_doras = Hora.count_doras(all_pais, doras)
        num_uradoras = Hora.count_doras(all_pais, uradoras)
        # num_akadora need calclate outside.
        is_menzen = len([f for f in furos if f.type != 'ankan']) == 0
        combination = "chitoitsu"
        
        hora_yaku_information = HoraYakuInformation(
            taken=taken,
            all_pais=all_pais,
            hora_type="tsumo",
            oya=oya,
            first_turn=False,
            num_doras=num_doras,
            num_uradoras=num_uradoras,
            num_akadoras=num_akadoras,
            reach=is_menzen,
            ippatsu=False,
            rinshan=False,
            chankan=False,
            haitei=False,
            double_reach=False,
            furos=furos,
            jikaze=jikaze,
            bakaze=bakaze,
        )
        
        best_candidate = Candidate(hora_yaku_information, combination, 0)
        if best_candidate.valid:
            return {
                "fu":best_candidate.fu,
                "fan":best_candidate.fan,
                "yakus":best_candidate.yakus,
                "points":best_candidate.points,
                "oya_payment":best_candidate.oya_payment,
                "ko_payment":best_candidate.ko_payment,
            }
        return {
            "fu":0,
            "fan":0,
            "yakus":[],
            "points":0,
            "oya_payment":0,
            "ko_payment":0,
        }


    @classmethod
    def from_already_spliteds(
        cls,
        head, # 9
        mentsus, # ((1,2,3), (4,5,6), (20,20,20), (32,32,32))
        furos, # List[Furo]
        taken, # 20
        oya,
        bakaze,
        jikaze,
        doras,
        uradoras,
        num_akadoras,
    ):
        
        # need before calclate dora
        pais_buffer = Pai.from_idlist([head, head])
         
        for mentsu in mentsus:
            pais_buffer.extend(Pai.from_idlist(list(mentsu)))
        
        free_pais = copy.copy(pais_buffer)

        for furo in furos:
            pais_buffer.extend(furo.pais)
        all_pais = pais_buffer
        is_menzen = len([f for f in furos if f.type != 'ankan']) == 0
        taken = Pai.from_id(taken)

        num_doras = Hora.count_doras(all_pais, doras)
        num_uradoras = Hora.count_doras(all_pais, uradoras)
        # num_akadora need calclate outside.
        
        num_same_as_taken = len([f for f in free_pais if taken.is_same_symbol(f)])
        
        combination = [
            ["toitsu", Pai.from_idlist([head,head])],
        ]
        for mentsu in mentsus:
            if mentsu[0] == mentsu[1]:
                combination.append(["kotsu", Pai.from_idlist(mentsu)])
            else:
                combination.append(["shuntsu", Pai.from_idlist(mentsu)])
        
        hora_yaku_information = HoraYakuInformation(
            taken=taken,
            all_pais=all_pais,
            hora_type="tsumo",
            oya=oya,
            first_turn=False,
            num_doras=num_doras,
            num_uradoras=num_uradoras,
            num_akadoras=num_akadoras,
            reach=is_menzen,
            ippatsu=False,
            rinshan=False,
            chankan=False,
            haitei=False,
            double_reach=False,
            furos=furos,
            jikaze=jikaze,
            bakaze=bakaze,
        )

        candidates = []
        for i in range(num_same_as_taken):
            candidates.append(Candidate(hora_yaku_information, combination, i))
        
        if len(candidates) > 0:
            best_candidate = max(candidates, key=lambda x:(x.fan, x.points))
            if best_candidate.valid:
                return {
                    "fu":best_candidate.fu,
                    "fan":best_candidate.fan,
                    "yakus":best_candidate.yakus,
                    "points":best_candidate.points,
                    "oya_payment":best_candidate.oya_payment,
                    "ko_payment":best_candidate.ko_payment,
                }
        return {
            "fu":0,
            "fan":0,
            "yakus":[],
            "points":0,
            "oya_payment":0,
            "ko_payment":0,
        }
        

    def __init__(self, hora_yakuinfo, combination, taken_index):
        self.hora_yakuinfo = hora_yakuinfo
        
        self.yakus = []
        self.combination = combination
        self.taken_index = taken_index
        self.all_pais = self.hora_yakuinfo.all_pais
        self.mentsus = []
        self.janto = None
        
        total_taken = 0
        if self.combination == 'chitoitsu':
            self.machi = 'tanki'
            uniq_all_pais = list(set(self.all_pais))
            for pai in uniq_all_pais:
                mentsu = Mentsu(pais=[pai, pai], type='toitsu', visibility='an')
                if pai.is_same_symbol(self.hora_yakuinfo.taken):
                    self.janto = mentsu
                else:
                    self.mentsus.append(mentsu)
        elif self.combination == 'kokushimuso':
            self.machi = 'tanki'
        else:
            for mentsu_type, mentsu_pais in self.combination:
                num_this_taken = len([p for p in mentsu_pais if self.hora_yakuinfo.taken.is_same_symbol(p)])
                has_taken = self.taken_index >= total_taken and self.taken_index < total_taken + num_this_taken
                if mentsu_type == 'toitsu':
                    if self.janto:
                        raise Exception('should not happen')
                    self.janto = Mentsu(pais=mentsu_pais, type='toitsu', visibility=None)
                else:
                    min_an = 'min' if (has_taken and self.hora_yakuinfo.hora_type == 'ron') else 'an'
                    self.mentsus.append(Mentsu(
                        pais=mentsu_pais,
                        type=mentsu_type,
                        visibility=min_an,
                    ))
                if has_taken > 0:
                    if mentsu_type == 'toitsu':
                        self.machi = 'tanki'
                    elif mentsu_type == 'kotsu':
                        self.machi = 'shanpon'
                    elif mentsu_type == 'shuntsu':
                        if self.hora_yakuinfo.taken.is_same_symbol(mentsu_pais[1]):
                            self.machi = 'kanchan'
                        elif (mentsu_pais[0].number == 1 and self.hora_yakuinfo.taken.number == 3) or \
                            (mentsu_pais[0].number == 7 and self.hora_yakuinfo.taken.number == 7):
                            self.machi = 'penchan'
                        else:
                            self.machi = 'ryanmen'
                total_taken += num_this_taken

        # assert self.machi is not None
        for furo in self.hora_yakuinfo.furos:
            an_min = 'an' if (furo.type == 'ankan') else 'min'
            self.mentsus.append(Mentsu(
                pais=sorted([f.remove_red() for f in furo.pais]),
                type=FUROTYPE_TO_MENTSU_TYPE[furo.type],
                visibility=an_min,
            ))

        self.get_yakus()
        self.fan = sum([y[1] for y in self.yakus])
        self.fu = self.get_fu()
        
        datum = PointDatam(self.fu, self.fan, self.hora_yakuinfo.oya, self.hora_yakuinfo.hora_type)
        self.points = datum.points
        self.oya_payment = datum.oya_payment
        self.ko_payment = datum.ko_payment
        #print("yakus:",self.yakus)


    def __str__(self):
        return str((self.fan, self.fu, self.yakus, self.combination))
        

    def get_fu(self):
        if self.combination == 'chitoitsu':
            return 25
        elif self.combination == 'kokushimuso':
            return 20
        else:
            fu = 20
            if self.menzen and self.hora_yakuinfo.hora_type == 'ron':
                fu += 10
            if self.hora_yakuinfo.hora_type == 'tsumo' and self.pinfu == False:
                fu += 2
            if self.menzen == False and self.pinfu:
                fu += 2
            for m in self.mentsus:
                mfu = BASE_FU_MAP[m.type]
                if m.pais[0].is_yaochu():
                    mfu *= 2
                if m.visibility == 'an':
                    mfu *= 2
                fu += mfu

            fu += self.fanpai_fan(self.janto.pais[0]) * 2
            if self.machi in ['kanchan', 'penchan', 'tanki']:
                fu += 2
            
            return math.ceil(fu / 10.0) * 10

    @property
    def valid(self):
        return len([y for y in self.yakus if y[0] not in ['dora', 'uradora', 'akadora']]) > 0

    def add_yaku(self, name, menzen_fan, kui_fan):
        fan = menzen_fan if self.menzen else kui_fan
        if fan > 0:
            self.yakus.append([name, fan])

    def delete_yaku(self, name):
        remove_index = -1
        for i,y in enumerate(self.yakus):
            if y[0] == name:
                remove_index = i
        if remove_index != -1:
            self.yakus.pop(remove_index)

    def get_yakus(self):
        if self.hora_yakuinfo.first_turn and self.hora_yakuinfo.hora_type == 'tsumo' and self.hora_yakuinfo.oya:
            self.add_yaku('tenho', YAKUMAN_FAN, 0)
        if self.hora_yakuinfo.first_turn and self.hora_yakuinfo.hora_type == 'tsumo' and not self.hora_yakuinfo.oya:
            self.add_yaku('chiho', YAKUMAN_FAN, 0)
        if self.combination == 'kokushimuso':
            self.add_yaku('kokushimuso', YAKUMAN_FAN, 0)
        if self.num_sangenpais == 3:
            self.add_yaku('daisangen', YAKUMAN_FAN, YAKUMAN_FAN)
        if self.num_anko == 4:
            self.add_yaku('suanko', YAKUMAN_FAN, 0)
        if all([p.is_jihai() for p in self.all_pais]):
            self.add_yaku('tsuiso', YAKUMAN_FAN, YAKUMAN_FAN)
        if self.ryuiso:
            self.add_yaku('ryuiso', YAKUMAN_FAN, YAKUMAN_FAN)
        if self.chinroto:
            self.add_yaku('chinroto', YAKUMAN_FAN, YAKUMAN_FAN)
        if self.daisushi:
            self.add_yaku('daisushi', YAKUMAN_FAN, YAKUMAN_FAN)
        if self.shosushi:
            self.add_yaku('shosushi', YAKUMAN_FAN, YAKUMAN_FAN)
        if self.num_kantsu == 4:
            self.add_yaku('sukantsu', YAKUMAN_FAN, YAKUMAN_FAN)
        if self.churenpoton:
            self.add_yaku('churenpoton', YAKUMAN_FAN, 0)
        
        if len(self.yakus) > 0:
            return # 役満の場合は後の役を考慮しない
        
        self.add_yaku('dora', self.hora_yakuinfo.num_doras, self.hora_yakuinfo.num_doras)
        self.add_yaku('uradora', self.hora_yakuinfo.num_uradoras, self.hora_yakuinfo.num_uradoras)
        self.add_yaku('akadora', self.hora_yakuinfo.num_akadoras, self.hora_yakuinfo.num_akadoras)

        # 一翻
        if self.hora_yakuinfo.reach:
            self.add_yaku('reach', 1, 0)
        if self.hora_yakuinfo.ippatsu:
            self.add_yaku('ippatsu', 1, 0)
        if self.menzen and self.hora_yakuinfo.hora_type == 'tsumo':
            self.add_yaku('menzenchin_tsumoho', 1, 0)
        if all([p.is_yaochu() == False for p in self.all_pais]):
            self.add_yaku('tanyaochu', 1, 1)
        if self.pinfu:
            self.add_yaku('pinfu', 1, 0)
        if self.ipeko:
            self.add_yaku('ipeko', 1, 0)
        self.add_yaku('sangenpai', self.num_sangenpais, self.num_sangenpais)
        if self.bakaze:
            self.add_yaku('bakaze', 1, 1)
        if self.jikaze:
            self.add_yaku('jikaze', 1, 1)
        if self.hora_yakuinfo.rinshan:
            self.add_yaku('rinshankaiho', 1, 1)
        if self.hora_yakuinfo.chankan:
            self.add_yaku('chankan', 1, 1)
        if self.hora_yakuinfo.haitei and self.hora_yakuinfo.hora_type == 'tsumo':
            self.add_yaku('haiteiraoyue', 1, 1)
        if self.hora_yakuinfo.haitei and self.hora_yakuinfo.hora_type == 'ron':
            self.add_yaku('hoteiraoyui', 1, 1)


        # 二翻
        if self.sanshokudojun:
            self.add_yaku('sanshokudojun', 2, 1)
        if self.ikkitsukan:
            self.add_yaku('ikkitsukan', 2, 1)
        if self.honchantaiyao:
            self.add_yaku('honchantaiyao', 2, 1)
        if self.combination == 'chitoitsu':
            self.add_yaku('chitoitsu', 2, 0)
        if all([m.type in ['kotsu', 'kantsu'] for m in self.mentsus]):
            self.add_yaku('toitoiho', 2, 2)
        if self.num_anko == 3:
            self.add_yaku('sananko', 2, 2)
        if all([p.is_yaochu() for p in self.all_pais]):
            self.add_yaku('honroto', 2, 2)
            self.delete_yaku('honchantaiyao')
        if self.sanshokudoko:
            self.add_yaku('sanshokudoko', 2, 2)
        if self.num_kantsu == 3:
            self.add_yaku('sankantsu', 2, 2)
        if self.shosangen:
            self.add_yaku('shosangen', 2, 2)
        if self.hora_yakuinfo.double_reach:
            self.add_yaku('double_reach', 2, 0)
            self.delete_yaku('reach')
        

        # 三翻
        if self.honiso:
            self.add_yaku('honiso', 3, 2)
        if self.junchantaiyao:
            self.add_yaku('junchantaiyao', 3, 2)
            self.delete_yaku('honchantaiyao')
        if self.ryanpeko:
            self.add_yaku('ryanpeko', 3, 0)
            self.delete_yaku('ipeko')
        
        # 六翻
        if self.chiniso:
            self.add_yaku('chiniso', 6, 5)
            self.delete_yaku('honiso')
        


    @property
    def menzen(self):
        return len([f for f in self.hora_yakuinfo.furos if f.type != 'ankan']) == 0

    @property
    def num_anko(self):
        return len([m for m in self.mentsus if ((m.type in ['kotsu', 'kantsu']) and (m.visibility == 'an'))])

    @property
    def num_kantsu(self):
        return len([m for m in self.mentsus if m.type == 'kantsu'])

    @property
    def ryuiso(self):
        return all([p.str in GREEN_PAIS for p in self.all_pais])
    
    @property
    def chinroto(self):
        return all([p.is_number() and p.number in [1,9] for p in self.all_pais])
    

    @property
    def daisushi(self):
        return len([m for m in self.mentsus if m.pais[0].is_wind() and m.type in ['kotsu', 'kantsu']]) == 4

    @property
    def shosushi(self):
        is_3_wind = len([m for m in self.mentsus if m.type in ['kotsu', 'kantsu'] and m.pais[0].is_wind()]) == 3
        return is_3_wind and self.janto.pais[0].is_wind()
    
    @property
    def churenpoton(self):
        if self.chiniso == False:
            return False
        all_nums = sorted([p.number for p in self.all_pais])
        return all_nums in CHUREN_NUMBERS_ALL_VALIATION

    @property
    def pinfu(self):
        return all([m.type == 'shuntsu' for m in self.mentsus]) and \
            self.machi == 'ryanmen' and \
            self.fanpai_fan(self.janto.pais[0]) == 0
    

    @property
    def ipeko(self):
        shuntsus = [m for m in self.mentsus if m.type == 'shuntsu']
        for i in range(len(shuntsus)):
            for j in range(i+1,len(shuntsus)):
                if shuntsus[i].pais[0].is_same_symbol(shuntsus[j].pais[0]):
                    return True
        return False

    @property
    def jikaze(self):
        return any([m.type in ['kotsu', 'kantsu'] and m.pais[0].str == self.hora_yakuinfo.jikaze for m in self.mentsus])

    @property
    def bakaze(self):
        return any([m.type in ['kotsu', 'kantsu'] and m.pais[0].str == self.hora_yakuinfo.bakaze for m in self.mentsus])

    @property
    def sanshokudojun(self):
        mentsu_types = ['shuntsu']
        same_types = [m for m in self.mentsus if m.type in mentsu_types]
        if len(same_types) < 3:
            return False

        for i in range(len(same_types)):
            m1 = same_types[i]
            target_start_number = m1.pais[0].number
            if any([m.pais[0].type == 'm' and m.pais[0].number == target_start_number for m in same_types]) and \
                any([m.pais[0].type == 'p' and m.pais[0].number == target_start_number for m in same_types]) and \
                any([m.pais[0].type == 's' and m.pais[0].number == target_start_number for m in same_types]):
                return True
        return False

    @property
    def sanshokudoko(self):
        mentsu_types = ['kotsu','kantsu']
        same_types = [m for m in self.mentsus if m.type in mentsu_types]
        if len(same_types) < 3:
            return False

        for i in range(len(same_types)):
            m1 = same_types[i]
            target_start_number = m1.pais[0].number
            if any([m.pais[0].type == 'm' and m.pais[0].number == target_start_number for m in same_types]) and \
                any([m.pais[0].type == 'p' and m.pais[0].number == target_start_number for m in same_types]) and \
                any([m.pais[0].type == 's' and m.pais[0].number == target_start_number for m in same_types]):
                return True
        return False

    @property
    def ikkitsukan(self):
        shuntsus = [m for m in self.mentsus if m.type == 'shuntsu']
        if len(shuntsus) < 3:
            return False

        for mps in ['m', 'p', 's']:
            target_shuntsus = [s for s in shuntsus if s.pais[0].type==mps]
            
            if any([s.pais[0].number == 1 for s in target_shuntsus]) and \
                any([s.pais[0].number == 4 for s in target_shuntsus]) and \
                any([s.pais[0].number == 7 for s in target_shuntsus]):
                return True
        return False

    @property
    def honchantaiyao(self):
        return all([any([p.is_yaochu() for p in m.pais]) for m in self.mentsus + [self.janto]])

    @property
    def shosangen(self):
        return self.num_sangenpais == 2 and self.janto.pais[0].is_sangenpai()

    @property
    def honiso(self):
        for mps in ['m', 'p', 's']:
            if all([p.type in [mps,'z'] for p in self.all_pais]):
                return True
        return False

    @property
    def junchantaiyao(self):
        return all([m.pais[0].type in ['m', 'p', 's'] and any([p.number in [1, 9] for p in m.pais]) for m in self.mentsus + [self.janto]])

    @property
    def ryanpeko(self):
        if all([m.type == 'shuntsu' for m in self.mentsus]) == False:
            return False
        for m1 in self.mentsus:
            if any([m2 is not m1 and m2.pais[0].is_same_symbol(m1.pais[0]) for m2 in self.mentsus]) == False:
                return False
        return True

    @property
    def chiniso(self):
        for mps in ['m','p','s']:
            if all([p.type == mps for p in self.all_pais]):
                return True
        return False
    
    @property
    def num_sangenpais(self):
        return len([m for m in self.mentsus if ((m.type in ['kotsu', 'kantsu']) and (m.pais[0].is_sangenpai()))])
    
    def fanpai_fan(self, pai):
        if pai.is_sangenpai():
            return 1
        fan = 0
        if pai.str == self.hora_yakuinfo.bakaze:
            fan += 1
        if pai.str == self.hora_yakuinfo.jikaze:
            fan += 1
        return fan
        
class Hora():

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
        chankan
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
        
        self.all_pais = self.free_pais + furos_flatten

        self.num_doras = Hora.count_doras(self.all_pais, self.doras)
        self.num_uradoras = Hora.count_doras(self.all_pais, self.uradoras)
        self.num_akadoras = len([p for p in self.all_pais if p.is_red])

        # print(f"num_akadoras:{self.num_akadoras}")
        num_same_as_taken = len([f for f in self.free_pais if self.taken.is_same_symbol(f)])
        self.shanten = ShantenAnalysis(self.free_pais, -1)
        self.candidates = []
        for c in self.shanten.combinations:
            for i in range(num_same_as_taken):
                self.candidates.append(Candidate(self, c, i))
        
        if len(self.candidates) > 0:
            self.best_candidate = max(self.candidates, key=lambda x:(x.fan, x.points))
        else:
            self.best_candidate = None
    
    @property
    def fu(self):
        return self.best_candidate.fu

    @property
    def fan(self):
        return self.best_candidate.fan
    
    
    @property
    def points(self):
        return self.best_candidate.points
    
    @property
    def yakus(self):
        return self.best_candidate.yakus

    @property
    def valid(self):
        return self.best_candidate.valid

    @property
    def oya_payment(self):
        return self.best_candidate.oya_payment

    @property
    def ko_payment(self):
        return self.best_candidate.ko_payment

    @classmethod
    def count_doras(cls, all_pais, target_doras):
        dora_sum = 0
        for td in target_doras:
            dora_sum += sum([1 for p in all_pais if Pai.is_same_symbol(td,p)])
        return dora_sum


    def __str__(self):
        return self.best_candidate.__str__()


