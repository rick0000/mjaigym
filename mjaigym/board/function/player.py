from typing import List
import copy
import numpy as np
from mjaigym.board.function.mj_move import MjMove
from mjaigym.board.function.rs_shanten_analysis import RsShantenAnalysis
from mjaigym.board.function.tenpai_analysis import TenpaiAnalysis
from mjaigym.board.function.hora import Hora
from mjaigym.board.function.pai import Pai, UNKNOWN_PAI, UNKNOWN_PAI_STR
from mjaigym.board.function.furo import Furo
import itertools
from collections import defaultdict

class Player():
    def __init__(self, board, seat=None):
        self.id = seat
        self.board = board
        self.reset()
        self.shanten_analysis = RsShantenAnalysis()

    def reset(self):
        self.score = 25000
        self.tehais = []
        self.furos = []
        self.ho = [] # 鳴かれた牌を含まない
        self.sutehais = [] # 鳴かれた牌を含む
        self.extra_anpais = [] # sutehais以外のこのプレーヤーに対する安牌
        self.reach_state = "none"
        self.reach_ho_index = None
        self.reach_sutehais_index = None
        self.double_reach = False
        self.pao_for_id = None
        self.ippatsu_chance = False
        self.rinshan = False


    @property
    def str_sep_tehais(self):
        return [t.str for t in self.tehais]

    @property
    def red_dora_num(self):
        num = 0
        num += len([t for t in self.tehais if t.is_red])
        for furo in self.furos:
            num += len([p for p in furo.pais if p.is_red])
            
        return num

    @property
    def furo_open_red_dora_num(self):
        num = 0
        for furo in self.furos:
            num += len([p for p in furo.pais if p.is_red])    
        return num


    @property
    def str_tehais(self):
        # add hidden tehai
        type_history = set()
        show_str_reverse = []
        
        if (not self.tehais) or len(self.tehais) == 0:
            return ''
        
        if len(self.tehais) % 3 == 2:
            # tsumo timing
            show_str_reverse.append(self.tehais[-1].str)
            show_str_reverse.append("  ")
            loop_tehais = self.tehais[:-1]
        else:
            loop_tehais = self.tehais
            
        for t in reversed(loop_tehais):
            if t.is_number() and t.type in type_history:
                if t.is_red:
                    show_str_reverse.append(t.str[0]+t.str[2])
                else:
                    show_str_reverse.append(t.str[0])

            else:
                type_history.add(t.type)
                show_str_reverse.append(t.str)
        

        # add furo        

        return ''.join(reversed(show_str_reverse))
        
    @property
    def anpais(self):
        return list(sorted(set(self.sutehais + self.extra_anpais)))

    @property
    def reach(self):
        return self.reach_state == "accepted"
    
    @property
    def menzen(self):
        return len([f for f in self.furos if f.type != MjMove.ankan.value]) == 0

    def update_state(self, action):
        if self.board.previous_action is not None and \
                self.board.previous_action['type'] in [MjMove.dahai.value, MjMove.kakan.value] and \
                'actor' in self.board.previous_action and \
                self.board.previous_action['actor'] != self.id and \
                action['type'] != MjMove.hora.value:
            
            self.extra_anpais.append(Pai.from_str(self.board.previous_action['pai']))





        action_type = action['type']

        if action_type == MjMove.start_game.value:
            if not self.id and 'id' in action:
                self.id = action['id']
                self.name = f"player{self.id}"
            if 'names' in action:
                self.name = action['names'][self.id]
            self.score = 25000
            self.tehais = []
            self.furos = []
            self.ho = []
            self.sutehais = []
            self.extra_anpais = []
            self.reach_state = None
            self.reach_ho_index = None
            self.reach_sutehais_index = None
            self.double_reach = False
            self.ippatsu_chance = False
            self.pao_for_id = None
            self.rinshan = False

        elif action_type == MjMove.start_kyoku.value:
            self.tehais = sorted(Pai.from_list(action['tehais'][self.id]))
            
            self.furos = []
            self.ho = []
            self.sutehais = []
            self.extra_anpais = []
            self.reach_state = "none"
            self.reach_ho_index  = None
            self.reach_sutehais_index = None
            self.double_reach = False
            self.ippatsu_chance = False
            self.pao_for_id = None
            self.rinshan = False

        elif action_type in [
            MjMove.chi.value,
            MjMove.pon.value, 
            MjMove.daiminkan.value, 
            MjMove.ankan.value
            ]:
            self.ippatsu_chance = False
        elif action_type == MjMove.tsumo.value:
            if self.board.previous_action['type'] == MjMove.kakan.value:
                self.ippatsu_chance = False


        if 'actor' in action and action['actor'] == self.id:
            if action_type == MjMove.tsumo.value:
                pai = Pai.from_str(action['pai'])
                self.tehais = sorted(self.tehais)
                self.tehais.append(pai)
            elif action_type == MjMove.dahai.value:
                pai = Pai.from_str(action['pai'])
                self.delete_tehai(pai)
                self.tehais = sorted(self.tehais)
                self.ho.append(pai)
                self.sutehais.append(pai)
                self.ippatsu_chance = False
                self.rinshan = False
                if self.reach == False:
                    self.extra_anpais.clear()
            elif action_type in [
                MjMove.chi.value, 
                MjMove.pon.value, 
                MjMove.daiminkan.value, 
                MjMove.ankan.value
                ]:

                consumed_pais = Pai.from_list(action['consumed'])
                for c in consumed_pais:
                    self.delete_tehai(c)

                furo = {
                        'type':action['type'],
                        'consumed':consumed_pais,   
                    }
                if action_type != MjMove.ankan.value:
                    pai = Pai.from_str(action['pai'])
                    furo['taken'] = pai

                if action_type == MjMove.chi.value:                    
                    furo['pai_id'] = min(
                        Pai.str_to_id(action['pai']),
                        min(
                            Pai.str_to_id(action['consumed'][0]), 
                            Pai.str_to_id(action['consumed'][1]))
                    )
                else:
                    furo['pai_id'] = Pai.str_to_id(action['consumed'][0])

                if action_type == MjMove.ankan.value:
                    furo['target'] = self.id
                else:
                    furo['target'] = action['target']

                self.furos.append(Furo(furo))
                if action_type in [MjMove.daiminkan.value, MjMove.ankan.value]:
                    self.rinshan = True

                # pao
                if action_type in [MjMove.daiminkan.value, MjMove.pon.value]:
                    pai = Pai.from_str(action['pai'])
                    if pai.is_sangenpai():
                        if self.is_daisangen_pao():
                            self.pao_for_id = action['target']
                    elif pai.is_wind():
                        if self.is_daisushi_pao():
                            self.pao_for_id = action['target']
            
            elif action_type == MjMove.kakan.value:
                pai = Pai.from_str(action['pai'])
                self.delete_tehai(pai)
                pon_index = -1
                for i,f in enumerate(self.furos):
                    if f.type == MjMove.pon.value and pai.is_same_symbol(f.taken):
                        pon_index = i
                if pon_index == -1:
                    raise Exception('not have same symbole pon')
                self.furos[pon_index] = Furo({
                    'type':MjMove.kakan.value,
                    'taken':self.furos[pon_index].taken,
                    'consumed':self.furos[pon_index].consumed + [pai],
                    'target':self.furos[pon_index].target,
                    'pai_id':self.furos[pon_index].pai_id,
                })
                self.rinshan = True
            
            elif action_type == MjMove.reach.value:
                self.reach_state = 'declared'
                self.double_reach = self.board.first_turn
            
            elif action_type == MjMove.reach_accepted.value:
                self.reach_state = 'accepted'
                self.reach_ho_index = len(self.ho)-1
                self.reach_sutehais_index = len(self.sutehais)-1
                self.ippatsu_chance = True
                
        
        if 'target' in action and action['target'] == self.id:
            pai = Pai.from_str(action['pai'])
            if action_type in [
                MjMove.pon.value,
                MjMove.chi.value, 
                MjMove.daiminkan.value
                ]:
                taken = self.ho.pop()
                # assert taken == pai
        
        if 'scores' in action:
            self.score = action['scores'][self.id]

        

    @property
    def jikaze(self):
        if self.board.oya is not None:
            wind_index = (4 + self.id - self.board.oya) % 4
            return ['E','S','W','N'][wind_index]
        else:
            return None
    
    @property
    def tenpai(self):
        return self.shanten <= 0

    @property
    def furiten(self):
        if len(self.tehais) % 3 != 1:
            return False
        if UNKNOWN_PAI in self.tehais:
            return False
        
        tenpai_info = TenpaiAnalysis(self.tehais)
        
        if tenpai_info.tenpai == False:
            return False
        
        anpais = self.anpais
        return any([a.is_same_symbol(b) for (a,b) in itertools.product(anpais, tenpai_info.waited_pais)])

    def is_daisangen_pao(self):
        return len([f for f in self.furos if f.pais[0].is_sangenpai()]) == 3

    def is_daisushi_pao(self):
        return len([f for f in self.furos if f.pais[0].is_wind()]) == 4


    def delete_tehai(self, pai):
        if pai in self.tehais:
            pai_index = self.tehais.index(pai)
        else:
            pai_index = 0
            # assert self.tehais[pai_index].str == UNKNOWN_PAI_STR
        del self.tehais[pai_index]
            

    def get_chi_dahai(self, chi_action):
        dahai_dic = {}
        chi_pai = Pai.from_str(chi_action['pai'])
        chi_consumed = Pai.from_list(chi_action['consumed'])

        upper_kuikae_exists = False
        lower_kuikae_exists = False
        if chi_consumed[0].number + 1 == chi_consumed[1].number:
            if chi_pai.number + 1 == chi_consumed[0].number and chi_pai.number != 7:
                upper_kuikae_exists = True
            elif chi_pai.number - 2 == chi_consumed[0].number and chi_pai.number != 3:
                lower_kuikae_exists = True
            
        for tehai in self.tehais:
            if tehai.str in dahai_dic:
                continue
            if chi_pai.is_same_symbol(tehai):
                continue
            if upper_kuikae_exists and \
                tehai.type == chi_pai.type and \
                tehai.number - 3 == chi_pai.number:
                    continue
            if lower_kuikae_exists and \
                tehai.type == chi_pai.type and \
                tehai.number + 3 == chi_pai.number:
                    continue
            
            dahai_dic[tehai.str] = ''
        return [d for d in dahai_dic.keys()]


            




    def __str__(self):        
        return f"tehai:{self.tehais}, furo:{self.furos}\n" \
            f"sutehai:{self.sutehais}\n" \
            f"score:{self.score}, reache:{self.reach}"
    
    @property
    def shanten(self):
        tehai = [0] * 34
        for t in self.tehais:
            tehai[t.id] += 1
        return self.shanten_analysis.calc_shanten(tehai, len(self.furos))
    
    
    def calc_dahaied_shanten(self, pai:str):
        pai = Pai.from_str(pai)
        if pai not in self.tehais:
            raise Exception(f'tehais not condains pai:{pai}')
        dahaied_tehai = self.tehais.copy()
        dahaied_tehai.remove(pai)

        tehai = [0] * 34
        for t in dahaied_tehai:
            tehai[t.id] += 1
        return self.shanten_analysis.calc_shanten(tehai, len(self.furos))
    
    def calc_added_shanten(self, pai:str):
        pai = Pai.from_str(pai)
        dahaied_tehai = self.tehais.copy()
        dahaied_tehai.append(pai)

        tehai = [0] * 34
        for t in dahaied_tehai:
            tehai[t.id] += 1
        return self.shanten_analysis.calc_shanten(tehai, len(self.furos))
        

    def can_hora(self):
        previous_action = self.board.previous_action
        if not previous_action:
            return False

        if previous_action['type'] == MjMove.tsumo.value and self.id == previous_action['actor']:
            hora_type = 'tsumo'
            pais = self.tehais
            shanten = self.shanten
        elif previous_action['type'] in [MjMove.dahai.value, MjMove.kakan.value] and self.id != previous_action['actor']:
            hora_type = 'ron'
            pais = self.tehais + [Pai.from_str(previous_action['pai'])]
            shanten = self.calc_added_shanten(previous_action['pai'])
        else:
            return False

        if shanten != -1:
            return False

        action = {
            'type':MjMove.hora.value,
            'pai':previous_action['pai'],
            'actor':self.id,
            'target':previous_action['actor'],    
        }
        hora = self.board.get_hora(action, **{'previous_action':previous_action})
        
        return hora.valid and (hora_type == 'tsumo' or self.furiten == False)
    


    def can_daiminkan(self, pai:str):
        if self.reach_state == "accepted":
            return False, []
        
        target_pai = Pai.from_str(pai)
        candidates = [t.str for t in self.tehais if target_pai.is_same_symbol(t)]
        if len(candidates) < 3:
            return False, []
        else:
            return True, candidates

    def can_ankan(self, pai:str=None):
        if pai is None:
            pai_count = {}
            for t in self.tehais:
                if t.id not in pai_count:
                    pai_count[t.id] = 0
                
                pai_count[t.id] += 1

            can_ankan_ids = [p for p in pai_count if pai_count[p] == 4]
            if len(can_ankan_ids) == 0:
                return False, []
            
            candidates = []
            for can_ankan_id in can_ankan_ids:
                candidate = [p.str for p in self.tehais if p.id == can_ankan_id]
                candidates.append(candidate)
            
            return True, candidates

        else:
            in_tehai = [p for p in self.tehais if p.is_same_symbol(Pai.from_str(pai))]
            return len(in_tehai) == 4, in_tehai

    def can_kakan(self, pai:str=None):
        if self.reach_state == "accepted":
            return False, []

        if pai is None:
            pons = {}
            for f in self.furos:
                if f.type == 'pon':
                    pons[f.pais[0].id] = [p.str for p in f.pais]

            can_kakan_ids = [p.id for p in self.tehais if p.id in pons]
            if len(can_kakan_ids) == 0:
                return False, []
            
            candidates = []
            for can_kakan_id in can_kakan_ids:
                candidate = [p.str for p in self.tehais if p.id == can_kakan_id]
                # assert len(candidate) == 1
                candidates.append([candidate[0], pons[can_kakan_id]])
            
            return True, candidates
        else:
            target_pai = Pai.from_str(pai)
            in_tehai = [p for p in self.tehais if p.is_same_symbol(target_pai)]
            in_pon = [[p.str for p in f.pais] for f in self.furos if f.type == 'pon' and f.pais[0].is_same_symbol(target_pai)]
            return len(in_tehai)==1 & len(in_pon)==1, [in_tehai[0], in_pon]


    def can_pon(self, pai:str):
        if self.reach_state == "accepted":
            return False, []

        target_pai = Pai.from_str(pai)
        candidates_pai = [t for t in self.tehais if target_pai.is_same_symbol(t)]
        
        if len(candidates_pai) < 2:
            return False, []
        
        red_consumed = ([c for c in candidates_pai if c.is_red])
        not_red_consumed = ([c for c in candidates_pai if c.is_red == False])
        if len(red_consumed) > 0:
            if  len(candidates_pai) == 2:
                return True, [
                    [red_consumed[0].str,     not_red_consumed[0].str],
                    ]
            else:
                return True, [
                    [red_consumed[0].str,     not_red_consumed[0].str],
                    [not_red_consumed[0].str, not_red_consumed[0].str],
                    ]
        else:
            return True, [
                [not_red_consumed[0].str, not_red_consumed[1].str]
                ]
    
    def can_chi(self, pai:str):
        if self.reach_state == "accepted":
            return False, []
        

        target_pai = Pai.from_str(pai)
        if not target_pai.is_number():
            return False, []

        
        can_chi = False
        candidates = []
        
        tehais_single = {}
        for t in self.tehais:
            if t.str not in tehais_single:
                tehais_single[t.str] = t

        single_tehais = tehais_single.values()

        minus2_candidates_pai = [t.str for t in single_tehais if t.number == target_pai.number-2  and t.type == target_pai.type]
        minus1_candidates_pai = [t.str for t in single_tehais if t.number == target_pai.number-1  and t.type == target_pai.type]
        plus1_candidates_pai = [t.str for t in single_tehais if t.number == target_pai.number+1  and t.type == target_pai.type]
        plus2_candidates_pai = [t.str for t in single_tehais if t.number == target_pai.number+2  and t.type == target_pai.type]
        
        cannot_dahai_number = []
        # right chi [1,2] [3]
        if len(minus2_candidates_pai) > 0 and len(minus1_candidates_pai) > 0:
            cannot_dahai_number.append(target_pai.number-3)
            cannot_dahai_number.append(target_pai.number)
            for m2, m1 in itertools.product(minus2_candidates_pai, minus1_candidates_pai):
                candidates.append([m2, m1])
                
            
        # center chi [1,3] [2]
        if len(minus1_candidates_pai) > 0 and len(plus1_candidates_pai) > 0:
            cannot_dahai_number.append(target_pai.number)
            for m1, p1 in itertools.product(minus1_candidates_pai, plus1_candidates_pai):
                candidates.append([m1, p1])
        # left chi [2,3] [1]        
        if len(plus1_candidates_pai) > 0 and len(plus2_candidates_pai) > 0:
            cannot_dahai_number.append(target_pai.number)
            cannot_dahai_number.append(target_pai.number+3)
            for p1, p2 in itertools.product(plus1_candidates_pai, plus2_candidates_pai):
                candidates.append([p1, p2])


        # kuikae check
        ignore_candidates = []
        if all([t for t in self.tehais if t.type == target_pai.type]):
            for candidate in candidates:
                temp_tehai = copy.copy(self.tehais)
                temp_tehai.remove(Pai.from_str(candidate[0]))
                temp_tehai.remove(Pai.from_str(candidate[1]))
                if len([t for t in temp_tehai if t.number not in cannot_dahai_number]) == 0:
                    ignore_candidates.append(candidate)
        
        for ignore_candidate in ignore_candidates:
            candidates.remove(ignore_candidate)

        return len(candidates) > 0, candidates

if __name__ == "__main__":
    from mjaigym.board.function.board import Board
    b = Board(0)
    p = Player(b, 0)
    print(p)



