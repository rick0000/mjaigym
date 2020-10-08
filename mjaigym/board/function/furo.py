from mjaigym.board.function.mentsu import Mentsu
from mjaigym.board.function.mj_move import MjMove

FUROTYPE_TO_MENTSU_TYPE = {
    MjMove.chi.value:'shuntsu',
    MjMove.pon.value:'kotsu',
    MjMove.daiminkan.value:'kantsu',
    MjMove.kakan.value:'kantsu',
    MjMove.ankan.value:'kantsu',
}


class Furo():
    """ represents furo information

    type:str    furo type
    target:int  furo target player
    taken:Pai   furo target pai
    consumed:List[Pai]  opend from tehai pais
    pai_id:int  min pai id used in target and consumed
    """
    def __init__(self, action):
        self.type = action['type']
        if self.type != 'ankan':
            self.taken = action['taken']
        else:
            self.taken = None
            
        self.consumed = action['consumed']
        self.target = action['target']
        if "pai_id" in action:
            self.pai_id = action['pai_id']
        else:
            self.pai_id = min([p.id for p in self.pais])
        
        self._hash = 0
        for i, pai in enumerate(self.pais):
            self._hash += (34**i + pai.id)
            if pai.is_red:
                self._hash += 34 ** 4

    @property
    def is_kan(self):
        return FUROTYPE_TO_MENTSU_TYPE[self.type] == 'kantsu'

    @property
    def pais(self):
        pais = ([self.taken] if self.taken else []) + self.consumed
        return pais
    
    def to_mentsu(self):
        visibility = 'an' if self.type == MjMove.ankan.value else 'min'
        return Mentsu(**{
            'type':FUROTYPE_TO_MENTSU_TYPE[self.type],
            'pais':self.pais,
            'visibility':visibility,
            })
    
    def to_rs_hora_furo(self):
        if self.type == MjMove.ankan.value:
            return {
                "type":self.type,
                "consumed":[c.str for c in self.consumed],
            }
        else:
            return {
                "type":self.type,
                "taken":self.taken.str,
                "consumed":[c.str for c in self.consumed],
            }


    def __str__(self):
        if self.type == MjMove.ankan.value:
            return f"[# {self.consumed[0]} {self.consumed[1]} #]"
        else:
            consumed = " ".join([c.str for c in self.consumed])
            return f"[{self.taken}^({self.target}) {consumed}]"

    def __repr__(self):
        return self.__dict__.__repr__()

    def __eq__(self, other):
        # if not isinstance(other, Pai):
        #     return False
        return self.type == other.type and\
            self.target == other.target and\
            self.taken == other.taken and\
            self.consumed == other.consumed


    def __hash__(self):
        return self._hash

    def __lt__(self, other):
        return self.pai_id < other.pai_id