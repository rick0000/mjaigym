from typing import List, Dict
import pprint 
from mjaigym.board.function.pai import Pai, UNKNOWN_PAI
from mjaigym.board.function.mj_move import MjMove
from mjaigym.board.function.furo import Furo
import copy
from collections import namedtuple
from typing import NamedTuple


class BoardState():
    """current board information
    """
    def __init__(self, **state):
        self.state = state

    @property
    def valid(self)->bool:
        return len(self.state) > 0

    @property
    def previous_action(self)->Dict:
        """直前に行われたアクション。
        ドラ表示アクションは除外される。
        """
        return self.state['previous_without_dora_action']

    @property
    def possible_actions(self)->Dict[int,List[Dict]]:
        """各プレーヤーの行動可能なアクション一覧
        """
        return self.state['possible_actions']
    
    @property
    def tehais(self)->List[List[Pai]]:
        """各プレーヤーの手牌一覧
        """
        return self.state['tehais']

    @property
    def furos(self)->List[List[Furo]]:
        """各プレーヤーの副露一覧
        """
        return self.state['furos']

    @property
    def sutehais(self)->List[List[Pai]]:
        """各プレーヤーの捨て牌一覧
        副露されてもそのまま残る。
        """
        return self.state['sutehais']
    
    @property
    def dora_markers(self)->List[Pai]:
        """ドラ表示牌
        """
        return self.state['dora_markers']
    @property
    def bakaze(self)->str:
        """場風
        """
        return self.state['bakaze']
    @property
    def kyoku(self)->int:
        """局番号
        [1,2,3,4]
        """
        return self.state['kyoku']
    @property
    def honba(self)->int:
        """本場
        """
        return self.state['honba']
    @property
    def kyotaku(self)->int:
        """供託リーチ棒
        """
        return self.state['kyotaku']
    @property
    def scores(self)->List[int]:
        """各プレーヤーの持ち点
        """
        return self.state['scores']
    @property
    def jikaze(self)->List[str]:
        """各プレーヤーの自風
        """
        return self.state['jikaze']
    @property
    def reach(self)->List[bool]:
        """各プレーヤーがリーチしているか
        """
        return self.state['reach']
    @property
    def double_reach(self)->List[bool]:
        """各プレーヤーがダブルリーチしているか
        """
        return self.state['double_reach']
    @property
    def reach_sutehais_index(self)->List[int]:
        """各プレーヤーの立直宣言牌の捨て牌のindex
        """
        return self.state['reach_sutehais_index']
    
    @property
    def yama_rest_num(self)->int:
        """山の残りツモ回数
        """
        return self.state['yama_rest_num']
    @property
    def oya(self)->int:
        """親
        """
        return self.state['oya']
    @property
    def chicha(self)->int:
        """起親
        """
        return self.state['chicha']
    @property
    def ippatsu(self)->List[bool]:
        """各プレーヤーについて一発の役が発生する可能性があるか
        """
        return self.state['ippatsu']
    @property
    def rinshan(self)->List[bool]:
        """各プレーヤーについて嶺上開花の役が発生する可能性があるか
        """
        return self.state['rinshan']
    @property
    def haitei(self)->List[bool]:
        """各プレーヤーについてハイテイの役が発生する可能性があるか
        """
        return self.state['haitei']
    @property
    def first_turn(self)->bool:
        """1巡目かつ誰も副露していない状態か
        """
        return self.state['first_turn']
    @property
    def chankan(self)->List[bool]:
        """各プレーヤーについてチャンカンの役が発生する可能性があるか
        """
        return self.state['chankan']
    @property
    def anpais(self)->List[List[Pai]]:
        """各プレーヤーについての安全牌
        """
        return self.state['anpais']
    @property
    def red_dora_nums(self)->List[int]:
        """各プレーヤーの所持している赤牌の数。
        手牌の中にある赤牌を含む。
        """        
        return self.state['red_dora_nums']
    @property
    def furo_open_red_dora_nums(self)->List[int]:
        """各プレーヤーの所持している他から見えている赤牌の数
        手牌の中にある赤牌は含まない。
        """        
        return self.state['furo_open_red_dora_nums']
    @property
    def restpai_in_view(self)->List[List[int]]:
        """各プレーヤー視点から見た場合の残り牌枚数
        """
        return self.state['restpai_in_view']
        
        


    def __repr__(self):
        return pprint.pformat(self.state, width=180)

    def get_masked(self, view_point):
        """ある視点から見えない情報をマスクした状態のBoardStateを返す。
        
        Returns:
            BaordState:
        """
        masked = copy.copy(self.state)
        
        # mask tsumo
        if 'previous_action' in masked and \
                masked['previous_action']['type'] == MjMove.tsumo.value and\
                masked['previous_action']['actor'] != view_point:
            masked['previous_action'] = copy.copy(masked['previous_action'])
            masked['previous_action']['pai'] = UNKNOWN_PAI
        
        # mask tsumo
        if 'previous_without_dora_action' in masked and \
                masked['previous_without_dora_action']['type'] == MjMove.tsumo.value and\
                masked['previous_without_dora_action']['actor'] != view_point:
            masked['previous_without_dora_action'] = copy.copy(masked['previous_without_dora_action'])
            masked['previous_without_dora_action']['pai'] = UNKNOWN_PAI

        # create another instance to protect raw instance
        masked['possible_actions'] = copy.copy(masked['possible_actions'])
        masked['tehais'] = copy.copy(masked['tehais'])
        masked['red_dora_nums'] = copy.copy(masked['red_dora_nums'])
        masked['restpai_in_view'] = copy.copy(masked['restpai_in_view'])

        for i in range(4):
            if i != view_point:    
                masked['possible_actions'][i] = []
                masked['tehais'][i] = [UNKNOWN_PAI] * len(masked['tehais'][i])
                masked['red_dora_nums'][i] = masked['furo_open_red_dora_nums'][i]
                masked['restpai_in_view'][i] = [4] * 34

        return BoardState(**masked)

if __name__ == "__main__":
    from mjaigym.board import Board

    import random
    import pprint
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    board = Board(game_type="tonpu")
    board.reset()
    while board.is_end == False:
        # board.render_console(0)
        

        board_state = board.get_state()
        print(board_state)
        possible_actions = board.possible_actions
        
        
        actions = {}
        for i in range(board.PLAYER_NUM):
            sellected = random.choice(possible_actions[i])
            actions[i] = sellected
        
        board.step(actions)