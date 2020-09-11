from mjaigym.board.function.yama import Yama
from mjaigym.board.function.player import Player
from mjaigym.board.function.mj_move import MjMove
from mjaigym.board.function.pai import Pai
from mjaigym.board.function.hora import Hora
from mjaigym.board.board_state import BoardState
import numpy as np
import os
import pprint
from typing import List
import copy


class Board(object):
    '''
    class for mahjong state
    '''
    PLAYER_NUM = 4

    def __init__(
            self, 
            game_type='one_kyoku', 
            seed=np.random.randint(low=0, high=os.sys.maxsize),
            names=None,
            dealer_message_callback=None,
        ):
        self._raw_seed = seed
        self.pending_message = []
        if game_type not in ['one_kyoku', 'tonpu', 'tonnan']:
            raise Exception('Unknown game_type')
        self.game_type = game_type
        self.dealer_message_callback = dealer_message_callback
        if names:
            self.names = names
        else:
            self.names = ["player0","player1","player2","player3"]
        self.players = [Player(self, i) for i in range(self.PLAYER_NUM)]
        self.yama = None
        self.rest_pai_view = [[4]*34 for _ in range(4)]
        
        self.reset()
    

    def restnum_update(self, action):
        if action is None or "type" not in action:
            return

        if action["type"] == MjMove.start_kyoku.value:
            # initialize
            self.rest_pai_view = [[4]*34 for _ in range(4)]

            dora_marker_id = Pai.str_to_id(action["dora_marker"])
            for i in range(4):
                self.rest_pai_view[i][dora_marker_id] -= 1
                for pai in action["tehais"][i]:
                    pai_id = Pai.str_to_id(pai)
                    self.rest_pai_view[i][pai_id] -= 1

        elif action["type"] == MjMove.tsumo.value:
            pai_id = Pai.from_str(action["pai"]).id
            actor = action["actor"]
            self.rest_pai_view[actor][pai_id] -= 1

        elif action["type"] == MjMove.dahai.value:
            pai_id = Pai.from_str(action["pai"]).id
            actor = action["actor"]
            for i in range(4):
                if actor != i:
                    self.rest_pai_view[i][pai_id] -= 1
        
        elif action["type"] == MjMove.dora.value:
            pai_id = Pai.from_str(action["dora_marker"]).id
            for i in range(4):
                self.rest_pai_view[i][pai_id] -= 1
        
        elif action["type"] == MjMove.chi.value:
            actor = action["actor"]
            for i in range(4):
                if actor != i:
                    for consume in action["consumed"]:
                        self.rest_pai_view[i][Pai.str_to_id(consume)] -= 1
        
        elif action["type"] == MjMove.pon.value:
            actor = action["actor"]
            for i in range(4):
                if actor != i:
                    for consume in action["consumed"]:
                        self.rest_pai_view[i][Pai.str_to_id(consume)] -= 1
                
        elif action["type"] == MjMove.daiminkan.value:
            actor = action["actor"]
            for i in range(4):
                if actor != i:
                    for consume in action["consumed"]:
                        self.rest_pai_view[i][Pai.str_to_id(consume)] -= 1
        
        elif action["type"] == MjMove.ankan.value:
            actor = action["actor"]
            for i in range(4):
                if actor != i:
                    for consume in action["consumed"]:
                        self.rest_pai_view[i][Pai.str_to_id(consume)] -= 1

        elif action["type"] == MjMove.kakan.value:
            actor = action["actor"]
            pai_id = Pai.str_to_id(action["pai"])
            for i in range(4):
                if actor != i:
                    self.rest_pai_view[i][pai_id] -= 1
        
        # # assert all([all([r>=0 for r in rest]) for rest in self.rest_pai_view])
        



    def get_state(self)->BoardState:
        return BoardState(**{
            "previous_action":self.previous_action,
            "previous_without_dora_action":self.previous_without_dora_action,
            "possible_actions":self.possible_actions,
            "tehais":[copy.copy(p.tehais) for p in self.players],
            "furos":[copy.copy(p.furos) for p in self.players],
            "sutehais":[copy.copy(p.sutehais) for p in self.players],
            "dora_markers":copy.copy(self.dora_markers),
            "bakaze":self.bakaze,
            "kyoku":self.kyoku,
            "honba":self.honba,
            "kyotaku":self.kyotaku,
            "scores":[p.score for p in self.players],
            "jikaze":[p.jikaze for p in self.players],
            "reach":[p.reach for p in self.players],
            "double_reach":[p.double_reach for p in self.players],
            "reach_sutehais_index":[p.reach_sutehais_index for p in self.players],
            "yama_rest_num":self.yama.get_rest_num() if self.yama else None,
            "oya":self.oya,
            "chicha":self.chicha,
            "ippatsu":[p.ippatsu_chance for p in self.players],
            "rinshan":[p.rinshan for p in self.players],
            "haitei":[p.rinshan == False and self.yama and (self.yama.get_rest_num() == 0) for p in self.players],
            "first_turn":self.first_turn,
            "chankan":['actor' in self.previous_without_dora_action and p.id != self.previous_without_dora_action['actor'] and self.previous_without_dora_action['type']==MjMove.kakan.value for p in self.players],
            "anpais":[copy.copy(p.anpais) for p in self.players],
            "red_dora_nums":[p.red_dora_num for p in self.players],
            "furo_open_red_dora_nums":[p.furo_open_red_dora_num for p in self.players],
            "restpai_in_view":[copy.copy(view) for view in self.rest_pai_view],
        })


        
    def reset(self):
        
        self._current_seed = self._raw_seed
        
        self.dealer_history = []
        self.renponse_history = []
        self.chicha = None
        self.last = False


        self.kyoku = 1
        self.bakaze = 'E'
        self.honba = 0
        self.kyotaku = 0
        self.chicha = np.random.randint(0,4)
        self.oya = (self.chicha + self.kyoku - 1) % self.PLAYER_NUM
        
        self.dora_markers = []


        self.kandora_pending = False
        self.rinshan_pending = False
        self.reach_pending = False
        self.next_reach_pending = False

        self.first_turn = False

        self.rest_pai_view = [[4]*34 for _ in range(4)]

        start_geme_message = {
            'type':'start_game',
            'seed':self._current_seed,
            'names':self.names,
        }
        self.dealer_history.append(start_geme_message)


    def step(self, responses):
        ''' apply response and go next state
        '''
        # レスポンスをログに追加
        self.renponse_history.append(copy.copy(responses))
        
        # 4人のレスポンスから最も重要度の高いアクションを抽出
        apply_responses = Board.get_highest_prior_response(responses)
        # 和了のみ複数同時処理するため特別に扱う。
        if all([a['type'] == MjMove.hora.value for a in apply_responses]):
            hora_results = self._do_hora(apply_responses)
            
            if len(hora_results) > 1:
                for action in hora_results[1:]:
                    self.pending_message.append(action)

            this_time_handle_hora = hora_results[0]
            self.dealer_history.append(this_time_handle_hora)
            self.on_action(this_time_handle_hora)
            return this_time_handle_hora
        
        # handling reach_accept
        if self.reach_pending and \
                (self.previous_action['type'] == MjMove.dahai.value) and \
                (any([r['type'] == MjMove.hora.value for r in apply_responses]) == False):
            action = self._do_reach_accept()
            self.dealer_history.append(action)
            self.next_reach_pending = True
            self.next_apply_responses = apply_responses
            self.reach_pending = False
            self.on_action(action)
            return action


        if self.next_reach_pending:
            self.next_reach_pending = False
            apply_responses = self.next_apply_responses

        # assert len(apply_responses) == 1
        apply_response = apply_responses[0]
        dealer_message = self._dealer_step(apply_response)
        if dealer_message is not None:
            self.dealer_history.append(dealer_message)
            self.on_action(dealer_message)
        
        self.restnum_update(dealer_message)
        return dealer_message



    def on_action(self, action):
        '''
        アクション実行後のコールバック
        '''
        pass
        if self.dealer_message_callback:
            self.dealer_message_callback(action)
        '''
        if action['type'] == 'daiminkan':
            print('daiminkan !', action)
        
        if action['type'] == MjMove.reach.value:
            self.render_console(action['actor'])
        '''
        
    def consume_pending_message(self):
        return self.pending_message.pop(0)


    def _dealer_step(self, response):
        '''
        一つ前のアクションとレスポンスを元に次の状態に移行する。
        '''
        response_type = response['type']

        #print(f"dealer step  prev_type:{prev_type}, response_type:{response_type}")
        # レスポンスがnone以外の場合はそのレスポンスを適用する。
        if response_type != MjMove.none.value:
            if response_type == MjMove.dahai.value:
                dealer_message = self._do_dahai(response)
            elif response_type == MjMove.pon.value:
                dealer_message = self._do_pon(response)
            elif response_type == MjMove.chi.value:
                dealer_message = self._do_chi(response)
            elif response_type == MjMove.daiminkan.value:
                dealer_message = self._do_daiminkan(response)
            elif response_type == MjMove.ankan.value:
                dealer_message = self._do_ankan(response)
            elif response_type == MjMove.kakan.value:
                dealer_message = self._do_kakan(response)
            elif response_type == MjMove.reach.value:
                dealer_message = self._do_reach(response)
            elif response_type == MjMove.hora.value:
                # 別の箇所で扱っている
                raise Exception('not intended path')
            elif response_type == MjMove.ryukyoku.value:
                dealer_message = self._do_ryukyoku(response)
            else:
                raise Exception('not intended path')

            return dealer_message

        # レスポンスがnoneの場合は次のアクションを生成して適用する。
        prev_type = self.previous_action['type']
        if prev_type == MjMove.start_game.value:
            dealer_message = self._do_start_kyoku()
        elif prev_type == MjMove.start_kyoku.value:
            dealer_message = self._do_tsumo()

        elif prev_type == MjMove.dahai.value:
            # assert self.yama.get_rest_num() >= 0 
            if self.yama.get_rest_num() == 0:
                dealer_message = self._do_fanpai_ryukyoku()
            elif self.kandora_pending:
                dealer_message = self._do_opendora()
                self.kandora_pending = False
            else:
                dealer_message = self._do_tsumo()

        elif prev_type == MjMove.reach_accepted.value:
            dealer_message = self._do_tsumo()

        elif prev_type == MjMove.ankan.value:
            dealer_message = self._do_opendora()
            self.rinshan_pending = True

        elif prev_type in [MjMove.daiminkan.value, MjMove.kakan.value]:
            dealer_message = self._do_tsumo()
            self.kandora_pending = True

        elif prev_type == MjMove.dora.value:
            if self.rinshan_pending:
                dealer_message = self._do_tsumo()
                self.rinshan_pending = False
            else:
                dealer_message = self._do_tsumo()

        elif prev_type == MjMove.hora.value:
            dealer_message = self._do_end_kyoku()
        elif prev_type == MjMove.ryukyoku.value:
            dealer_message = self._do_end_kyoku()
        elif prev_type == MjMove.end_kyoku.value:
            if self.last or self.game_type == 'one_kyoku':
                dealer_message = self._do_end_game()
            else:
                dealer_message = self._do_start_kyoku()
        elif prev_type == MjMove.end_game.value:
            dealer_message = None
        else:
            for line in self.dealer_history:
                print(line)
            print(response)
            raise Exception('not intended path')


        return dealer_message
            
 
    def need_action_select(self):
        for actions in self.possible_actions.values():
            if len(actions) != 1:
                return False
        return True

    @classmethod
    def get_highest_prior_response(cls, responses):
        '''
        4人のレスポンスから最も優先度が高いレスポンスを返す
        '''
        # assert type(responses) == dict
        prior_order = [
            MjMove.hora, MjMove.daiminkan, MjMove.pon, MjMove.chi,
            MjMove.dahai, MjMove.ankan, MjMove.kakan, MjMove.reach,
            MjMove.ryukyoku, MjMove.none
        ]
        
        for move_type in prior_order:    
            moves = [r for r in responses.values() if ('type' in r) and (r['type'] == move_type.value)]
            if len(moves) > 0:
                if move_type == MjMove.none:
                    return moves[0:1]
                else:
                    return moves
        raise Exception(f'not intended response type:{responses}')
        

    @property
    def possible_actions(self):
        # calclate valid move
        if self.previous_action is None or \
            self.previous_action['type'] in [
                MjMove.start_game.value,
                MjMove.start_kyoku.value,
                
                MjMove.reach_accepted.value,
                MjMove.ryukyoku.value,
                
                MjMove.end_kyoku.value,
                MjMove.end_game.value,
            ]:
            # pattern of all player returns none
            return self.get_all_none_response()
        
        elif self.previous_action['type'] == MjMove.tsumo.value:
            # tsumo next actions
            
            tsumo_actor = self.previous_action['actor']
            tehais = self.players[tsumo_actor].tehais
            actions = []
            
            if self.players[tsumo_actor].reach or self.players[tsumo_actor].double_reach:
                # add dahai
                action = {
                    'type':MjMove.dahai.value,
                    'actor':tsumo_actor,
                    'pai':self.previous_action['pai'],
                    'tsumogiri':True,
                }
                actions.append(action)
            else:
                dahais_dic = {}
                for i,tehai in enumerate(tehais[:-1]):
                    # remove duplicate
                    if tehai.str in dahais_dic:
                        continue
                    dahais_dic[tehai.str] = ''
                    action = {
                        'type':MjMove.dahai.value,
                        'actor':tsumo_actor,
                        'pai':tehai.str,
                        'tsumogiri':False,
                    }
                    actions.append(action)
                # add tsumogiri
                action = {
                    'type':MjMove.dahai.value,
                    'actor':tsumo_actor,
                    'pai':tehais[-1].str, # tsumo pai is in last index.
                    'tsumogiri':True, 
                }
                actions.append(action)

            # add reach
            if self.players[tsumo_actor].reach_state == 'none' and \
                self.players[tsumo_actor].menzen and \
                self.players[tsumo_actor].shanten <= 0 and \
                self.yama.get_rest_num() >= 1:
                action = {
                    'type':MjMove.reach.value,
                    'actor':tsumo_actor,
                }
                actions.append(action)
            # add ankan
            can_ankan, consumeds = self.players[tsumo_actor].can_ankan()
            yama_can_kan = self.yama.get_rest_num() > 0
            if can_ankan and yama_can_kan:
                for consumed in consumeds:
                    action = {
                        'type':MjMove.ankan.value,
                        'actor':tsumo_actor,
                        'consumed':consumed,
                    }
                    actions.append(action)

            # add kakan
            can_kakan, pai_consumeds = self.players[tsumo_actor].can_kakan()
            if can_kakan and yama_can_kan:
                for pai_consumed in pai_consumeds:
                    action = {
                        'type':MjMove.kakan.value,
                        'actor':tsumo_actor,
                        'pai':pai_consumed[0],
                        'consumed':pai_consumed[1],
                    }
                    actions.append(action)

            # add hora
            can_hora = self.players[tsumo_actor].can_hora()
            if can_hora:
                previous_actor = self.previous_action['actor']
                previous_pai = self.previous_action['pai']
                action = {
                    'type':MjMove.hora.value,
                    'actor':tsumo_actor,
                    'target':previous_actor,
                    'pai':previous_pai,
                }
                actions.append(action)

            response = self.get_all_none_response()
            response[tsumo_actor] = actions # not use none
            return response

        elif self.previous_action['type'] == MjMove.reach.value:
            tsumo_actor = self.previous_action['actor']
            tehais = self.players[tsumo_actor].tehais
            actions = []
            
            dahais_dic = {}
            for i,tehai in enumerate(tehais[:-1]):
                # remove duplicate
                if tehai.str in dahais_dic:
                    continue
                dahais_dic[tehai.str] = ''
                action = {
                    'type':MjMove.dahai.value,
                    'actor':tsumo_actor,
                    'pai':tehai.str,
                    'tsumogiri':False,
                }
                actions.append(action)
            # add tsumogiri
            action = {
                'type':MjMove.dahai.value,
                'actor':tsumo_actor,
                'pai':tehais[-1].str, # tsumo pai is in last index.
                'tsumogiri':True, 
            }
            actions.append(action)

            tenpai_actions = []
            for a in actions:
                if self.players[tsumo_actor].calc_dahaied_shanten(a['pai']) <= 0:
                    tenpai_actions.append(a)
            # assert len(tenpai_actions) > 0
                
            response = self.get_all_none_response()
            response[tsumo_actor] = tenpai_actions # not use none
            return response



        elif self.previous_action['type'] == MjMove.dahai.value:
            # check ryukyoku
            if self.yama.get_rest_num() == 0:
                # none
                return self.get_all_none_response()
            
            # dahai next actions

            previous_actor = self.previous_action['actor']
            previous_pai = self.previous_action['pai']
            response = self.get_all_none_response()
            for i in range(self.PLAYER_NUM):
                if i == previous_actor:
                    continue

                # add hora
                if self.players[i].can_hora():
                    action = {
                        'type':MjMove.hora.value,
                        'actor':i,
                        'target':previous_actor,
                        'pai':previous_pai
                    }
                    response[i].append(action)
                
                # add daiminkan 
                can_daiminkan, consumed = self.players[i].can_daiminkan(previous_pai)
                if can_daiminkan:
                    action = {
                        'type':MjMove.daiminkan.value,
                        'actor':i,
                        'target':previous_actor,
                        'pai':previous_pai,
                        'consumed':consumed
                    }
                    response[i].append(action)
                
                # add pon
                # NOTE: consumed could be 2 pattern, with red, without red. 
                can_pon, consumed_list = self.players[i].can_pon(previous_pai)
                if can_pon:
                    for consumed in consumed_list:
                        action = {
                            'type':MjMove.pon.value,
                            'actor':i,
                            'target':previous_actor,
                            'pai':previous_pai,
                            'consumed':consumed
                        }
                        response[i].append(action)

                # add chi
                # check distance
                if self.distance(i, previous_actor) == 1:
                    can_chi, consumed_list = self.players[i].can_chi(previous_pai)
                    if can_chi:
                        for consumed in consumed_list:
                            action = {
                                'type':MjMove.chi.value,
                                'actor':i,
                                'target':previous_actor,
                                'pai':previous_pai,
                                'consumed':consumed
                            }
                            response[i].append(action)
            return response

        elif self.previous_action['type'] == MjMove.pon.value:
            previous_actor = self.previous_action['actor']
            previous_pai = Pai.from_str(self.previous_action['pai'])
            actions = []
            tehais = self.players[previous_actor].tehais
            dahais_dic = {}
            for i,tehai in enumerate(tehais):
                # ignore kuikae
                if previous_pai.is_same_symbol(tehai):
                    continue
                # ignore already considerd symbol
                if tehai.str in dahais_dic:
                    continue

                dahais_dic[tehai.str] = 0
                action = {
                    'type':MjMove.dahai.value,
                    'actor':previous_actor,
                    'pai':tehai.str,
                    'tsumogiri':False,
                }
                actions.append(action)

            response = self.get_all_none_response()
            response[previous_actor] = actions # not use none
            return response

        elif self.previous_action['type'] == MjMove.chi.value:
            previous_actor = self.previous_action['actor']
            dahais = self.players[previous_actor].get_chi_dahai(self.previous_action)
            actions = []
            for d in dahais:
                action = {
                    'type':MjMove.dahai.value,
                    'actor':previous_actor,
                    'pai':d,
                    'tsumogiri':False,
                }
                actions.append(action)

            response = self.get_all_none_response()
            response[previous_actor] = actions # not use none
            return response
    
        elif self.previous_action['type'] == MjMove.daiminkan.value:
            response = self.get_all_none_response()
            return response
            
        elif self.previous_action['type'] == MjMove.dora.value:
            if self.previous_without_dora_action['type'] == MjMove.ankan.value:
                response = self.get_all_none_response()
                
            elif self.previous_without_dora_action['type'] == MjMove.dahai.value:
                response = self.get_all_none_response()

            elif self.previous_without_dora_action['type'] == MjMove.kakan.value:
                response = self.get_all_none_response()

            elif self.previous_without_dora_action['type'] == MjMove.daiminkan.value: # 大明槓即乗り
                response = self.get_all_none_response()

            elif self.previous_without_dora_action['type'] == MjMove.tsumo.value: # 大明槓後めくり時の打牌
                tsumo_actor = self.previous_two_action['actor']
                tehais = self.players[tsumo_actor].tehais
                actions = []
                dahais_dic = {}
                for i,tehai in enumerate(tehais[:-1]):
                    # remove duplicate
                    if tehai.str in dahais_dic:
                        continue
                    dahais_dic[tehai.str] = ''
                    action = {
                        'type':MjMove.dahai.value,
                        'actor':tsumo_actor,
                        'pai':tehai.str,
                        'tsumogiri':False,
                    }
                    actions.append(action)
                # add tsumogiri
                action = {
                    'type':MjMove.dahai.value,
                    'actor':tsumo_actor,
                    'pai':tehais[-1].str, # tsumo pai is in last index.
                    'tsumogiri':True, 
                }
                actions.append(action)
                response = self.get_all_none_response()
                response[tsumo_actor] = actions # not use none
                return response
            else:
                raise Exception('invalid path')
            return response

        elif self.previous_action['type'] == MjMove.ankan.value:
            response = self.get_all_none_response()
            return response

        elif self.previous_action['type'] == MjMove.kakan.value:
            response = self.get_all_none_response()
            # chankan
            previous_actor = self.previous_action['actor']
            previous_pai = self.previous_action['pai']
            for i in range(self.PLAYER_NUM):
                if i == previous_actor:
                    continue

                # add hora
                if self.players[i].can_hora():
                    action = {
                        'type':MjMove.hora.value,
                        'actor':i,
                        'target':previous_actor,
                        'pai':previous_pai
                    }
                    response[i].append(action)
            
            
            
            return response

        elif self.previous_action['type'] == MjMove.hora.value:
            response = self.get_all_none_response()
            return response

        else:
            raise Exception('invalid path')
            
    
    def get_all_none_response(self):
        return {
                0:[{'type':'none'}],
                1:[{'type':'none'}],
                2:[{'type':'none'}],
                3:[{'type':'none'}],
            }

    def _do_start_kyoku(self):
        self.yama = Yama(self.consume_seed(), shuffle=True)
        
        tehais = []
        for i in range(self.PLAYER_NUM):
            tehais.append([])
            for _ in range(13):
                next_tsumo_pai = self.yama.tsumo()
                tehais[i].append(next_tsumo_pai.str)
                

        self.dora_markers = []
        dora_marker = self.yama.open_doramarker()
        self.dora_markers.append(dora_marker.str)

        self.kyoku = (4 + self.oya - self.chicha) % 4 + 1 
        # add history
        action = {
            'type':MjMove.start_kyoku.value,
            'bakaze':self.bakaze,
            'kyoku':self.kyoku,
            'honba':self.honba,
            'kyotaku':self.kyotaku,
            'oya':self.oya,
            'dora_marker':dora_marker.str,
            'tehais':tehais,
        }

        
        self.first_turn = True
        self.next_reach_pending = False
        self.reach_pending = False

        for i in range(self.PLAYER_NUM):
            self.players[i].update_state(action)



        return action

    def _do_tsumo(self):
        # assert self.previous_action['type'] in [
        #         MjMove.dahai.value, 
        #         MjMove.start_kyoku.value, 
        #         MjMove.daiminkan.value, 
        #         MjMove.kakan.value, 
        #         MjMove.dora.value,
        #         MjMove.reach_accepted.value,
        #     ]
        last_actor = self.previous_action['actor'] if 'actor' in self.previous_action else 0
        
        tsumo_actor = ((last_actor + 1) % self.PLAYER_NUM)
        # assert type(tsumo_actor) == int
        next_tsumo_pai = self.yama.tsumo()

        if self.yama.get_tsumoed_num() > self.PLAYER_NUM:
            self.first_turn = False
        
        # add history
        action = {
            'type':MjMove.tsumo.value,
            'actor':tsumo_actor,
            'pai':next_tsumo_pai.str
        }
        
        for i in range(self.PLAYER_NUM):
            self.players[i].update_state(action)



        return action


    def _do_dahai(self, action):
        # assert self.previous_action['type'] in [
        #     MjMove.tsumo.value, MjMove.reach.value, MjMove.pon.value, 
        #     MjMove.chi.value, MjMove.daiminkan.value, MjMove.dora.value]
        
        if self.previous_action['type'] == MjMove.dora.value:
            last_actor = self.previous_two_action['actor']

        else:
            last_actor = self.previous_action['actor']
        # assert last_actor == action['actor']
        # assert Pai.from_str(action['pai']) in self.players[last_actor].tehais
        
        for i in range(self.PLAYER_NUM):
            self.players[i].update_state(action)
        

        return action

    def _do_pon(self, action):        
        # assert self.previous_action['type'] in [MjMove.dahai.value, MjMove.reach_accepted.value,]
        last_actor = self.previous_action['actor']
        actor = action['actor']
        # assert last_actor == action['target']
        # assert self.players[actor].can_pon(action['pai'])
        
        self.first_turn = False


        for i in range(self.PLAYER_NUM):
            self.players[i].update_state(action)

        
        return action
    
    def _do_chi(self, action):        
        # assert self.previous_action['type'] in [MjMove.dahai.value, MjMove.reach_accepted.value,]
        last_actor = self.previous_action['actor']
        actor = action['actor']
        # assert last_actor == action['target']
        # assert self.distance(actor, action['target']) == 1
        # assert self.players[actor].can_chi(action['pai'])
        
        self.first_turn = False

        for i in range(self.PLAYER_NUM):
            self.players[i].update_state(action)


        return action
    

    def _do_daiminkan(self, action):
        # assert self.previous_action['type'] in [MjMove.dahai.value, MjMove.reach_accepted.value,]
        last_actor = self.previous_action['actor']
        actor = action['actor']
        # assert last_actor == action['target']
        # assert self.players[actor].can_daiminkan(action['pai'])
        
        self.first_turn = False

        for i in range(self.PLAYER_NUM):
            self.players[i].update_state(action)



        
        return action

    def _do_ankan(self, action):
        # assert self.previous_action['type'] == MjMove.tsumo.value
        last_actor = self.previous_action['actor']
        actor = action['actor']
        # assert last_actor == actor
        can_ankan, _ = self.players[actor].can_ankan(action['consumed'][0])
        # assert can_ankan

        self.first_turn = False

        for i in range(self.PLAYER_NUM):
            self.players[i].update_state(action)



        
        return action

    def _do_kakan(self, action):
        # assert self.previous_action['type'] == MjMove.tsumo.value
        last_actor = self.previous_action['actor']
        actor = action['actor']
        # assert last_actor == actor
        can_kakan, _ = self.players[actor].can_kakan(action['pai'])
        # assert can_kakan

        self.first_turn = False

        for i in range(self.PLAYER_NUM):
            self.players[i].update_state(action)
        
        return action

    def _do_reach(self, action):
        # assert self.previous_action['type'] == MjMove.tsumo.value
        last_actor = self.previous_action['actor']
        actor = action['actor']
        # assert last_actor == actor
        can_reach = self.players[actor].reach_state == 'none' and self.players[actor].menzen
        # assert can_reach

        for i in range(self.PLAYER_NUM):
            self.players[i].update_state(action)
        self.reach_pending = True
        return action


    def _do_fanpai_ryukyoku(self):
        is_nagashi = False
        nagashi_deltas = [0] * self.PLAYER_NUM
        for player in self.players:
            if len(player.sutehais) == len(player.ho) and\
                all([s.is_yaochu() for s in player.sutehais]):
                is_nagashi = True
            if player.id == self.oya:
                
                for i in range(4):
                    nagashi_deltas[i] -= 4000
                    
                    if i == player.id:
                        nagashi_deltas[i] += 4000 + 12000
            else:
                
                for i in range(4):
                    if i == self.oya:
                        nagashi_deltas[i] -= 2000 * 2
                    else:
                        nagashi_deltas[i] -= 2000

                    if i == player.id:
                        nagashi_deltas[i] += 2000 + 8000
            

        

        tenpais = [bool(p.tenpai) for p in self.players]
        tenpais_num = sum(tenpais)
        if tenpais_num in [0, self.PLAYER_NUM]:
            deltas = [0] * self.PLAYER_NUM
        else:
            plus_score = 3000 // tenpais_num
            minus_score =  -3000 // (self.PLAYER_NUM-tenpais_num)
            deltas = [plus_score if t else minus_score for t in tenpais]


        reason = 'nagashimangan' if is_nagashi else 'fanpai'
        action = {
            'type':MjMove.ryukyoku.value,
            'reason':reason,
            'tehais':[p.str_sep_tehais for p in self.players],
            'tenpais':tenpais,
            'deltas':nagashi_deltas if is_nagashi else deltas,
            'scores':self.get_scores(deltas)
        }
        for i in range(self.PLAYER_NUM):
            self.players[i].update_state(action)
        
        self.update_oya(tenpais[self.oya], reason)
        return action


    def _do_opendora(self):
        dora = self.yama.open_doramarker()
        action = {
            'type':MjMove.dora.value,
            'dora_marker':dora.str
        }
        for i in range(self.PLAYER_NUM):
            self.players[i].update_state(action)

        return action
        
    def _do_reach_accept(self):
        reach_action = self.previous_two_action
        
        # assert reach_action['type'] == MjMove.reach.value
        self.kyotaku += 1
        deltas = [0] * self.PLAYER_NUM
        deltas[reach_action['actor']] -= 1000
        scores = self.get_scores(deltas)
        
        action = {
            'type':MjMove.reach_accepted.value,
            'actor':reach_action['actor'],
            'deltas':deltas,
            'scores':scores,
        }
        for i in range(self.PLAYER_NUM):
            self.players[i].update_state(action)

        return action
        
    
    def _do_hora(self, actions:List, ura:List[str]=None):
        tsumibo = self.honba
        
        distance_actions = {}
        for index, action in enumerate(actions):
            distance = self.distance(action['actor'], action['target'])
            distance_actions[distance] = action 

        distance_sorted_actions = sorted(distance_actions.items(), key=lambda x:x[0])
        result_actions = []
        for (dist, action) in distance_sorted_actions:
            
            hora_actor = action['actor']
            if self.players[hora_actor].reach and (ura is None):
                ura = []
                uradora_num = self.yama.get_doramarker_num()
                for _ in range(uradora_num):
                    ura.append(self.yama.tsumo().str)

            if self.players[hora_actor].reach_state == 'accepted':
                uradora_markers = ura
            else:
                uradora_markers = []
            
            hora = self.get_hora(
                action,
                **{
                    "uradora_markers":uradora_markers,
                    "previous_action":self.previous_action,
                })

            if hora.valid == False:
                raise Exception('no yaku')

            deltas = [0] * self.PLAYER_NUM
            deltas[hora_actor] += hora.points + tsumibo * 300 + self.kyotaku * 1000

            pao_id = self.players[hora_actor].pao_for_id
            if hora.hora_type == 'tsumo':
                if pao_id is not None:
                    deltas[pao_id] -= (hora.points + tsumibo * 300)
                else:
                    for player in self.players:
                        if player.id == hora_actor:
                            continue
                        deltas[player.id] -= ((hora.oya_payment if player.id == self.oya else hora.ko_payment) + tsumibo * 100)
            else:
                if pao_id == action['target']:
                    pao_id == None
                if pao_id is not None:
                    deltas[pao_id] -= (hora.points//2 + tsumibo * 300)
                    deltas[action['target']] -= (hora.points//2)
                else:
                    deltas[action['target']] -= (hora.points + tsumibo * 300)
            action = {
                'type':MjMove.hora.value,
                'actor':hora_actor,
                'target':action['target'],
                'pai':action['pai'],
                'hora_tehais':self.players[hora_actor].str_sep_tehais,
                'uradora_markers':uradora_markers,
                'yakus':hora.yakus,
                'fu':hora.fu,
                'fan':hora.fan,
                'hora_points':hora.points,
                'deltas':deltas,
                'scores':self.get_scores(deltas),
            }

            if pao_id:
                action['pao'] = pao_id
            result_actions.append(action)

            tsumibo = 0
            self.kyotaku = 0

            for i in range(self.PLAYER_NUM):
                self.players[i].update_state(action)

        self.update_oya(any([a['actor'] == self.oya for a in actions]), False)

        return result_actions

    def _do_ryukyoku(self, reason, actors=[]):
        if reason == 'kyushukyuhai':
            actor = actors[0]
        else:
            actor = None
        
        tenpais = []
        tehais = []

        for player in players:
            if reason == 'suchareach' or (player.id in actors):
                tenpais.append(bool(reason != 'kyushukyuhai'))
                tehais.push(player.str_sep_tehais)
            else:
                tenpais.push(false)
                tehais.push([Pai.UNKNOWN] * len(player.tehais))

        action = {
            'type':MjMove.ryukyoku.value,
            'actor':actor,
            'reason':reason,
            'tenpais':tenpais,
            'tehais':tehais,
            'deltas':[0] * self.PLAYER_NUM,
            'scores':[p.score for p in self.players]
        }
    
        for i in range(self.PLAYER_NUM):
            self.players[i].update_state(action)
        
        self.update_oya(True, reason)

        return action
        


    def _do_end_kyoku(self):
        action = {
            'type':MjMove.end_kyoku.value,
        }
        for i in range(self.PLAYER_NUM):
            self.players[i].update_state(action)
        
        return action

    def _do_end_game(self):
        action = {
            'type':MjMove.end_game.value,
        }
        for i in range(self.PLAYER_NUM):
            self.players[i].update_state(action)
        
        return action
        


    def get_hora(self, action, **params):
        
        if action['type'] != MjMove.hora.value:
            raise Exception('shoud not happen')

        hora_type = 'tsumo' if action['actor'] == action['target'] else 'ron'

        if hora_type == 'tsumo':
            tehais = self.players[action['actor']].tehais[:-1] # remove tsumo
        else:
            tehais = self.players[action['actor']].tehais
        
        doras = [Pai.from_str(p).succ for p in self.dora_markers]
        if 'uradora_markers' in params:
            uradoras = [Pai.from_str(p).succ for p in params['uradora_markers']]
        else:
            uradoras = []
        
        hora_player = self.players[action['actor']]
        return Hora(
            tehais=tehais,
            furos =hora_player.furos,
            taken=Pai.from_str(action['pai']),
            hora_type=hora_type,
            oya=self.oya == action['actor'],
            bakaze=self.bakaze,
            jikaze=self.players[action['actor']].jikaze,
            doras=doras,
            uradoras=uradoras,
            reach=hora_player.reach_state == 'accepted',
            double_reach=hora_player.double_reach,
            ippatsu=hora_player.ippatsu_chance,
            rinshan=hora_player.rinshan,
            haitei=(self.yama.get_rest_num() == 0) and (hora_player.rinshan == False),
            first_turn=self.first_turn,
            chankan=('previous_action' in params) and (params['previous_action']['type'] == MjMove.kakan.value),
        )


    def consume_seed(self):
        consumed_seed = self._current_seed
        self._current_seed += 1
        return consumed_seed


    @property
    def is_end(self):
         return 'type' in self.previous_action and self.previous_action['type'] == MjMove.end_game.value

    @property
    def previous_action(self):
        if len(self.dealer_history) == 0:
            return None
        return self.dealer_history[-1]
    @property
    def previous_two_action(self):
        if len(self.dealer_history) < 2:
            return None
        return self.dealer_history[-2]
    
    @property
    def previous_without_dora_action(self):
        if len(self.dealer_history) == 0:
            return None
        
        index = len(self.dealer_history)-1
        while True:
            if self.dealer_history[index]["type"] != MjMove.dora.value:
                return self.dealer_history[index]
            index -= 1
        
        raise Exception("not intended path")

        # return [h for h in self.dealer_history if h['type'] != MjMove.dora.value][-1]

    def distance(self, player1_id, player2_id):
        """clock rotation distance from player1. ex) (1, 2) => 3,  (0, 3) => 1
        """
        return (self.PLAYER_NUM + player1_id - player2_id) % self.PLAYER_NUM
    
    def get_scores(self, deltas):
        moved_scores = []
        for i in range(self.PLAYER_NUM):
            moved_score = self.players[i].score + deltas[i]
            moved_scores.append(moved_score)
        return moved_scores
    
    def update_oya(self, renchan, ryukyoku_reason):
        if renchan == False:
            self.oya = (self.oya + 1) % self.PLAYER_NUM
            if self.oya == self.chicha:
                self.bakaze = Pai.from_str(self.bakaze).succ.str

        if renchan or ryukyoku_reason:
            self.honba += 1
        else:
            self.honba = 0

        if self.game_type == 'tonpu':
            self.last = self.decide_last(Pai.from_str('E'), renchan, ryukyoku_reason)
        elif self.game_type == 'tonnan':
            self.last = self.decide_last(Pai.from_str('S'), renchan, ryukyoku_reason)

    
    def decide_last(self, last_bakaze, renchan, ryukyoku_reason):
        # assert isinstance(self.bakaze, str)
        # assert isinstance(last_bakaze, Pai)
        if any([p.score < 0 for p in self.players]):
            return True
        if self.bakaze == last_bakaze.succ.succ.str:
            return True
        if ryukyoku_reason and (ryukyoku_reason not in ['fanpai', 'nagashimangan']):
            return False
        
        if renchan:
            if (self.bakaze == last_bakaze.succ.str) or (self.bakaze == last_bakaze.str and self.kyoku == 4):
                oya_over30000 = self.players[self.oya].score >= 30000
                oya_top = True
                for i, player in enumerate(self.players):
                    if self.oya == i:
                        continue

                    if self.players[self.oya].score <= player.score:
                        oya_top = False
                        break
                return oya_over30000 and oya_top
        elif self.bakaze == last_bakaze.succ.str:
            return any([p.score >= 30000 for p in self.players])
        return False
    
    def render_console(self, seat):
        s = f"""
kyoku:{self.kyoku}, bakaze:{self.bakaze}, chicha:{self.chicha}
last_action:
\t{self.previous_action}
player{seat} tehai:
\t{self.players[seat].tehais}
"""
        possible_actions_string = f'player{seat} action space:\n'
        for a in self.possible_actions[seat]:
            possible_actions_string += '\t' + str(a) + '\n'
        print(s + possible_actions_string)




if __name__ == "__main__":
    import random
    import pprint
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    board = Board(game_type="tonpu")
    board.reset()
    while board.is_end == False:
        # board.render_console(0)
        print(board.get_state())

        possible_actions = board.possible_actions
        print("---")
        # pprint.pprint(possible_actions)
        actions = {}
        for i in range(board.PLAYER_NUM):
            sellected = random.choice(possible_actions[i])
            actions[i] = sellected
        
        board.step(actions)
    
        