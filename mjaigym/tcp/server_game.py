import socket
from typing import Tuple, List
import json
import threading
import random
from mjaigym.board import Board
from mjaigym.board.mj_move import MjMove
import copy
import os
import datetime
import uuid

import mjaigym.loggers as lgs

LOOP_NUM = 70
SHOW_VALID_MOVE = True

class ServerGame:
    def __init__(self, 
            room_name:str, 
            client_keys:List[str], 
            clients:List[socket.socket], 
            names:List[str]
        ):

        # shuffle players
        key_client_names = list(zip(client_keys, clients, names))
        random.shuffle(key_client_names)
        client_keys = [c[0] for c in key_client_names]
        clients = [c[1] for c in key_client_names]
        names = [c[2] for c in key_client_names]

        self.room_name = room_name
        self.client_keys = client_keys
        self.client_key_clients = dict(zip(client_keys, clients))
        self.id_clients = dict(zip(range(4), clients))
        self.client_key_ids = dict(zip(client_keys, range(4)))
        self.recieve_lock = threading.Lock()
        self.is_end = False
        self._buf = []
        self.message_buf = {}
        self.names = names

        self.seed = random.randint(0,2**32-1)
        self.board = Board(seed=self.seed, names=names, game_type='tonpu')

        self._error_message = None
    
    def set_error_message(self, error_message):
        self._error_message = error_message

    @property
    def is_ok_end(self):
        return self._error_message is None and \
            len(self.board.dealer_history) > 0 and \
            self.board.dealer_history[-1]['type'] == MjMove.end_game.value

    def on_message(self, message, client_key:str):

        with self.recieve_lock:
            player_id = self.client_key_ids[client_key]
            self.message_buf[player_id] = message

            if len(self.message_buf) < 4:
                return
            


            if len(self.board.pending_message) > 0:
                dealer_message = self.board.consume_pending_message()
            else:
                dealer_message = self.board.step(self.message_buf)
            # add possible actions
            dealer_message["possible_actions"] = self.board.possible_actions
            

            self.message_buf.clear()
            
            if dealer_message is None:
                return

            if dealer_message['type'] == MjMove.end_game.value :
                lgs.logger_server.info("end_game")
                self.is_end = True

            """add valid move
            """
            for i in range(4):
                masked_message = self.message_in_view(dealer_message, i)

                if SHOW_VALID_MOVE:
                    masked_message["possible_actions"] = self.board.possible_actions[i]

                self.send_message(masked_message, i)
            


    def message_in_view(self, message, player_id):
        rewritee_message = copy.deepcopy(message)
        if message['type'] not in [MjMove.start_game.value, MjMove.start_kyoku.value, MjMove.tsumo.value]:
            return rewritee_message

        if message['type'] == MjMove.start_game.value:
            for i in range(4):
                rewritee_message['id'] = i

        elif message['type'] == MjMove.start_kyoku.value:
            for i in range(4):
                if i != player_id:
                    rewritee_message['tehais'][i] = ["?"] * len(rewritee_message['tehais'][i])

        elif message['type'] == MjMove.tsumo.value:
            if message['actor'] != player_id:
                rewritee_message['pai'] = '?'

        return rewritee_message


    def send_message(self, message, player_id):
        if player_id not in self.id_clients:
            return

        message = json.dumps(message)
        client = self.id_clients[player_id]
        
        client.settimeout(10)
        try:
            client.send(f"{message}\n".encode('utf-8'))
        except:
            self.is_end = True


    def start_game(self):
        assert len(self.board.dealer_history) == 1
        assert self.board.previous_action['type'] == MjMove.start_game.value
        
        for i in range(4):
            start_game_message = copy.deepcopy(self.board.previous_action)
            start_game_message["id"] = i
            start_game_message["names"] = self.names
            self.send_message(start_game_message, i)


    def send_error_for_all_client(self, error_message:str):
        """send error message and disconnect clients socket
        """
        error_message = {
            "type":"error",
            "message":error_message,
        }

        for i in range(4):
            self.send_message(error_message, i)
                

    def dump_mjson(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        fname = _get_fname()
        fpath = os.path.join(dir_path, fname)
        history = self.board.dealer_history
        with open(fpath,'wt') as f:
            formatted_lines = [json.dumps(l)+os.linesep for l in self.board.dealer_history]
            f.writelines(formatted_lines)
        return fname
        

    def dump_error_mjson(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        fname = _get_fname()
        fpath = os.path.join(dir_path, fname)
        history = self.board.dealer_history
        with open(fpath,'wt') as f:
            formatted_lines = [json.dumps(l)+os.linesep for l in self.board.dealer_history]
            f.writelines(formatted_lines)
            f.write(f"---\n")
            formatted_responses = [json.dumps(l)+os.linesep for l in self.board.renponse_history]
            f.writelines(formatted_responses)
            f.write(f"---\n")
            f.write(f"error:{self._error_message}")
        return fname


def _get_fname():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + "_" + str(uuid.uuid4()) + '.mjson'
