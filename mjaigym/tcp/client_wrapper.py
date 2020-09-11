import sys
import socket
import threading
import time
import json
from typing import Dict
import signal
from mjaigym.board.mj_move import MjMove
from mjaigym.board import ClientBoard
from mjaigym.client import Client


signal.signal(signal.SIGINT, signal.SIG_DFL)


def default_on_message(mes, name):
    pass
def default_on_send(mes, name):
    pass

class ClientWrapper:
    def __init__(
            self, 
            server_ip, 
            server_port, 
            identify_string,
            client:Client, 
            on_message_handler=default_on_message, 
            on_send_handler=default_on_send, 
            name="default_name", 
            room_name="default",
            ):
        self.server_ip = server_ip
        self.server_port = server_port
        self.client = client
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.on_message_handler = on_message_handler
        self.on_send_handler = on_send_handler
        self.id = None
        self.name = name
        self.is_end = False
        self.recieve_thread = None
        self.room_name = room_name
        self.board = ClientBoard()
        self.identify_string = identify_string
        
    def run(self):
        """alies of connect
        """
        self.connect()

    def connect(self):
        self.sock.connect((self.server_ip,self.server_port))
        self.recieve_thread = threading.Thread(target=self.recieve_handler, daemon=True)
        self.recieve_thread.start()

    def recieve_handler(self):
        while True:
            try:
                buf = b""
                while True:
                    data = self.sock.recv(4096)
                    if len(data) <= 0:
                        print(f"disconnected")
                        return
                        
                    if len(buf) > 8192:
                        raise Exception(f"ignore big message")
                    
                    buf += data
                    if b"\n" in buf:
                        break
                
                data_str = buf.decode("utf-8")
                
                if '\n' not in data_str:
                    raise Exception(f"newline not found")

                messages = data_str.split('\n')
                if len(messages) != 2:
                    print(messages)
                    raise Exception(f"multiple new line found")
                
                message = messages[0]
                self.on_message(message)
                
                if self.is_end:
                    # print(f"exit recieve thread")
                    return
            
            
            except Exception as e:
                print(e)
                self.sock.shutdown(socket.SHUT_RDWR)
                self.sock.close()
                return


    def on_message(self, message):
        """overwrite this function
        """
        self.on_message_handler(message, self.name)
        # time.sleep(0.1)
        message = json.loads(message)

        if message['type'] == 'error':
            raise Exception(f"got error message:{message}")
        elif message['type'] == 'hello':
            response = {
                "type":MjMove.join.value,
                "room":self.room_name,
                "name":self.name,
                "hash":self.identify_string,
            }
        else:
            self.board.step(message)
            state = self.board.get_state()
            response = self.client.think(state)

        self.send(response)
        

        if message['type'] == MjMove.end_game.value:
            self.is_end = True


    def send(self, message:Dict):
        message = json.dumps(message)
        self.on_send_handler(message, self.name)
        message = f"{message}\n".encode("utf-8")
        self.sock.send(message)
        


    def __del__(self):
        # print(f"del {self}")
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
            self.sock.close()
        except Exception:
            pass

if __name__ == "__main__":
    lock = threading.Lock()
    def sync_recieve_handler(mes, name):
        with lock:
            print(f"<-{mes} @{name}")

    def sync_send_handler(mes, name):
        with lock:
            print(f"->{mes} @{name}")

    threads = []
    host = 'localhost'
    # host = '52.155.110.248'
    # host = 'game.mjaigym.net'
    

    clients = []



    for room_id in range(1):
        for player_id in range(4):
            from mjaigym.client.max_ukeire_client import MaxUkeireClient
            agent = MaxUkeireClient(id=player_id)
            identiry_string = f"sample_string_{player_id}"
            client = ClientWrapper(host, 48000, identiry_string, agent, sync_recieve_handler, sync_send_handler, f"name{player_id}", f"room{room_id}")
            client.connect()
            threads.append(client.recieve_thread)
            clients.append(client)

    for t in threads:
        t.join()

    time.sleep(1)
            
