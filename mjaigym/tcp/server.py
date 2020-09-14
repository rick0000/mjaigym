import argparse
import socket
import select
import threading
import json
import subprocess
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)
from mjaigym.tcp.server_game import ServerGame
from typing import Tuple
import time
from pathlib import Path
import traceback
import mjaigym.loggers as lgs



class Server:
    
    def __init__(self, port, clientlimit, logdir):
        self.port = port
        self.clientlimit = clientlimit

        # hold sockets, room, client name
        self.clients = {}

        # hold before start rooms
        self.waiting_rooms = {}

        # hold active rooms
        self.rooms = {}


        self.server_sock = None
        self.room_handle_lock = threading.Lock()
        self.finish_games = None
        
        # rooms need to be closed
        self.end_rooms = []
        # clients need to be closed
        self.disconnect_client_keys = []

        self.logdir = Path(logdir) / "ok"
        self.error_logdir = Path(logdir) / "error"
        

    def __del__(self):
        
        try:
            if self.server_sock:
                self.server_sock.shutdown(socket.SHUT_RDWR)
                self.server_sock.close()
        except Exception:
            pass


    def show_status(self):
        while True:
            time.sleep(3)
            lgs.logger_server.info(f"-------------")
            for k, clients in self.waiting_rooms.items():
                lgs.logger_server.info(f"waiting_room:{k}, {len(clients)}")
            lgs.logger_server.info(f"rooms:{list(self.rooms.keys())}")
            lgs.logger_server.info(f"room_members:{[r.names for r in self.rooms.values()]}")
            lgs.logger_server.info(f"clients count:{len(self.clients)}")


    def run_async(self):
        server_thread = threading.Thread(target=self.run, daemon=True)
        server_thread.start()

    def run(self):
        host = "0.0.0.0"
        lgs.logger_server.info(f"start server {host}:{self.port} clientlimit:{self.clientlimit}")
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((host, self.port))
        self.server_sock.listen(self.clientlimit)
        
        # show_status_thread = threading.Thread(target=self.show_status, daemon=True)
        # show_status_thread.start()

        client_watch_thread = threading.Thread(target=self.handle_disconnected_clint, daemon=True)
        client_watch_thread.start()

        room_watch_thread = threading.Thread(target=self.handle_end_games, daemon=True)
        room_watch_thread.start()

        # wait loop
        while True:
            try:
                connection, address = self.server_sock.accept()
            except KeyboardInterrupt:
                break
            except Exception as e:
                lgs.logger_server.info(e)

            # send hello message
            initial_message = json.dumps({"type":"hello","protocol":"mjsonp","protocol_version":1})
            initial_message = f"{initial_message}\n".encode("utf-8")
            connection.send(initial_message)

            # lgs.logger_server.info(connection, address)
            client_key = f"{address[0]}:{address[1]}"
            self.clients[client_key] = {
                'client':connection
            }
            thread = threading.Thread(target=self.recieve_handler, args=(connection, client_key), daemon=True)
            thread.start()

        
        # clean up after KeyboardInterrupt
        for client_key in self.clients:
            try:
                client = self.clients[client_key]['client']
                client.shutdown(socket.SHUT_RDWR)
                client.close()
            except:
                pass


    def recieve_handler(self, connection:socket.socket, client_key:str):
        """wait and handle message for each client connection.
        if disconnected, error message will be send for same room clients and room will be closed.
        """
        while True:
            try:
                buf = b""
                while True:
                    data = connection.recv(4096)
                    
                    if len(data) <= 0:
                        # end recieve thread
                        # lgs.logger_server.info(f"disconnected @{client_key}")
                        self.disconnect_client_keys.append(client_key)
                        return

                    if len(buf) > 8192:
                        raise Exception(f"ignore big message @{client_key}")
                    buf += data
                    if b"\n" in buf:
                        break
                    
                
                data_str = buf.decode("utf-8")

                if '\n' not in data_str:
                    raise Exception(f"newline not found @{client_key}")

                messages = data_str.split('\n')
                if len(messages) != 2:
                    lgs.logger_server.info(data_str, messages)
                    raise Exception(f"multiple new line found @{client_key}")
                
                message = messages[0]
                self.on_message(message, connection, client_key)
                
            except Exception as e:
                lgs.logger_server.info(e)
                self.disconnect_client_keys.append(client_key)
                return
        

    def handle_disconnected_clint(self):
        while True:
            time.sleep(1)
            try:
                self._handle_disconnected_clint()
            except:
                lgs.logger_server.info(f"exception in handle_disconnected_clint")
                error_message = traceback.format_exc()
                lgs.logger_server.info(error_message)
    def _handle_disconnected_clint(self):
        """close disconnected clients socket"""

        with self.room_handle_lock:

            for client_key in self.disconnect_client_keys:
                if client_key in self.clients:
                    client = self.clients[client_key]
                    
                    # handle waiting room
                    if "waiting_room" not in client:
                        continue

                    waiting_room = client['waiting_room']
                    if waiting_room in self.waiting_rooms and client_key in self.waiting_rooms[waiting_room]:
                        del self.waiting_rooms[waiting_room][client_key]

                    target_room = [room for room in self.rooms.values() if client_key in room.client_keys]
                    if len(target_room) != 0:
                        # if room is not closed, notify same room client
                        room = target_room[0]
                        if room.is_ok_end == False:
                            name = client['name'] if 'name' in client else 'someone'
                            error_message = f"disconnect by {name}"
                            room.send_error_for_all_client(error_message)
                            room.set_error_message(error_message)
                        
                        # disconnect socket
                        for room_client_key, room_client in room.client_key_clients.items():
                            try:
                                room_client.shutdown(socket.SHUT_RDWR)
                                room_client.close()
                            except Exception:
                                pass
                            
                            if room_client_key in self.clients:
                                del self.clients[room_client_key]

                        # end room
                        room.is_end = True
                    else:
                        # if not attend to room, only do close socket 
                        connection = self.clients[client_key]['client']
                        try:
                            connection.shutdown(socket.SHUT_RDWR)
                            connection.close()
                        except Exception:
                            pass

                if client_key in self.clients:
                    del self.clients[client_key]

    

    def handle_end_games(self):
        while True:
            time.sleep(1)
            try:
                self._handle_end_games()
            except Exception as e:
                lgs.logger_server.info(f"exception in handle_end_games")
                error_message = traceback.format_exc()
                lgs.logger_server.info(error_message)
                

    def _handle_end_games(self):
        """close room"""
        with self.room_handle_lock:
            for room_name, room in self.rooms.items():
                if room.is_end:
                    self.end_rooms.append(room)

            for room in self.end_rooms:
                lgs.logger_server.info(f"close room {room}")
                if room.is_ok_end:
                    fname = room.dump_mjson(self.logdir)
                    lgs.logger_server.info(f"dumped {fname}")
                else:
                    fname = room.dump_error_mjson(self.error_logdir)
                    lgs.logger_server.info(f"dumped error {fname}")

                for client in room.id_clients.values():
                    try:
                        client.shutdown(socket.SHUT_RDWR)
                        client.close()
                    except:
                        pass

                room_name = room.room_name
                if room_name in self.waiting_rooms:
                    del self.waiting_rooms[room_name]
                if room_name in self.rooms:
                    del self.rooms[room_name]
                del room
        
            self.end_rooms.clear()


    def on_message(self, message, connection, client_key):
        message = json.loads(message)
        if message['type'] == 'join':
            self.on_join(message, connection, client_key)
        else:
            if 'room' not in self.clients[client_key]:
                raise Exception(f"get invalid message {message} from a client who does not join to the room.")
            
            room = self.clients[client_key]['room']
            
            try:
                room.on_message(message, client_key)
             
            except Exception as e:
                error_message = traceback.format_exc()
                room.set_error_message(error_message)
                lgs.logger_server.warn(error_message)

    def on_join(self, message, connection, client_key):
        for check_key in ['type', 'room', 'name']:
            if check_key not in message:
                message = json.dumps({
                        "type":"error",
                        "message":f"json key \"{check_key}\" not found in message",
                })
                connection.send(f"{message}\n".encode('utf-8'))
            
        
        assert message['type'] == 'join', "type is not join"

        name = message['name']
        self.clients[client_key]['name'] = name

        room = message['room']
        self.clients[client_key]['waiting_room'] = room
        
        # require lock when handle rooms
        with self.room_handle_lock:

            if room not in self.waiting_rooms:
                self.waiting_rooms[room] = {}
            
            # check can join
            if len(self.waiting_rooms[room]) >= 4:
                # if cannnot join, return error message
                message = json.dumps({
                    "type":"error",
                    "message":"room is busy, try after few minutes.",
                })
                connection.send(f"{message}\n".encode('utf-8'))
                if name == 'Manue1':
                    raise Exception("Manue1 cannot join, no probrem.")
                else:
                    raise Exception("client cannot join.")

            # add new client
            self.waiting_rooms[room][client_key] = {
                "client":connection,
                "name":name,
            }

            
            if len(self.waiting_rooms[room]) == 4:
                # start room
                lgs.logger_server.info(f"start room {room}")
                client_keys = [r for r in self.waiting_rooms[room].keys()]
                clients = [r['client'] for r in self.waiting_rooms[room].values()]
                names = [r['name'] for r in self.waiting_rooms[room].values()]
                
                server_game = ServerGame(room, client_keys, clients, names)
                self.rooms[room] = server_game
                
                for client_key in client_keys:
                    self.clients[client_key]['room'] = server_game

                server_game.start_game()
                

            # manue room
            if len(self.waiting_rooms[room]) == 1 and 'manue' in room :
                for i in range(3):
                    thread = threading.Thread(target=self.run_manue, args=(room, f"manue{i}"), daemon=True)
                    thread.start()
                    

    def run_manue(self, room_name, player_name):
        proc = subprocess.run([
                f"docker build -t manue:dev manue/. && docker run --rm manue:dev /mjai/mjai/run_manue.sh {room_name} {player_name}"
                ], 
            shell=True,
            stdout=subprocess.DEVNULL,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=48000)
    parser.add_argument("--clientlimit", type=int, default=32)
    parser.add_argument("--logdir", type=str, default="./tcplog")

    args = parser.parse_args()
    server = Server(args.port, args.clientlimit, args.logdir)
    server.run()


