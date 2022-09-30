import json
import signal
import socket
import sys
import threading
import time
from typing import Dict

from mjaigym.board import ClientBoard
from mjaigym.board.mj_move import MjMove
from mjaigym.client import Client
from mjaigym.client.max_ukeire_client import MaxUkeireClient
from mjaigym.tcp.tcp_wrapper import TcpWrapper

signal.signal(signal.SIGINT, signal.SIG_DFL)


def default_on_message(mes, name):
    pass


def default_on_send(mes, name):
    pass


class ClientTcpWrapper(TcpWrapper):
    def __init__(
        self,
        server_ip,
        server_port,
        identify_string,
        client: Client,
        on_message_handler=default_on_message,
        on_send_handler=default_on_send,
        name="default_name",
        room_name="default",
    ):
        super(ClientTcpWrapper, self).__init__(
            server_ip=server_ip,
            server_port=server_port,
            identify_string=identify_string,
            on_message_handler=on_message_handler,
            on_send_handler=on_send_handler,
            name=name,
            room_name=room_name,
        )
        self.client = client

    def on_message(self, message):
        """overwrite this function"""
        self.on_message_handler(message, self.name)
        # time.sleep(0.1)
        message = json.loads(message)

        if message["type"] == "error":
            raise Exception(f"got error message:{message}")
        elif message["type"] == "hello":
            response = {
                "type": MjMove.join.value,
                "room": self.room_name,
                "name": self.name,
                "hash": self.identify_string,
            }
        else:
            self.board.step(message)
            state = self.board.get_state()
            response = self.client.think(state)

        self.send(response)

        if message["type"] == MjMove.end_game.value:
            self.is_end = True


if __name__ == "__main__":
    lock = threading.Lock()

    def sync_recieve_handler(mes, name):
        with lock:
            print(f"<-{mes} @{name}")

    def sync_send_handler(mes, name):
        with lock:
            print(f"->{mes} @{name}")

    threads = []
    host = "localhost"
    # host = '52.155.110.248'
    # host = 'game.mjaigym.net'

    clients = []

    for room_id in range(1):
        for player_id in range(4):

            client = MaxUkeireClient(id=player_id)
            identiry_string = f"sample_string_{player_id}"
            client = ClientTcpWrapper(
                host,
                48000,
                identiry_string,
                client,
                sync_recieve_handler,
                sync_send_handler,
                f"name{player_id}",
                f"room{room_id}",
            )
            client.connect()
            threads.append(client.recieve_thread)
            clients.append(client)

    for t in threads:
        t.join()

    time.sleep(1)
