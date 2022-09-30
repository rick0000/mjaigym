import json
import signal
import socket
import sys
import threading
import time
from typing import Dict

from mjaigym.board import ClientBoard
from mjaigym.board.mj_move import MjMove

signal.signal(signal.SIGINT, signal.SIG_DFL)


def default_on_message(mes, name):
    pass


def default_on_send(mes, name):
    pass


class TcpWrapper:
    def __init__(
        self,
        server_ip,
        server_port,
        identify_string,
        on_message_handler=default_on_message,
        on_send_handler=default_on_send,
        name="default_name",
        room_name="default",
    ):
        self.server_ip = server_ip
        self.server_port = server_port
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
        """alies of connect"""
        self.connect()

    def connect(self):
        self.sock.connect((self.server_ip, self.server_port))
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

                if "\n" not in data_str:
                    raise Exception(f"newline not found")

                messages = data_str.split("\n")
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
        """overwrite this function"""
        raise NotImplementedError()

    def send(self, message: Dict):
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
