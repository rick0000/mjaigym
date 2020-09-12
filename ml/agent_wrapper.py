import json

from mjaigym.board.mj_move import MjMove
from mjaigym.tcp.client_wrapper import ClientWrapper
from ml.agent import MjAgent

class AgentWrapper(ClientWrapper):
    def __init__(
            self, 
            server_ip, 
            server_port, 
            identify_string,
            agent:MjAgent, 
            on_message_handler, 
            on_send_handler, 
            name="default_name", 
            room_name="default",
            ):
        super(AgentWrapper, self).__init__(
            server_ip,
            server_port,
            identify_string,
            agent,
            on_message_handler,
            on_send_handler,
            name,
            room_name,
        )
        self.agent = agent
        self.observe = 
    
    def on_message(self, message):
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
            response = self.agent.(state)

        self.send(response)
        

        if message['type'] == MjMove.end_game.value:
            self.is_end = True

