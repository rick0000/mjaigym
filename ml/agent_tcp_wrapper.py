import json
import socket
import threading
from typing import Dict
import traceback

import numpy as np

from mjaigym.board.mj_move import MjMove
from mjaigym.tcp.tcp_wrapper import TcpWrapper
from ml.agent import MjAgent
from mjaigym.board import ClientBoard
from mjaigym.reward.kyoku_score_reward import KyokuScoreReward
from mjaigym.config import ModelConfig
from ml.model import  Head2SlModel, Head34SlModel
from ml.custom_observer import SampleCustomObserver, MjObserver
from ml.agent import InnerAgent, MjAgent, DahaiTrainableAgent, FixPolicyAgent

def default_on_message(mes, name):
    print(f"<- {mes}")
def default_on_send(mes, name):
    print(f"-> {mes}")


class AgentTcpWrapper(TcpWrapper):
    def __init__(
            self, 
            server_ip, 
            server_port, 
            identify_string,
            env:MjObserver,
            agent:MjAgent, 
            on_message_handler=default_on_message, 
            on_send_handler=default_on_send, 
            name="default_name", 
            room_name="default",
            ):
        super(AgentTcpWrapper, self).__init__(
            server_ip=server_ip,
            server_port=server_port,
            identify_string=identify_string,
            on_message_handler=on_message_handler,
            on_send_handler=on_send_handler,
            name=name,
            room_name=room_name,
        )
        self.id = None
        self.name = name
        self.is_end = False
        self.recieve_thread = None
        self.room_name = room_name
        self.identify_string = identify_string
        self.env = env
        self.agent = agent

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
            self.send(response)
            return


        if message['type'] == 'start_game':
            self.id = message['id']
            self.env.reset()

        state, reward, done, info = self.env.step(message)
        player_observation = state[self.id]
        if "possible_actions" in message:
            player_possible_actions = message["possible_actions"]
        else:
            player_possible_actions = [{"type":"none"}]
        board_state = info["board_state"]
        
        response = self.agent.think_one_player(player_observation, player_possible_actions, self.id, board_state)
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
    
    env = SampleCustomObserver(board=ClientBoard(), reward_calclator_cls=KyokuScoreReward)
    actions = env.action_space
    model_config = ModelConfig(
        resnet_repeat=40,
        mid_channels=128,
        learning_rate=0.0005,
        batch_size=128,
    )
    
    dahai_agent = DahaiTrainableAgent(actions["dahai_agent"], Head34SlModel, model_config)
    reach_agent = FixPolicyAgent(np.array([0.0, 1.0])) # always do reach
    chi_agent = FixPolicyAgent(np.array([1.0, 0.0])) # never chi
    pon_agent = FixPolicyAgent(np.array([1.0, 0.0])) # never pon
    kan_agent = FixPolicyAgent(np.array([1.0, 0.0])) # never kan
    
    mj_agent = MjAgent(dahai_agent, reach_agent, chi_agent, pon_agent, kan_agent)
    
    agent_wrapper = AgentTcpWrapper(
        "localhost",
        48000,
        "pass",
        env=env,
        agent=mj_agent,
        room_name="manue1",
        )

    agent_wrapper.connect()
    agent_wrapper.recieve_thread.join()