#Aqui vocês irão colocar seu algoritmo de aprendizado
from connection import connect, get_state_reward
import random
import time
import pickle
from numpy import argmax


Q = pickle.load(open("q_table.pkl", "rb"))
PORT = 2037
ACTIONS = ["left", "right", "jump"]
DIR = ["North", "East", "South", "West"]

def get_action_index(action):
    global ACTIONS
    return ACTIONS.index(action)

def to_int(state):
    return int(str(state), 2)

socket = connect(PORT)

_ = get_state_reward(socket, "left") 
state, _ = get_state_reward(socket, "right") 
s = to_int(state)

while True:
    action = ACTIONS[argmax(Q[s])]
    
    state, r = get_state_reward(socket, action) 
    s_new = to_int(state)
    print(f"{action} - New State: (Plat. {s_new // 4} {DIR[s_new % 4]}) - Reward: {r}")

    s = s_new

    if r in [-100, 300]:
        # Dead or won
        print(f"Trial finished")
        break

socket.close()
