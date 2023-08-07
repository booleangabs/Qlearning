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
    s = str(state)
    platform = s[2:7]
    orientation = s[7:]
    return int(f"0b{platform}", 2), int(f"0b{orientation}", 2)

socket = connect(PORT)

_ = get_state_reward(socket, "left") 
state, _ = get_state_reward(socket, "right") 
p, d = to_int(state)
s = p * 4 + d

while True:
    action = ACTIONS[argmax(Q[s])]
    
    state, r = get_state_reward(socket, action) 
    p, d = to_int(state)
    print(f"{action} - New State: (Plat. {p} {DIR[d]}) - Reward: {r}")

    s = p * 4 + d

    if r in [-100, 300]:
        # Dead or won
        print(f"Trial finished")
        break

socket.close()
