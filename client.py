#Aqui vocês irão colocar seu algoritmo de aprendizado
from connection import connect, get_state_reward
import random
import time
import pickle
from os import path
import numpy as np

PORT = 2037
ALPHA = 0.2
GAMMA = 0.99
N_TRIALS = 50000
ACTIONS = ["left", "right", "jump"]
TERM = {-100: "DEAD", 300: "WON"}
DIR = ["North", "East", "South", "West"]

def get_action_index(action):
    global ACTIONS
    return ACTIONS.index(action)

def to_int(state):
    return int(str(state), 2)
    
def get_next_action(state, eps):
    global Q
    return np.random.choice(ACTIONS) if np.random.random() < eps else np.argmax(Q[state])

socket = connect(PORT)

if path.exists("q_table.pkl"):
    Q = pickle.load(open("q_table.pkl", "rb"))
else:
    Q = {i: [0, 0, 0] for i in range(96)} # initializer q table

eps = 1
for t in range(N_TRIALS):
    time.sleep(0.01)
    _ = get_state_reward(socket, "left") 
    s = get_state_reward(socket, "right")[0] # get initial state
    s = to_int(s)

    while True:
        action = get_next_action(s, eps)

        s_new, r = get_state_reward(socket, action) 
        s_new = to_int(s_new)
        print(f"{action} - New State: (Plat. {s_new // 4} {DIR[s_new % 4]}) - Reward: {r}")
        
        a = get_action_index(action)
        
        
        Q[s][a] = Q[s][a] + ALPHA * (r + GAMMA * max(Q[s_new]) - Q[s][a])
        
        s = s_new

        if r in [-100, 300]:
            # Dead or won
            print(f"[{TERM[r]}] Trial {t + 1}/{N_TRIALS} finished")
            break
    eps = max(0.1, eps - (0.5 / N_TRIALS))

    with open("q_table.pkl", "wb") as file:
        pickle.dump(Q, file)

socket.close()
