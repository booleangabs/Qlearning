#Aqui vocês irão colocar seu algoritmo de aprendizado
from connection import connect, get_state_reward
import random
import time
import pickle
from os import path
import numpy as np
import matplotlib.pyplot as plt

PORT = 2037
ALPHA = 0.1
GAMMA = 0.99
N_TRIALS = 700
ACTIONS = ["left", "right", "jump"]
TERM = {-100: "DEAD", 300: "WON"}
DIR = ["North", "East", "South", "West"]

def get_action_index(action):
    global ACTIONS
    return ACTIONS.index(action)

def to_int(state):
    return int(str(state), 2)
    
def get_next_action(q_table, state, eps):
    random_a = np.random.choice(ACTIONS)
    best_a = ACTIONS[np.argmax(q_table[state])]
    return random_a if np.random.random() < eps else best_a

socket = connect(PORT)

if path.exists("q_table.pkl"):
    Q = pickle.load(open("q_table.pkl", "rb"))
else:
    Q = {i: [0, 0, 0] for i in range(96)} # initializer q table

eps = 0.7
final_eps = 0.1
decay = (eps - final_eps) / N_TRIALS
rewards = []
for t in range(N_TRIALS):
    time.sleep(0.01)
    _ = get_state_reward(socket, "left") 
    s = get_state_reward(socket, "right")[0] # get initial state
    s = to_int(s)
    initial_s = s
    print(f"Initial State: (Plat. {s // 4} {DIR[s % 4]})")

    total_r = 0
    n_actions = 0
    while True:
        action = get_next_action(Q, s, eps)

        s_new, r = get_state_reward(socket, action) 
        s_new = to_int(s_new)
        print(f"{action} - New State: (Plat. {s_new // 4} {DIR[s_new % 4]}) - Reward: {r}")
        
        a = get_action_index(action)
        
        total_r += r
        n_actions += 1
        
        if r in [-100, 300]:
            # Dead or finished
            Q[s][a] = (1 - ALPHA) * Q[s][a] + ALPHA * r
            print(f"[{TERM[r]}] Trial {t + 1}/{N_TRIALS} finished")
            break
        else:
            Q[s][a] = (1 - ALPHA) * Q[s][a] + ALPHA * (r + GAMMA * max(Q[s_new]))
        
        s = s_new
            
    rewards.append(total_r / n_actions)
    eps -= decay 

    with open("q_table.pkl", "wb") as file:
        pickle.dump(Q, file)
plt.plot(rewards)
plt.show()

socket.close()
