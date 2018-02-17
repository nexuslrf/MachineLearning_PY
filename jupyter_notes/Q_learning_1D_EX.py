import numpy as np
import pandas as pd
import time

np.random.seed(2) #reproducible

N_STATES = 6   # the length of the 1 dimensional world
ACTIONS = ['left', 'right']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 13   # maximum episodes
FRESH_TIME = 0.3    # fresh time for one move

def build_q_table(n_state, actions):
	table = pd.DataFrame(data=np.zeros((n_state,len(actions))), index=None, columns=actions, dtype=None, copy=False)
	print(table)
	return table

def choose_action(state, q_table):
	

build_q_table(N_STATES,ACTIONS)