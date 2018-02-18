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
	#This is how to choose an action
    state_actions = q_table.iloc[state,:]
    if (np.random.uniform()> EPSILON) or ((state_actions == 0).all()): # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else: # act greedy
        action_name = state_actions.idxmax()
         # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name

# 做出行为后, 环境也要给我们的行为一个反馈, 反馈出下个 state (S_) 和 在上个 state (S) 做出 action (A) 所得到的 reward (R).
# 这里定义的规则就是, 只有当 o 移动到了 T, 探索者才会得到唯一的一个奖励, 奖励值 R=1, 其他情况都没有奖励.
def get_env_feedback(S,A):
    # This is how agent will interact with the environment
    if A == 'right':  # move right
        if S == N_STATES - 2: #terminate
            S_ = 'terminal'
            R = 1       
        else:
            S_ = S+1
            R = 0
    else: # move left
        R = 0
        if S == 0:
            S_ = S # reach the wall
        else:
            S_ = S - 1
    return S_, R

def update_env(S,episode,step_counter):
    env_list = ['-']*(N_STATES-1)+['T'] # '---------T'
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction),end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction),end='')
        time.sleep(FRESH_TIME)

def r1():
    # main part of RL loop
    q_table = build_q_table(N_STATES,ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0   #  回合初始位置
        is_terminated = False  # 回合是否结束
        update_env(S,episode,step_counter) # 环境更新
        while not is_terminated:
            A = choose_action(S,q_table)
            S_, R = get_env_feedback(S,A)
            q_predict = q_table.loc[S,A] # 估算的（s-A）值
            if S_ != 'terminal':
                q_target = R + GAMMA* q_table.iloc[S_,:].max() #  实际的(状态-行为)值 (回合没结束)
            else:
                q_target = R # next state is terminal
                is_terminated = True

            q_table.loc[S,A]+=ALPHA*(q_target- q_predict) # update
            S = S_ #move to next state

            update_env(S,episode,step_counter+1)
            step_counter +=1
    return q_table

if __name__ == '__main__':
    q_table = r1()
    print('\r\nQ_table:\n')
    print(q_table)
