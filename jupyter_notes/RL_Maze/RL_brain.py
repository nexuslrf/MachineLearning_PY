import numpy as np
import pandas as pd

class RL(object):
    def __init__(self, actions,learning_rate=0.01,reward_decay=0.9,e_greedy=0.8):
        self.actions = actions # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns = self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state in the q_table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index = self.q_table.columns,
                    name = state
                    )
                )

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # select an action
        if np.random.uniform()<self.epsilon:
            state_action = self.q_table.loc[observation,:]
            state_action = state_action.reindex(np.random.permutation(state_action.index)) # some actions have the same value to make them disordered 
            action=state_action.idxmax()
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass

# off-policy
class QLearningTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9,e_greedy=0.8):
        super().__init__(actions, learning_rate, reward_decay, e_greedy)
    
    def learn(self, s,a,r,s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s,a]
        if s_ != 'terminal':
            q_target = r + self.gamma*self.q_table.loc[s_,:].max()
        else:
            q_target = r
        self.q_table.loc[s,a] += self.lr * (q_target - q_predict)

# on-policy
class SarsaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9,e_greedy=0.8):
        super().__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s,a,r,s_,a_): # add a_ to the learn compared with Qlearning
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s,a]
        if s_ != 'terminal':
            q_target = r + self.gamma*self.q_table.loc[s_,a_]
        else:
            q_target = r
        self.q_table.loc[s,a] += self.lr * (q_target - q_predict)

class SarsaLambdaTable(RL):
    # here trace_decay represent lambda
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9,e_greedy=0.8,trace_decay=0.9):
        super().__init__(actions, learning_rate, reward_decay, e_greedy)

         # 后向观测算法, eligibility trace.
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()    # 空的 eligibility trace 表

    #check_state_exist 和之前的是高度相似的. 唯一不同的地方是我们考虑了 eligibility_trace,
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            to_be_append = pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            self.q_table = self.q_table.append(to_be_append)

            # also update eligibility trace
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def learn(self, s,a,r,s_,a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s,a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_,a_]
        else:
            q_target = r
        error = q_target - q_predict

        # increase trace amount for visited state-action pair

        # Method 1:
        # self.eligibility_trace.loc[s, a] += 1

        # Method 2:
        self.eligibility_trace.loc[s, :] *= 0
        self.eligibility_trace.loc[s, a] = 1

        # Q update
        # if error>0:
        self.q_table += self.lr * error * self.eligibility_trace

        # decay eligibility trace after update
        self.eligibility_trace *= self.gamma*self.lambda_

    