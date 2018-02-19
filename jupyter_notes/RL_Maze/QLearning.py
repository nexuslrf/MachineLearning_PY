"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
"""

from maze_env import Maze
from RL_brain import QLearningTable

def update():
    cnt_success = 0
    cnt_failure = 0
    for episode in range(50):
        # print(RL.q_table,'\n')
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

             # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break this loop
            if done:
                if reward == 1:
                    cnt_success += 1
                elif reward == -1:
                    cnt_failure +=1
                break

    # end of game
    print("game over")
    print("success: %d\nfailure: %d"%(cnt_success,cnt_failure))
    print(RL.q_table)
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions = list(range(env.n_actions)))

    env.after(50,update)
    env.mainloop()