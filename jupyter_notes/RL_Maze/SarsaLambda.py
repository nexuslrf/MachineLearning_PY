from maze_env import Maze
from RL_brain import SarsaLambdaTable

def update():
    cnt_success = 0
    cnt_failure = 0
    for episode in range(50):
        # reset environment
        observation = env.reset()

        # RL choose action based on observation 
        # Differ from Qlearning
        action = RL.choose_action(str(observation))

        # initial all zero eligibility trace
        RL.eligibility_trace *= 0

        while True:
            env.render()
            #  # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap
            observation = observation_
            action = action_

            if done:
                if reward == 1:
                    cnt_success += 1
                elif reward == -1:
                    cnt_failure +=1
                break

    print("game over")
    print("success: %d\nfailure: %d"%(cnt_success,cnt_failure))
    print(RL.q_table)
    env.destroy()


if __name__ == "__main__":
    env = Maze('SarsaLambda')
    RL = SarsaLambdaTable(actions = list(range(env.n_actions)),e_greedy=0.9)

    env.after(50,update)
    env.mainloop()