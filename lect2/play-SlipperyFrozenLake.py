import gym
from gym.envs.registration import register
import readchar

#MACROS
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

#Key mapping
arrow_keys = {
    '\x1b[A' : UP,
    '\x1b[B' : DOWN,
    '\x1b[C' : RIGHT,
    '\x1b[D' : LEFT,
}


# game settings
env = gym.make('FrozenLake-v0')
env.render()
env.reset()

while True:
    key = readchar.readkey()
    if key not in arrow_keys.keys():
        print("Game aborted!")
        break

    action = arrow_keys[key]
    state, reward, done, info = env.step(action)
    env.render()
    print("State: ", state, "Action: ", action, "Reward: ", reward, "Info: ", info)

    if done:
        print("Finished with reward", reward)
        break
