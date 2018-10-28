import gym

env = gym.make("MountainCar-v0")
env.reset()

random_episodes = 0
reward_sum = 0

while random_episodes < 10:
    env.render()

    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    #print(observation, reward, done)

    reward_sum += reward

    if done:
        random_episodes += 1
        print("Reward for ", random_episodes, "th episode was ", reward_sum)
        reward_sum = 0
        env.reset()