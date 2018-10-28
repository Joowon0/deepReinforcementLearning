import gym
import numpy as np
import tensorflow as tf

env = gym.make('CartPole-v0')

# Input and output size based on the Env
input_size = env.observation_space.shape[0] # 4 : don;t know what state means
output_size = env.action_space.n # 2 : left right
learning_rate = 0.1

# These lines establish the feed-forward part of the network used to choose actions
X = tf.placeholder(shape=[None, input_size], dtype=tf.float32, name="input_x") # None : could be 1~4, but 1 in this case
Weight = tf.get_variable("W1", shape=[input_size, output_size],
                     initializer=tf.contrib.layers.xavier_initializer())

Qpred = tf.matmul(X, Weight)
Y = tf.placeholder(shape=[None, output_size], dtype=tf.float32)

loss = tf.reduce_sum(tf.square(Y - Qpred))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Set Q-learning related parameters
discountedReward = .99
num_episodes = 5000

# Create lists to contain total rewards and steps for each episode
rList = []

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    e = 1. / ((i / 10) + 1)
    step_count = 0
    done = False

    # The Q-Network training
    while not done:
        step_count += 1
        x = np.reshape(s, [1, input_size])

        # Find out Q^ for all actions in state s
        Qs = sess.run(Qpred, feed_dict={X: x})
        # Choose an action by E-greedy algorithm from the Q-network
        if np.random.rand(1) < e:
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)

        # Get new state and reward from executing a
        s1, reward, done, _ = env.step(a)
        # Update Q_s with new value, using reward
        if done: # terminal node
            Qs[0, a] = -100 # pole 쓰러짐 => 굉장히 잘못했다!!
        else:
            x1 = np.reshape(s1, [1, input_size])
            # Obtain Q_s1 values
            Qs1 = sess.run(Qpred, feed_dict={X: x1})
            Qs[0, a] = reward + discountedReward * np.max(Qs1)

        # Train our network using target (Y) and predicted Q (Qpred) values
        sess.run(train, feed_dict={X: x, Y: Qs})
        s = s1
    rList.append(step_count)
    print("Episode: {} steps: {}".format(i, step_count))

    # If last 10's avg steps are 500, it's good enough
    if len(rList) > 10 and np.mean(rList[-10:]) > 500:
        break


# See our trained network in action
num_episodes = 10

for i in range(num_episodes):
    observation = env.reset()
    reward_sum = 0
    while True:
        # env.render()

        x = np.reshape(observation, [1, input_size])
        Qs = sess.run(Qpred, feed_dict={X: x})
        a = np.argmax(Qs)

        observation, reward, done, _ = env.step(a)
        reward_sum += reward
        if done:
            print("Total score: {}". format(reward_sum))
            break

sess.close()