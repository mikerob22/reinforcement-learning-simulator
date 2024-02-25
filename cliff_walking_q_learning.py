import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from gymnasium.wrappers import RecordVideo
# import os
# os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"


def run(episodes, epsilon, learning_rate, discount_factor, is_training):

    env = gym.make("CliffWalking-v0", render_mode=None if is_training else 'rgb_array')

    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))  # init cliff walking env array
    else:
        env = RecordVideo(env, 'static/uploads')
        f = open("static/uploads/cliff_walking_QLearning.pkl", "rb")
        q = pickle.load(f)
        f.close()

    # learning_rate_a = 0.9  # alpha / learning rate
    # discount_factor_g = 0.9  # gamma/discount rate. Near 0: more weight put on now state. Near 1: more on future state
    # epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 0.0001        # epsilon decay rate. 1/0.0001 = 10,000
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]  # states
        terminated = False      # True when fall off cliff or reached goal
        truncated = False       # True when actions > 200
        total_reward = 0

        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()  # actions: 0=left,1=down,2=right,3=up
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)
            #  print("New state: ", new_state, "Reward: ", reward)  # Debug print

            # Update rewards at each step
            total_reward += reward

            if is_training:
                q[state, action] = q[state, action] + learning_rate * (
                    reward + discount_factor * np.max(q[new_state, :]) - q[state, action]
                )

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if epsilon == 0:
            learning_rate = 0.0001

        rewards_per_episode[i] = total_reward

        # Print rewards for each episode
        # print("Episode {}: Total Reward {}".format(i, total_reward))

    env.close()
    # print(rewards_per_episode)

    if is_training:
        sum_rewards = np.zeros(episodes)
        window_size = 100  # Adjust the window size as needed

        for t in range(episodes):
            sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - window_size + 1):(t + 1)])

        # Calculate the moving average
        moving_avg = sum_rewards / window_size

        # Plot the moving average
        plt.plot(moving_avg, label='Moving Average (window_size={})'.format(window_size))
        plt.xlabel('Episodes')
        plt.ylabel('Sum of Rewards')
        plt.legend()
        plt.title('Training Progress - Q-Learning')
        plt.savefig('static/uploads/cliff_walking_QLearning.png')
        plt.show()

    if is_training:
        f = open("static/uploads/cliff_walking_QLearning.pkl", "wb")
        pickle.dump(q, f)
        f.close()

