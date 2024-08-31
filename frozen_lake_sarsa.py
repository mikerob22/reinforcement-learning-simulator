import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pickle
from gymnasium.wrappers import RecordVideo


def run(episodes, epsilon, learning_rate, discount_factor, is_training):

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode=None if is_training else 'rgb_array')

    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))  # init a 64 x 4 array
    else:
        env = RecordVideo(env, 'static/uploads')
        f = open("static/uploads/frozen_lake_sarsa.pkl", "rb")
        q = pickle.load(f)
        f.close()

    # epsilon-greedy policy
    def policy(state, explore=0.0):
        action = int(np.argmax(q[state]))
        if np.random.random() <= explore:
            action = int(np.random.randint(low=0, high=4, size=1))
        return action

    # learning_rate_a = 0.9  # alpha / learning rate
    # discount_factor_g = 0.9  # gamma/discount rate. Near 0: more weight put on now state.Near 1: more on future state.
    # epsilon = 1         # 1 = 100% random actions
    # epsilon_decay_rate = 0.0001        # epsilon decay rate. 1/0.0001 = 10,000

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
        total_reward_per_eps = 0
        terminated = False      # True when fall in hole or reached goal
        truncated = False       # True when actions > some amount of steps

        action = policy(state, epsilon)

        while not terminated and not truncated:
            new_state, reward, terminated, truncated, _ = env.step(action)

            total_reward_per_eps += reward

            # print(f"State: {state}, Action: {action}, Q-values: {q[state]}, Reward: {reward}, New State: {new_state}")

            if is_training:
                next_action = policy(new_state, epsilon)

                q[state][action] += learning_rate * (
                        reward + discount_factor * q[new_state][next_action] - q[state][action])
                action = next_action
            else:
                action = np.argmax(q[state, :])

            state = new_state

        # epsilon = max(epsilon - epsilon_decay_rate, 0)

        # if epsilon == 0:
        #     learning_rate_a = 0.0001
        rewards_per_episode[i] = total_reward_per_eps
        # if reward == 1:
        #     rewards_per_episode[i] = 1

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
        plt.plot(moving_avg)
        # plt.plot(rewards_per_episode, label='Rewards per Episode')
        plt.xlabel('Episodes')
        plt.ylabel('Sum of Rewards')
        plt.title('Training Progress : Frozen Lake - SARSA')
        plt.savefig('static/uploads/frozen_lake_sarsa.png')
        # plt.show()

    if is_training:
        f = open("static/uploads/frozen_lake_sarsa.pkl", "wb")
        pickle.dump(q, f)
        f.close()
