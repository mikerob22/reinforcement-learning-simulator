import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo
import pickle


def run(episodes, epsilon, learning_rate, discount_factor, is_training):

    env = gym.make("CliffWalking-v0", render_mode=None if is_training else 'rgb_array')

    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))  # init cliff walking env array
    else:
        env = RecordVideo(env, 'static/uploads')
        f = open("static/uploads/cliff_walking_sarsa.pkl", "rb")
        q = pickle.load(f)
        f.close()

    # epsilon-greedy policy
    def policy(state, explore=0.0):
        action = int(np.argmax(q[state]))
        if np.random.random() <= explore:
            action = int(np.random.randint(low=0, high=4, size=1))
        return action

    # PARAMETERS
    epsilon_decay_rate = 0.0001
    # EPSILON = 0.1       # 1 = argmax, pick optimal action
    # ALPHA = 0.1
    # GAMMA = 1

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]  # initial state
        total_reward_per_eps = 0
        terminated = False      # True when fall off cliff or reached goal
        truncated = False       # True when actions > 200

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
            learning_rate = 0.0001

        rewards_per_episode[i] = total_reward_per_eps

        # print("Episode:", i, "Total Reward:", total_reward_per_eps, "Optimal Policy Action:", action)

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
        plt.title('Training Progress - SARSA')
        plt.savefig('static/uploads/cliff_walking_sarsa.png')
        plt.show()

    if is_training:
        f = open('static/uploads/cliff_walking_sarsa.pkl', "wb")
        pickle.dump(q, f)
        f.close()
        print("Training Complete. Q Table saved.")
