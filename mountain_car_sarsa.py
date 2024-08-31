import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pickle
from gymnasium.wrappers import RecordVideo


def run(episodes, epsilon, learning_rate, discount_factor, is_training):

    env = gym.make('MountainCar-v0', render_mode=None if is_training else 'rgb_array')

    # Divide position and velocity into segments
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)    # Between -1.2 and 0.6
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)    # Between -0.07 and 0.07

    if is_training:
        q = np.zeros((len(pos_space), len(vel_space), env.action_space.n)) # init a 20x20x3 array
    else:
        env = RecordVideo(env, 'static/uploads')
        f = open('static/uploads/mountain_car_sarsa.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    # learning_rate_a = 0.9  # alpha or learning rate
    # discount_factor_g = 0.9  # gamma or discount factor.
    #
    # epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 2/episodes  # epsilon decay rate
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]      # Starting position, starting velocity always 0
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)

        terminated = False          # True when reached goal
        rewards = 0

        # Choose initial action
        if is_training and rng.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q[state_p, state_v, :])

        while not terminated and rewards > -1000:
            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            if is_training and rng.random() < epsilon:
                # Choose random action (0=drive left, 1=stay neutral, 2=drive right)
                new_action = env.action_space.sample()
            else:
                new_action = np.argmax(q[new_state_p, new_state_v, :])

            if is_training:
                q[state_p, state_v, action] = q[state_p, state_v, action] + learning_rate * (
                    reward + discount_factor*np.max(q[new_state_p, new_state_v, new_action]) - q[state_p, state_v, action]
                )

            # state = new_state
            state_p = new_state_p
            state_v = new_state_v

            action = new_action

            rewards += reward

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        rewards_per_episode[i] = rewards

    env.close()

    # Save Q table to file
    if is_training:
        f = open('static/uploads/mountain_car_sarsa.pkl', 'wb')
        pickle.dump(q, f)
        f.close()

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
        plt.title('Training Progress : Mountain Car - SARSA')
        plt.savefig('static/uploads/mountain_car_sarsa.png')
        print("Figure saved successfully!")
        # plt.show()


# if __name__ == '__main__':
    # run(1, 1, 0.9, 0.9, is_training=False)

    # run(5000, 1, 0.9, 0.9, is_training=True)
