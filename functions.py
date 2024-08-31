# Define available environments
ENVIRONMENTS = {
    "CliffWalking-v0": "Cliff Walking",
    "FrozenLake-v1": "Frozen Lake",
    "MountainCar-v0": "Mountain Car",
    # Add more environments as needed
}

# Define available algorithms
ALGORITHMS = {
    "QLearning": "Q-Learning",
    "SARSA": "SARSA",
    # Add more environments as needed
}

MAPS = {
    "8x8": "8 x 8",
    "5x5": "5 x 5"
}

# Define mappings between environment and the corresponding Python files containing algorithm implementations
ENVIRONMENT_TO_ALGORITHM_MODULE = {
    "CliffWalking-v0": {
        "QLearning": "cliff_walking_q_learning",
        "SARSA": "cliff_walking_sarsa",
    },
    "FrozenLake-v1": {
        "QLearning": "frozen_lake_q_learning",
        "SARSA": "frozen_lake_sarsa",
    },
    "MountainCar-v0": {
        "QLearning": "mountain_car_q_learning",
        "SARSA": "mountain_car_sarsa",
    },
    # Add more environment-algorithm mappings as needed
}


# Function to run the selected environment and algorithm
def run_algorithm(environment, algorithm, episodes, epsilon, learning_rate, discount_factor, is_training):
    # Check if the environment is supported
    if environment in ENVIRONMENT_TO_ALGORITHM_MODULE:
        # Check if the algorithm is supported for the environment
        if algorithm in ENVIRONMENT_TO_ALGORITHM_MODULE[environment]:
            module_name = ENVIRONMENT_TO_ALGORITHM_MODULE[environment][algorithm]
            try:
                # Dynamically import the module
                module = __import__(module_name)
                # Call the function to run the algorithm
                module.run(episodes, epsilon, learning_rate, discount_factor, is_training)
                return "Algorithm executed successfully."
            except ImportError:
                return f"Failed to import module {module_name}."
        else:
            return f"Algorithm '{algorithm}' is not supported for the '{environment}' environment."
    else:
        return f"Environment '{environment}' is not supported."


def get_figure_path(environment, algorithm):
    if algorithm == "QLearning" and environment == "CliffWalking-v0":
        return "static/uploads/cliff_walking_Qlearning.png"
    elif algorithm == "SARSA" and environment == "CliffWalking-v0":
        return "static/uploads/cliff_walking_sarsa.png"
    elif algorithm == "QLearning" and environment == "FrozenLake-v1":
        return "static/uploads/frozen_lake_QLearning.png"
    elif algorithm == "SARSA" and environment == "FrozenLake-v1":
        return "static/uploads/frozen_lake_sarsa.png"
    elif algorithm == "QLearning" and environment == "MountainCar-v0":
        return "static/uploads/mountain_car_QLearning.png"
    elif algorithm == "SARSA" and environment == "MountainCar-v0":
        return "static/uploads/mountain_car_sarsa.png"


def get_video_path(algorithm):
    if algorithm == "QLearning":
        return "static/uploads/rl-video-episode-0.mp4"
    elif algorithm == "SARSA":
        return "static/uploads/rl-video-episode-0.mp4"

