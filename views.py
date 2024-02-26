from flask import Blueprint, render_template, request, send_from_directory
import time
import pygame

views = Blueprint("views", __name__)

# Pygame initialization
pygame.init()

# Define available environments
ENVIRONMENTS = {
    "CliffWalking-v0": "Cliff Walking",
    "FrozenLake-v1": "Frozen Lake",
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
    # Add more environment-algorithm mappings as needed
}


def process_form_data(request_data):
    # Get form data
    environment = request_data.form["environment"]
    algorithm = request_data.form["algorithm"]
    episodes = int(request_data.form["episodes"])
    epsilon = float(request_data.form["epsilon"])
    learning_rate = float(request_data.form["learning_rate"])
    discount_factor = float(request_data.form["discount_factor"])
    mode = request_data.form["mode"]
    # Determine if training mode or not based on user's selection
    is_training = (mode == "train")

    return environment, algorithm, episodes, epsilon, learning_rate, discount_factor, mode, is_training


@views.route("/")  # the main route
def home():
    return render_template("home.html", environments=ENVIRONMENTS, algorithms=ALGORITHMS)


@views.route("/static/uploads/<path:filename>")
def serve_static(filename):
    return send_from_directory('static/uploads', filename)


@views.route("/simulate", methods=["POST"])
def simulate():
    # Get form data
    environment, algorithm, episodes, epsilon, learning_rate, discount_factor, mode, is_training = process_form_data(request)

    # Measure the time taken for training
    start_time = time.time()

    # Run simulation
    simulation = run_algorithm(environment, algorithm, episodes, epsilon, learning_rate, discount_factor, is_training)

    # Calculate the training duration
    end_time = time.time()
    training_duration = round(end_time - start_time, 3)

    # Determine the file path based on the selected algorithm
    if algorithm == "QLearning" and environment == "CliffWalking-v0":
        figure_path = "static/uploads/cliff_walking_Qlearning.png"
    elif algorithm == "SARSA" and environment == "CliffWalking-v0":
        figure_path = "static/uploads/cliff_walking_sarsa.png"
    elif algorithm == "QLearning" and environment == "FrozenLake-v1":
        figure_path = "static/uploads/frozen_lake_QLearning.png"
    elif algorithm == "SARSA" and environment == "FrozenLake-v1":
        figure_path = "static/uploads/frozen_lake_sarsa.png"

    # Render HTML template with training information
    return render_template("training_result.html", environments=ENVIRONMENTS, algorithms=ALGORITHMS,
                           episodes=episodes, epsilon=epsilon, learning_rate=learning_rate,
                           discount_factor=discount_factor, simulation=simulation,
                           training_duration=training_duration, figure_path=figure_path)


@views.route("/testing", methods=["POST"])
def testing():
    # Get form data
    environment, algorithm, episodes, epsilon, learning_rate, discount_factor, mode, is_training = process_form_data(request)

    test = run_algorithm(environment, algorithm, episodes=1, epsilon=epsilon, learning_rate=learning_rate,
                         discount_factor=discount_factor, is_training=False)

    # Determine the file path based on the selected algorithm
    if algorithm == "QLearning":
        video_path = "static/uploads/rl-video-episode-0.mp4"
    elif algorithm == "SARSA":
        video_path = "static/uploads/rl-video-episode-0.mp4"

    # video_url = "/static/uploads/rl-video-episode-0.mp4"

    return render_template("testing_result.html", episodes=episodes, epsilon=epsilon,
                           learning_rate=learning_rate, discount_factor=discount_factor, environments=ENVIRONMENTS,
                           algorithms=ALGORITHMS, test=test, video_path=video_path)


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


# def run_algorithm(environment, episodes, epsilon, learning_rate, discount_factor, is_training):
#     # Define a mapping of environments to corresponding Q-learning implementation files
#     environment_to_q_learning_file = {
#         "CliffWalking-v0": "cliff_walking_q_learning.py",
#         # Add more environment-Q-learning file mappings as needed
#     }
#
#     # Get the Q-learning implementation file for the specified environment
#     q_learning_file = environment_to_q_learning_file.get(environment)
#
#     if q_learning_file is None:
#         # Handle the case where the environment is not found in the mapping
#         return "Error: Q-learning implementation not found for the specified environment."
#
#     # Import the Q-learning implementation module dynamically
#     try:
#         q_learning_module = __import__(q_learning_file[:-3])  # Remove ".py" extension
#     except ImportError:
#         # Handle the case where the Q-learning implementation file cannot be imported
#         return "Error: Failed to import Q-learning implementation for the specified environment."
#
#     q_learning_module.run(episodes, epsilon, learning_rate, discount_factor, is_training)
#
#     progress_messages = [f"Episode {i + 1} completed." for i in range(episodes)]
#     return progress_messages
