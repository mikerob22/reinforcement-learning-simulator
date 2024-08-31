from flask import Blueprint, render_template, request, send_from_directory
import time
import pygame

from functions import *

views = Blueprint("views", __name__)

# Pygame initialization
pygame.init()


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
    figure_path = get_figure_path(environment, algorithm)

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
    video_path = get_video_path(algorithm)

    return render_template("testing_result.html", episodes=episodes, epsilon=epsilon,
                           learning_rate=learning_rate, discount_factor=discount_factor, environments=ENVIRONMENTS,
                           algorithms=ALGORITHMS, test=test, video_path=video_path)


@views.route("/environments")
def environments():
    return render_template("environments.html")


@views.route("/about")
def about():
    return render_template("about.html")


@views.route("/contact")
def contact():
    return render_template("contact.html")




