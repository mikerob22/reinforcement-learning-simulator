def get_figure_path(environment, algorithm):
    if algorithm == "QLearning" and environment == "CliffWalking-v0":
        return "static/uploads/cliff_walking_Qlearning.png"
    elif algorithm == "SARSA" and environment == "CliffWalking-v0":
        return "static/uploads/cliff_walking_sarsa.png"
    elif algorithm == "QLearning" and environment == "FrozenLake-v1":
        return "static/uploads/frozen_lake_QLearning.png"
    elif algorithm == "SARSA" and environment == "FrozenLake-v1":
        return "static/uploads/frozen_lake_sarsa.png"


def get_video_path(algorithm):
    if algorithm == "QLearning":
        return "static/uploads/rl-video-episode-0.mp4"
    elif algorithm == "SARSA":
        return "static/uploads/rl-video-episode-0.mp4"
