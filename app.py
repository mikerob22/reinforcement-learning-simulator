# app.py
import os
from flask import Flask, render_template, request, send_file, jsonify, send_from_directory
from views import views


def create_app():
    app = Flask(__name__)

    app.register_blueprint(views, url_prefix="/")

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)

