# app.py
import os
from flask import Flask, render_template, request, send_file, jsonify, send_from_directory
from views import views
# from werkzeug.serving import run_simple


app = Flask(__name__)

app.register_blueprint(views, url_prefix="/views")


if __name__ == '__main__':
    app.run(debug=True)
