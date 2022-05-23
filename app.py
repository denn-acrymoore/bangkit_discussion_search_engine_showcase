from crypt import methods
from flask import Flask, render_template, url_for, redirect

app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route('/', methods=['GET'])
def index():
    return redirect(url_for('ml_demo'))

@app.route('/ml-demo', methods=['GET', 'POST'])
def ml_demo():
    return render_template('ml_demo.html')

@app.route('/algorithm-demo', methods=['GET', 'POST'])
def algorithm_demo():
    return render_template('algorithm_demo.html')