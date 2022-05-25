from flask import Flask, render_template, url_for, redirect, request
import json
import os
import re

app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route('/', methods=['GET'])
def index():
    return redirect(url_for('ml_demo'))

@app.route('/ml-demo', methods=['GET', 'POST'])
def ml_demo():
    if request.method == 'GET':
        return render_template('ml_demo.html', discussions=dummy_data)

    if request.method == 'POST':
        query = request.form['query']
        query_list = preprocess_text_data(query)
        return render_template('algorithm_demo.html', 
                                query=query,
                                discussions=dummy_data,
                                preprocessed_query_list=query_list)

@app.route('/algorithm-demo', methods=['GET', 'POST'])
def algorithm_demo():
    if request.method == 'GET':
        return render_template('algorithm_demo.html', discussions=dummy_data)

    if request.method == 'POST':
        query = request.form['query']
        query_list = preprocess_text_data(query)
        return render_template('algorithm_demo.html', 
                                query=query,
                                discussions=dummy_data,
                                preprocessed_query_list=query_list)

def init_dummy_data():
    # Get the JSON Path:
    # NOTE: __file__ is the absolute path of this app.py
    # NOTE: os.pardir is the parent directory (..)
    json_path = os.path.join(__file__, os.pardir, "discussion_dummy_data.json")
    print(json_path)

    # Open the JSON Dummy Data:
    with open(json_path, 'r') as file:
        dummy_data = json.load(file)

    return dummy_data

def preprocess_text_data(query):
    preprocessed_query = query.lower()
    
    regex = r"\b[a-z]+\b"
    preprocessed_query_list = re.findall(regex, preprocessed_query)

    return preprocessed_query_list

if __name__ == "__main__":
    # Initialize the JSON dummy data:
    dummy_data = init_dummy_data()

    # Run the Flask App:
    app.run(debug=True, port=5000)