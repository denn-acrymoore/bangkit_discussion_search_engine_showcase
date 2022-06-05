from flask import Flask, render_template, url_for, redirect, request
import json
import os
import re
import pandas as pd

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
    """Read the dummy data and convert it into a list of objects"""

    # Get the Excel Path:
    # NOTE: __file__ is the absolute path of this app.py
    # NOTE: os.pardir is the parent directory (..)
    dummy_path = os.path.join(__file__, os.pardir, "discussion_dummy_data.xlsx")
    print("Excel Path:", dummy_path, end="\n\n")

    # Open the Excel Dummy Data as Pandas DataFrame:
    dummy_data_df = pd.read_excel(dummy_path)
    print(dummy_data_df.head(), end="\n\n")

    # Convert Pandas DataFrame into Python List of Dictionaries:
    dummy_data = dummy_data_df.to_dict(orient='records')

    # Convert all keywords into list:
    regex = r"\b[a-z]+\b"
    for data in dummy_data:
        data['keywords'] = re.findall(regex, data['keywords'])

    print(dummy_data, end="\n\n")
    
    return dummy_data

def preprocess_text_data(query):
    """Process the text data and output a list of each processed words"""

    preprocessed_query = query.lower()
    
    regex = r"\b[a-z]+\b"
    preprocessed_query_list = re.findall(regex, preprocessed_query)

    return preprocessed_query_list

if __name__ == "__main__":
    # Initialize the JSON dummy data:
    dummy_data = init_dummy_data()

    # Run the Flask App:
    app.run(debug=True, port=5000)