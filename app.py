from flask import Flask, render_template, url_for, redirect, request
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import string
import re
import nltk
import copy
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


""" Flask Related Functions """
app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route('/', methods=['GET'])
def index():
    return redirect(url_for('ml_onenum_demo'))

@app.route('/ml-onenum-demo', methods=['GET', 'POST'])
def ml_onenum_demo():
    if request.method == 'GET':
        return render_template('ml_onenum_demo.html', discussions=dummy_data)

    if request.method == 'POST':
        query = request.form['query']
        if not query:
            return render_template('ml_onenum_demo.html', discussions=dummy_data)

        preprocessed_query = preprocessing(query)
        return render_template('ml_onenum_demo.html', 
                                query=query,
                                discussions=dummy_data,
                                preprocessed_query=preprocessed_query)

@app.route('/ml-multinum-demo', methods=['GET', 'POST'])
def ml_multinum_demo():
    if request.method == 'GET':
        print(dummy_data)
        return render_template('ml_multinum_demo.html', discussions=dummy_data)

    if request.method == 'POST':
        query = request.form['query']
        if not query:
            return render_template('ml_multinum_demo.html', discussions=dummy_data)

        # Preprocess the query
        preprocessed_query = preprocessing(query)

        # Convert the query into multinum data
        multinum_query = multinum_lstm_model.predict([preprocessed_query])[0]

        # Create a deep copy of the dummy_data
        dummy_data_copy = copy.deepcopy(dummy_data)

        # Predict the relevancy of each discussion data
        for data in dummy_data_copy:
            input_data = np.append(data['multinum_value'], multinum_query)
            input_data = np.expand_dims(input_data, axis=0)
            prediction = multinum_dense_model.predict(input_data)[0]
            data['relevant'] = true_binary_prediction_converter(prediction)

        print(dummy_data_copy)

        return render_template('ml_multinum_demo.html', 
                                query=query,
                                discussions=dummy_data_copy,
                                preprocessed_query=preprocessed_query)                              

@app.route('/algorithm-demo', methods=['GET', 'POST'])
def algorithm_demo():
    if request.method == 'GET':
        return render_template('algorithm_demo.html', discussions=dummy_data)

    if request.method == 'POST':
        query = request.form['query']
        if not query:
            return render_template('algorithm_demo.html', discussions=dummy_data)

        preprocessed_query = preprocessing(query)
        return render_template('algorithm_demo.html', 
                                query=query,
                                discussions=dummy_data,
                                preprocessed_query=preprocessed_query)


""" Helper Functions """
def convert_stringlist_to_list(stringlist):
    """Convert a string representing a list into actual python float list"""
    stringlist = stringlist.strip("[] ").split()
    floatlist = list(map(lambda x: float(x), stringlist))

    return floatlist

def true_binary_prediction_converter(np_array):
    if np_array[0] >= 0.5:
        return 1
    
    else:
        return 0

""" Initialization Functions """
def init_dummy_data():
    """Read the dummy data and convert it into a list of objects"""

    # Get the Excel Path:
    # NOTE: __file__ is the absolute path of this app.py
    # NOTE: os.pardir is the parent directory (..)
    DUMMY_PATH = os.path.join(__file__, os.pardir, "discussion_dummy_data_v2.xlsx")
    print("Excel Path:", DUMMY_PATH, end="\n\n")

    # Open the Excel Dummy Data as Pandas DataFrame:
    dummy_data_df = pd.read_excel(DUMMY_PATH)

    # Convert Pandas DataFrame into Python List of Dictionaries [{column -> value}, ...]:
    dummy_data = dummy_data_df.to_dict(orient='records')

    # Convert all keywords into list:
    regex = r"\b[a-z]+\b"
    for data in dummy_data:
        data['keywords'] = re.findall(regex, data['keywords'])
        data['multinum_value'] = np.array(convert_stringlist_to_list(data['multinum_value']))
        data['relevant'] = 1
    
    return dummy_data

def init_onenum_models():
    """Load the One-Num Models (Untrained LSTM and Trained Dense Models)"""
    pass

def init_multinum_models():
    """Load the Multi-Num Models (Untrained LSTM and Trained Dense Models)"""
    LSTM_MODEL_PATH = os.path.join(__file__, os.pardir, "multi_num_untrained_LSTM_model_v1")
    DENSE_MODEL_PATH = os.path.join(__file__, os.pardir, "multi_num_trained_dense_model_v1")

    multinum_lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
    multinum_dense_model = tf.keras.models.load_model(DENSE_MODEL_PATH)

    return multinum_lstm_model, multinum_dense_model

""" Preprocessing Functions """
def preprocessing(sentence):
    # Remove HTML tags
    CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    sentence = re.sub(CLEANR, ' ', sentence)

    # Remove number
    sentence = re.sub(r"\d+", "", sentence)

    # Case folding
    sentence = sentence.lower()

    # Remove punctuation
    for p in string.punctuation:
        sentence = sentence.replace(p, " ")

    # Remove whitespace leading & trailing
    sentence = sentence.strip()

    # Remove multiple whitespace into single whitespace
    sentence = re.sub('\s+',' ', sentence)

    # Tokenization
    tokens = nltk.tokenize.word_tokenize(sentence)

    # Remove stopwords in Bahasa Indonesia
    bahasa_stopwords = set(stopwords.words('indonesian'))
    tokens_without_bahasa_stopwords = [token for token in tokens if not token in bahasa_stopwords]

    # Remove stopwords in English
    english_stopwords = set(stopwords.words('english'))
    tokens_without_bilingual_stopwords = [token for token in tokens_without_bahasa_stopwords if not token in english_stopwords]

    # Lemmatization for English words
    lemmatizer = WordNetLemmatizer()
    english_base_word_tokens = [lemmatizer.lemmatize(token) for token in tokens_without_bilingual_stopwords]

    # Stemming words in Bahasa Indonesia
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    bilingual_base_word_tokens = [stemmer.stem(token) for token in english_base_word_tokens]

    # Combine the list of string into string separated by a whitespace
    return " ".join(bilingual_base_word_tokens)


""" Main Function """
if __name__ == "__main__":
    # Initialize the dummy data:
    dummy_data = init_dummy_data()

    # Initialize the onenum models:
    # TODO: Add onenum models initialization

    # Initialize the multinum models:
    multinum_lstm_model, multinum_dense_model = init_multinum_models()

    # Run the Flask App:
    app.run(debug=True, port=5000)