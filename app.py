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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


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
        
        # Return if query is empty
        if not query:
            return render_template('ml_onenum_demo.html', discussions=dummy_data)

        # Preprocess the query
        preprocessed_query = preprocessing(query)

        # Return if preprocessed query is empty
        if not preprocessed_query:
            return render_template('ml_onenum_demo.html', discussions=dummy_data)

        # Convert the query into onenum data
        onenum_query = onenum_lstm_model.predict([preprocessed_query])[0]

        # Create a deep copy of the dummy_data
        dummy_data_copy = copy.deepcopy(dummy_data)

        # Predict the relevancy of each discussion data
        for data in dummy_data_copy:
            input_data = np.append(data['onenum_value'], onenum_query)
            input_data = np.expand_dims(input_data, axis=0)
            prediction = onenum_dense_model.predict(input_data)[0]
            # data['relevant'] = prediction
            data['relevant'] = true_binary_prediction_converter(prediction)

        print(dummy_data_copy)

        return render_template('ml_onenum_demo.html', 
                                query=query,
                                discussions=dummy_data_copy,
                                preprocessed_query=preprocessed_query)

@app.route('/ml-multinum-demo', methods=['GET', 'POST'])
def ml_multinum_demo():
    if request.method == 'GET':
        print(dummy_data)
        return render_template('ml_multinum_demo.html', discussions=dummy_data)

    if request.method == 'POST':
        query = request.form['query']

        # Return if query is empty
        if not query:
            return render_template('ml_multinum_demo.html', discussions=dummy_data)

        # Preprocess the query
        preprocessed_query = preprocessing(query)

        # Return if preprocessed query is empty
        if not preprocessed_query:
            return render_template('ml_multinum_demo.html', discussions=dummy_data)

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

        # Return if query is empty
        if not query:
            return render_template('algorithm_demo.html', discussions=dummy_data)

        # Preprocess the query
        preprocessed_query = preprocessing(query)

        # Return if preprocessed query is empty
        if not preprocessed_query:
            return render_template('algorithm_demo.html', discussions=dummy_data)

        # Do the Cosine Similarity Search (TF-IDF)
        result = cosine_sim_search(preprocessed_query)

        return render_template('algorithm_demo.html', 
                                query=query,
                                discussions=result,
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

def cosine_sim_search(preprocessed_query):
    # Regular expression to get the index of the document in dataset that matches the title
    idx = None
    for curr_idx, data in enumerate(dummy_data):
        if re.search(preprocessed_query, data['combined_processed']):
            idx = curr_idx
            break

    # Check if idx = None, return empty list (no result)
    if idx == None:
        return []

    #Get the pairwise similarity scores of all document in the dataset with the title
    sim_scores = list(enumerate(cosine_sim[idx]))

    #Sort the document based on the similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    #Get the scores of the 10 most similar documents
    above_threshold = []
    for idx, score in sim_scores:
        if score >= 0.1:
            above_threshold.append((idx, score))

    #Get the indices of the documents
    dummy_data_indices = [i[0] for i in above_threshold]

    result = [dummy_data[idx] for idx in dummy_data_indices]

    return result

""" Initialization Functions """
def init_dummy_data_and_cosine_similarity():
    """Read the dummy data and convert it into a list of objects"""

    # Get the Excel Path:
    # NOTE: __file__ is the absolute path of this app.py
    # NOTE: os.pardir is the parent directory (..)
    DUMMY_PATH = os.path.join(__file__, os.pardir, "discussion_dummy_data_v2.xlsx")
    print("Excel Path:", DUMMY_PATH, end="\n\n")

    # Open the Excel Dummy Data as Pandas DataFrame:
    dummy_data_df = pd.read_excel(DUMMY_PATH)

    # Convert Pandas DataFrame into Python List of Dictionaries [{column -> value}, ...]:
    """
    [
        {
            'title': title row 1, 
            'content': content row 1,
            'keywords': keywords row 1,
            'combined': combined row 1,
            'combined_processed': combined_processed row 1,
            'onenum_value': onenum_value row 1,
            'multinum_value': multinum_value row 1,

        },
        {
            'title': title row 2, 
            'content': content row 2,
            'keywords': keywords row 2,
            'combined': combined row 2,
            'combined_processed': combined_processed row 2,
            'onenum_value': onenum_value row 2,
            'multinum_value': multinum_value row 2,
        },
        ...
    ]
    """
    dummy_data = dummy_data_df.to_dict(orient='records')

    # Convert all keywords into list:
    regex = r"\b[a-z]+\b"
    for data in dummy_data:
        data['keywords'] = re.findall(regex, data['keywords'])
        data['multinum_value'] = np.array(convert_stringlist_to_list(data['multinum_value']))
        data['onenum_value'] = np.array(data['onenum_value'])
        data['relevant'] = 1
    
    # Initialize TfidfVectorizer:
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(dummy_data_df['combined_processed'])    

    # Initialize Cosine Similarities:
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    return dummy_data, cosine_sim

def init_onenum_models():
    """Load the One-Num Models (Untrained LSTM and Trained Dense Models)"""
    LSTM_MODEL_PATH = os.path.join(__file__, os.pardir, "one_num_untrained_LSTM_model_v1")
    DENSE_MODEL_PATH = os.path.join(__file__, os.pardir, "one_num_trained_dense_model_v1")

    onenum_lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
    onenum_dense_model = tf.keras.models.load_model(DENSE_MODEL_PATH)

    return onenum_lstm_model, onenum_dense_model

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
    dummy_data, cosine_sim = init_dummy_data_and_cosine_similarity()

    # Initialize the onenum models:
    onenum_lstm_model, onenum_dense_model = init_onenum_models()

    # Initialize the multinum models:
    multinum_lstm_model, multinum_dense_model = init_multinum_models()

    # Run the Flask App:
    app.run(debug=True, port=5000)