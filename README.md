# Bangkit 2022 x Dicoding Discussion Search Engine Optimization Showcase (Company-based Capstone Project)

## Team Members (Team 3):
- Denn Sebastian Emmanuel - M2271F2345
- Faza Nanda Yudistira - M2008F0810
- Muhammad Yusuf Daa Izzalhaqqi - M7008F0833

## About this Project:
This is a Python Flask source code that's used to showcase the optimization of 
Dicoding's Discussion Search Engine by our team, which is the one of the 
requirement needed to pass Kampus Merdeka's Bangkit Academy 2022 program.
  
This showcase will be separated into 3 demo: ML One-Num Demo, ML Multi-Num Demo, and Algorithm Demo. 
  
Both ML One-Num and ML Multi-Num use two models, an `untrained LSTM model` and a `trained Dense model`. The `untrained LSTM model` will convert the `preprocessed discussion data` and `preprocessed search query data` into numerical values. These two values will then be fed into the `trained Dense model` to determine if they are relevant or not (only relevant discussion data will be shown in the page). The only difference between ML One-Num and ML Multi-Num is the number of numerical values outputted by the untrained LSTM model:
- ML One-Num: 1 numerical output from `untrained LSTM model` -> 2 numerical data fed into `trained Dense model`.
- ML Multi-Num: 128 numerical output from `untrained LSTM model` -> 256 numerical data fed into `trained Dense model`.

The Algorithm Demo will showcase the use of TF-IDF algorithm as a search algorithm. This algorithm will sort the discussion data based on its relevancy to the search query.

## Note:
This source code won't go into detail about the development process of the algorithm and Machine Learning model used in this demonstration, or the data processing steps. This source code is purely made for 
demonstration of the final result of this Capstone Project.

## Minimum Requirement to Run These Code (This may change during development):
- Python v3.10.x
- TensorFlow v2.8.x
- Flask v2.0.x
- Sastrawi v1.0.1
- nltk v3.7 or above
- Pandas v1.4.2 or above
- Numpy v1.21.5 or above
- Scikit-Learn v1.0.2 or above

## How to Run these Code:
Run the `app.py` file using Python Environment containing the above requirement. 
After that, go to the website link provided by `app.py` in the terminal 
(defaults to `http://127.0.0.1:5000/`).

## Result (One-Num Model):
**Note: The One-Num Models' architectures are too small to give any meaningful prediction.**
  
![one_num_gif](https://drive.google.com/uc?export=view&id=1b8ANLyrS8UrInNmX4Ej7jg0JX9ojhPqF)

## Result (Multi-Num Model):
![multi_num_gif](https://drive.google.com/uc?export=view&id=1jJqW9Oaj5jWzRp79hVJnPrZd6aA5akQw)

## Result (TF-IDF Algorithm):
![tf-idf_gif](https://drive.google.com/uc?export=view&id=1lY4jkrmoYZWyu_JetbhaX6_Wn_SmHgFE)
