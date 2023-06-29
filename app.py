from flask import Flask,render_template,request
import pickle
import numpy as np
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('index.html')

# preprocessing function
def preprocess_text(text):
    # remove url
    text = re.sub(r"http\S+", " ", text)
    # remove angka
    text = re.sub(r"\d+", " ", text)
    # case folding
    text = text.lower()
    # tokenization
    text = word_tokenize(text)
    # remove punctuation
    text = [word for word in text if not word in string.punctuation]
    # remove stopwords
    stop_words = set(stopwords.words('indonesian'))
    text = [word for word in text if not word in stop_words]
    # stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = [stemmer.stem(word) for word in text]
    # join text
    text = ' '.join(text)
    # remove whitespace leading & trailing
    text = text.strip()
    # remove multiple whitespace into single whitespace
    text = re.sub('\s+',' ',text)
    return text

# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1)
    model = pickle.load(open('model/sentiment_model.pkl', 'rb'))
    tfidf = pickle.load(open('model/tfidf.pkl', 'rb'))

    text = preprocess_text(to_predict[0])
    result = model.predict(tfidf.transform(text))
    return result[0]


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        komen = request.form['komen']
        
        to_predict_list = komen
        result = ValuePredictor(to_predict_list)
        if result == 'positive':
            prediction = 'Positive'
        elif result == 'negative':
            prediction = 'Negative'
        else:
            prediction = 'Neutral'

    return prediction

if __name__ == "__main__":
    app.run()