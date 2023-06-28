from flask import Flask,render_template,request
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('index.html')

# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1)
    model = pickle.load(open('model/sentiment_model.pkl', 'rb'))
    tfidf = pickle.load(open('model/tfidf.pkl', 'rb'))
    result = model.predict(tfidf.transform(to_predict))
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