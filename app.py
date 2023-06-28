from flask import Flask,render_template

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    return render_template('index.html', prediction_text='The predicted value is 0.0')

if __name__ == "__main__":
    app.run()