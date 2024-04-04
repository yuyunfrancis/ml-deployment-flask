import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create a Flask app
app = Flask(__name__)

# load pickle model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')

def Home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
    float_features = [float(x) for x in request.form.values()] # get independable variables
    
    features = [np.array(float_features)] # convert to array
    
    prediction = model.predict(features) # predict
    
    return render_template("index.html", prediction_text='The flower is {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)


    