import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app = Flask(__name__)
score_deploy = pickle.load(open('model.pkl','rb'))
test_clients = [
{'id' : '100013'},
{'id' : '100015'}
]
@app.route('/')
def home():
    return 'Scoring model application'

@app.route('/predict',methods = ['POST'])
def predict():
    #int_features = [float(x) for x in request.form.values()]
    final_features = request.get_json()
    prediction = score_deploy.predict(final_features)
    output = score_deploy.predict_proba(final_features)
    #output = round(prediction[0],2)
    
    return render_template('index.html',prediction_text ='The default risk score is  $ {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)