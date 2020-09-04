import numpy as np
from flask import Flask, request, render_template
from joblib import load
import joblib
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
app = Flask(__name__)
model= load_model('zomato.h5')
#trans=load('transform')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    da = request.form['Review']
    print(da)
    loaded=CountVectorizer(decode_error='replace',vocabulary=joblib.load('model.save'))
    da=da.split("delimiter")
    result=model.predict(loaded.transform(da))
    prediction=result>=0.5
    print(prediction)
    if prediction[0] == False:
        output="Negative"
    elif prediction[0] == True:
        output="Positive"
    
    
    return render_template('index.html', prediction_text='The sentiment value is : {}'.format(output))

'''@app.route('/predict_api',methods=['POST'])
def predict_api():
    
    #For direct API calls trought request
    
    data = request.get_json(force=True)
    prediction = model.y_predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)'''

if __name__ == "__main__":
    app.run(debug=True)
