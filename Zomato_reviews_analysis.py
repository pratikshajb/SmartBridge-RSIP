# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 21:54:33 2020

@author: Pratiksha Bongale
"""

import numpy as np
import pandas as pd
df = pd.read_csv(r"zomato_reviews.tsv", delimiter = "\t", quoting = 3)
import re
import nltk  
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
c = []
for i in range(0, 685): #since there are thousand records
    review = re.sub('[^a-zA-Z]', " ", df["Review"][i])   #to remove quotes and all and replace them with space in the first column only of the dataset
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]  #to perform stemming and removing stop words in one step
    #bringing back to string
    review = " ".join(review)
    c.append(review)

import joblib #to save transformation
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1000)  #top 1000 repeated words
x = cv.fit_transform(c).toarray()
joblib.dump(cv.vocabulary_ ,"model.save")
#so we have a dataset of 1000 columns as independent variables and one output column "liked"
#bag of words model is therefore created.

y = df.iloc[:, -1].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#now either of ML or DL can be applied here onwards.
#proceeding with neural networks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(input_dim=1000, kernel_initializer="random_uniform",activation="sigmoid", units=685)) #as output is either 0 or 1 we use sigmoid activation and this is the input layer
model.add(Dense(units=100, kernel_initializer="random_uniform",activation="sigmoid"))  #one hidden layer
model.add(Dense(units=1, activation="sigmoid")) #output layer
model.compile(optimizer="adam", loss="binary_crossentropy",metrics=["accuracy"]) #change optimizer as adam if you get less accuracy
model.fit(x_train,y_train,epochs=50,batch_size=10)
model.save('zomato.h5')
#to predict
y_pred=model.predict(x_test)
y_pred=(y_pred>=0.5) #since we have used sigmoid
#to check with actual review input - dynamic input
#we have to store the transformation

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

loaded = CountVectorizer(decode_error="replace", vocabulary=joblib.load("model.h5"))
#transformation is now present in loaded
joblib.dump(loaded, "transform")

da="Highly recommendedable app."  #keep changing the review here 
da=da.split("delimiter")
result = model.predict(loaded.transform(da))
prediction = result>=0.5
print(prediction)  #output will be in the top right box
