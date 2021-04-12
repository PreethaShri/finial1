# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:52:58 2020

@author: sai
"""
import os
from flask import Flask
from flask import request,render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
filename = "Crop_Recom_Model.pkl"
with open(filename, 'rb') as file:  
    loaded_model= pickle.load(file)

app = Flask(__name__)
@app.route('/')
def inputs():
    return render_template("index.html")
@app.route('/output', methods = ["POST","GET"])

def outputs():
    a = request.form["temperature"]
    b = request.form["humidity"]
    c = request.form["ph"]
    d = request.form["rainfall"]
    index2 = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute',
          'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange',
          'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon']
    p = loaded_model.predict(np.array([[a,b,c,d]]))
    ylist = int(p.flatten('F'))
    
    return render_template("index.html",y = "the suitable crop is : "+index2[ylist])
if __name__ == '__main__':
    app.run()