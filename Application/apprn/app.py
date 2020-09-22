import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import statistics
import tensorflowjs as tfjs
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from numpy import loadtxt
from keras.models import load_model

# load Flask 
import flask
from flask import request
from flask import render_template
from string import Template

#Creation d'une application flask
app = flask.Flask(__name__)



#load model
json_file=open('modelfolder/model_config.json','r')
loadedmodel_json=json_file.read()
json_file.close()
loadedmodel=tf.keras.models.model_from_json(loadedmodel_json)
loadedmodel.load_weights("modelfolder/model.h5")


#Adresse de la page d'acceuil : 127.0.0.1:8080/
@app.route("/")
def index():
        return render_template('home.html')
# Prediction fonction et son adresse
@app.route("/predict", methods=["GET","POST"])
def predict():
    if request.method=='GET':
        print('get')
    #Récuperation des données de la page web    
    if request.method=='POST':
            data=[]
            data.append(request.form['sex'])
            data.append(request.form['school'])
            data.append(request.form['age'])
            data.append(request.form['adress'])
            data.append(request.form['famsize'])
            data.append(request.form['pstatus'])
            data.append(request.form['medu'])
            data.append(request.form['fedu'])
            data.append(request.form['mjob'])
            data.append(request.form['fjob'])
            data.append(request.form['reason'])
            data.append(request.form['guardian'])
            data.append(request.form['traveltime'])
            data.append(request.form['studytime'])
            data.append(request.form['failure'])
            data.append(request.form['schoolsup'])
            data.append(request.form['famsup'])
            data.append(request.form['paid'])
            data.append(request.form['activities'])
            data.append(request.form['nursery'])
            data.append(request.form['higher'])
            data.append(request.form['internet'])
            data.append(request.form['romantic'])
            data.append(request.form['famrel'])
            data.append(request.form['freetime'])
            data.append(request.form['goout'])
            data.append(request.form['dalc'])
            data.append(request.form['walc'])
            data.append(request.form['health'])
            data.append(request.form['absences'])
            data.append(request.form['g1'])
            data.append(request.form['g2'])
    #Organisation des données
    result2=np.asarray(data, dtype=np.float64, order='C')
    result3=np.reshape(result2, (-1, 32))
    print(result3) 
    
    #prediction
    result = loadedmodel.predict(result3, batch_size=1)

    #affichage du resultat sur la console
    print(result)
    result=result.round()
    x=result.item(0)
    print(x)

    #retourner le resultat sur la page web
    return render_template('home.html',name=int(x))
# start the flask app, allow remote connections
app.run('0.0.0.0', 8080, debug=True)