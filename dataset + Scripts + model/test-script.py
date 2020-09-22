import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import model_from_json
import statistics
import tensorflowjs as tfjs
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from mlxtend.plotting import plot_linear_regression
from numpy import loadtxt
from keras.models import load_model

# import data
csv_path = "dataset.csv"
dataset=np.loadtxt(csv_path,delimiter=";")

#réorganiser le dataset de façon random
np.random.shuffle(dataset)

#Découpage du dataset en test,train
train,test=train_test_split(dataset,test_size=0.15)

X_train=train[:,0:32]
Y_train=train[:,32]

X_test=test[:,0:32]
Y_test=test[:,32]

sample = X_test[0:98]
target = Y_test[0:98]

#Importation du model
json_file=open('../Application/apprn/modelfolder/model_config.json','r')
loadedmodel_json=json_file.read()
json_file.close()
loadedmodel=tf.keras.models.model_from_json(loadedmodel_json)
loadedmodel.load_weights("../Application/apprn/modelfolder/model.h5")


#Prediction et affichage
result = loadedmodel.predict(sample, batch_size=1)
print('-------------------Target-----------------')
print(target)
print('-------------------Output-----------------')
result_rounded=[]
result_normal=[]
for i in result:
    for j in i:
        result_rounded.append(j)
for i in result:
    for j in i:
        result_normal.append(j)
result_rounded=np.array(result_rounded).round()
result_normal=np.array(result_normal)

print('------Output normal-------')
print(result_normal)
print('------Output rounded-------')
print(result_rounded)

