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

# import data
csv_path = "dataset.csv"
dataset=np.loadtxt(csv_path,delimiter=";")
np.random.shuffle(dataset)

train,test=train_test_split(dataset,test_size=0.15)

X_train=train[:,0:32]
Y_train=train[:,32]

X_test=test[:,0:32]
Y_test=test[:,32]



# initialiser les parametres
nblayers=4
nbneur=50
nbepochs=300

#fonctions d'activation : relu ou linear
act_func='relu'

#fonctions d'optimisation : ['Adam','Adamax','Ftrl','Adagrad']
opt_func='Adam'

funct_error='mse'
metrics = ['accuracy','mse','mae']

# creation du  modele

model = keras.Sequential()
model.add( layers.Dense(50, activation=act_func, input_shape=('32',)) )
model.add(layers.Dense(50, activation=act_func) )
model.add(layers.Dense(50, activation=act_func) )          
model.add(layers.Dense(50, activation=act_func) )
model.add( layers.Dense(1, activation='relu') )
model.compile ( optimizer= tf.keras.optimizers.Adamax(learning_rate=0.002), loss= funct_error, metrics=metrics )

print('------------------------------ Debut entrainement --------------------------------')       
history = model.fit( X_train, Y_train, epochs=nbepochs, batch_size=519, validation_split= 0.15, verbose=1)

evaluation = model.evaluate(X_test, Y_test, batch_size=X_test.shape[0], verbose=2 )

#Organisation du Test
sample = X_test[0:98]
target = Y_test[0:98]

#Prediction
result = model.predict(sample, batch_size=1)


#Organisé les données pour l'affichage
target_final=[]
result_final=[]
for i in result:
    for j in i:
        result_final.append(j)

target_ok=[]

result=result
target_final2=np.reshape(target_final, (-1, 1))


for i in target:
    target_ok.append(i)
for i in result:
    for j in i:
        target_final.append(j)
print ("\n")

print('---------------- Entrainement Fini ------------------')
print("Loss Train :",history.history['val_loss'][nbepochs-1])
print("Acc :",history.history['acc'][nbepochs-1])
print("MSE :",history.history["mean_squared_error"][nbepochs-1])
print("Loss test :", evaluation[2])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('MSE - Mean Squared Error')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

print('\n')
print('---------------------------------------- Resultats de test ---------------------------------')

print('-------Target-------')
print('\n')



target_ok=np.array(target_ok)
print(target_ok)
result_final=np.array(target_final)
roundedresulaffiche=result_final.round()

target_ok=target_ok.reshape(-1,1)
result_final=result_final.reshape(-1,1)
result_final=result_final.round()
print('\n')
print('-----Prediction-----')
print('\n')
print(roundedresulaffiche)
plt.title('Progression')
plt.scatter(target,result_final,color='black')
plt.ylabel('Ouput')
plt.xlabel('Target')
plt.show()


                             
# Sauvegarde du model en HDF5
model.save_weights('../Application/apprn/modelfolder/model_weights.h5') # sauvegarder juste les poids
model.save('../Application/apprn/modelfolder/model')
model.save('../Application/apprn/modelfolder/model.h5')
print("model sauvegardé ")

 # sauvegarder juste la config
model_json = model.to_json() 
with open("../Application/apprn/modelfolder/model_config.json", "w") as f:
    f.write(model_json)



#Convertir le model en model pret pour TensroflowJs pour l'importer sur une page web
#tfjs.converters.save_keras_model(model,'./keras_model' )



