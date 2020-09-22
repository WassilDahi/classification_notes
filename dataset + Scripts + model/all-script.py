import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import model_from_json
import statistics

# Importation du dataset
csv_path = "dataset.csv"
dataset=np.loadtxt(csv_path,delimiter=";")

#Mélanger le data pour une distribution random
np.random.shuffle(dataset)

#Découper le dataset en 85% Train et 15% Test
train,test=train_test_split(dataset,test_size=0.15)

#Découper 
X_train=train[:,0:32]
Y_train=train[:,32]

X_test=test[:,0:32]
Y_test=test[:,32]


# initialiser les  parameteres
list_layers=[1,2]
list_neur=[10,20]
list_epochs=[100]
list_act_func=['relu','linear']
list_opt_func=['SGD','Adamax']
bsize = 519
loss='mse'
list_losses = ['mse','mae']
metrics = ['accuracy']


               
res=[]

#Les boucles de variation des parametres
for act_func in list_act_func:
    for opt_fun in list_opt_func:
        for l in list_layers:
            for n in list_neur :
                for e in list_epochs:
                    hist_moy=[]
                    eval_moy=[]
                    mse_moy=[]
                    val_mse_moy=[]
                    accuracy_list=[]
                    acc_train_list=[]
                    print("Loss :",loss)
                    print("Fonction activation : ",act_func)
                    print("Fonction optimisation : ",opt_fun)
                    print("Nombre Layers : ",l)
                    print("Nombre neurones :",n)
                    print("Nb Epochs :",e)

                    #Faire 10 iterations pour chaque configuration
                    for j in range (0,10):
                        #Creation du model
                        model = keras.Sequential()
                        model.add( layers.Dense(n, activation=act_func, input_shape=('32',)) )
                        for i in range(1,l-1):
                            model.add(layers.Dense(n, activation=act_func) )
                        model.add( layers.Dense(1, activation='linear') )
                        if(opt_fun=='Adam'):
                            model.compile ( optimizer= tf.keras.optimizers.Adam(learning_rate=0.002), loss= loss, metrics=metrics )
                        elif (opt_fun=='Ftrl'):
                            model.compile ( optimizer= tf.keras.optimizers.Ftrl(learning_rate=0.002), loss= loss, metrics=metrics )
                        elif(opt_fun=='SGD'):
                            model.compile ( optimizer= tf.keras.optimizers.SGD(learning_rate=0.002), loss= loss, metrics=metrics )
                        elif(opt_fun=='Adagrad'):
                            model.compile ( optimizer= tf.keras.optimizers.Adagrad(learning_rate=0.002), loss= loss, metrics=metrics )
                        elif(opt_fun=='Adamax'):
                            model.compile ( optimizer= tf.keras.optimizers.Adamax(learning_rate=0.002), loss= loss, metrics=metrics )
                        
                        
                        #Entrainement du model
                        history = model.fit( X_train, Y_train, epochs=e, batch_size=519, validation_split= 0.15, verbose=2)
                        evaluation = model.evaluate(X_test, Y_test, batch_size=X_test.shape[0], verbose=2 )

                        #Sauvegarde des différentes stats
                        hist_moy.append(history.history['loss'][e-1])
                        acc_train_list.append(history.history['acc'][e-1])
                        '''val_mse_moy.append(history.history['val_mean_squared_error'][e-1])
                        mse_moy.append(history.history["mean_squared_error"][e-1])
                        eval_moy.append(evaluation[2])'''
                        
                        sample = X_test
                        target = Y_test
                        result = model.predict(sample, batch_size=20)
                        rounded_result=np.round(result)
                        total=np.shape(sample)[0]
                            
                        i=0
                        number_exact=0
                        for k in (Y_test):
                            
                                if (rounded_result[i]==k):
                                    number_exact=number_exact+1
                                i=i+1
                        accuracy_curret=number_exact/total
                        accuracy_list.append(accuracy_curret)
                        
                    
                    #Calcule des moyennes de statistiques et sauvegarder le fichier
                    res.append([act_func,opt_fun,l,n,e,statistics.mean(hist_moy),statistics.mean(list(map(float, accuracy_list))),statistics.mean(list(map(float, acc_train_list)))])
                    ok=pd.DataFrame(res, columns =['act_fun', 'opt_fun', 'l','n','e','loss train moy','losstest','val-mse','mse_moy','acc','acc_train'])

                    ok.to_csv('l1sdg.csv',encoding='utf-8',index=False)