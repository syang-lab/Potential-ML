#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 14:51:46 2020

@author: shanyang
"""
# import library 
import numpy as np 
from sklearn.model_selection import train_test_split;
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import initializers 
from keras import optimizers
import matplotlib.pyplot as plt

# fit the total energy with AIMD go get potential energy surface at finite temperatures

# preprocess data
class load_data:
    def read_xdatcar(self,path):
        """ read the XDARCAR file from AIMD (VASP) """
        try:
            with open(path,'r') as xdatcar:
                self._system=xdatcar.readline()
                self._scale_supercell=float(xdatcar.readline().rstrip('\n').rstrip());
                self._a_supercell_vector=np.array([float(i)*self._scale_supercell for i in xdatcar.readline().rstrip('\n').split()])
                self._b_supercell_vector=np.array([float(i)*self._scale_supercell for i in xdatcar.readline().rstrip('\n').split()])
                self._c_supercell_vector=np.array([float(i)*self._scale_supercell for i in xdatcar.readline().rstrip('\n').split()])
                self._latticevector_matrix_supercell=np.round(np.stack((self._a_supercell_vector,self._b_supercell_vector,self._c_supercell_vector)),6)
                self._element_names = [name for name in xdatcar.readline().rstrip('\n').split()]
                self._element_numbers = np.array([int(number) for number in xdatcar.readline().rstrip('\n').split()])
                self._total_number = np.sum(self._element_numbers)
                self._xdatcar=[]
                self._count = 0
                while True:
                    line=xdatcar.readline().rstrip('\n').split();
                    if not line:
                        break
                    if (self._isfloat(*[items for items in line])):
                        self._xdatcar.append(line)
                        self._count +=1
                #self._xdatcar_fract = np.asarray(self._xdatcar,dtype = float)
                self._steps = int(self._count/self._total_number)
        except FileNotFoundError as e:
            print('XDARCAR file does not exist:{}'.format(e))
            raise e
        """ reshape the data from XDATCAR to 3D matrix steps * atoms * xyz(direction)""" 
        self._xdatcar_fract = np.zeros((self._steps,self._total_number*3));
        for t in range(self._steps):
            self._xdatcar_fract[t,:] = np.asarray(self._xdatcar,dtype = float)[t*self._total_number:(t+1)*self._total_number,:].flatten();
            
        return self._xdatcar_fract;
    
    def _isfloat(self,*value):
        for it in value:
            try:
                float(it)
            except ValueError:
                return False
        return True;
    
    def read_energy(self,path):
        try: 
            self._energy = np.loadtxt(path);
        except FileNotFoundError as e:
            print('Energy file does not exist:{}'.format(e))
            raise e
        return self._energy;
    
    def get_total_steps(self):
        return self._steps
    
    def get_total_atoms(self):
        return self._total_number;
    
#don't need standerlization and normalization, because the inner coordinate are extracted from the XDATCAR
#data_loader = load_data();
#x_original=data_loader.read_xdatcar('/Users/shanyang/Desktop/Potential-ML/XDATCAR');
#y_original=data_loader.read_energy('/Users/shanyang/Desktop/Potential-ML/Energy');
    
# may save the data as csv file in this part 
#  create check point 
#x = x_original ;
# note: mayneed to try to scale the y according to the activation function in the last layer 
#y = (y_original-y_original[0])*1000; # meV

#np.savetxt('x',x)
#np.savetxt('y',y)
        
x=np.loadtxt('x')
y=np.loadtxt('y')

# train_test split 
#x=data_loader.read_xdatcar('/Users/shanyang/Desktop/Potential-ML/XDATCAR-test');
#y=data_loader.read_energy('/Users/shanyang/Desktop/Potential-ML/Energy-test');
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle = True ,random_state=42);

# Create the deep neural network model
model = Sequential();
#total_atoms = data_loader.get_total_atoms(); 
total_atoms = 96;
init = initializers.RandomNormal(mean=0, stddev=1, seed=None)
model.add(Dense(32, input_shape=(total_atoms*3,), activation='tanh',kernel_initializer= init));
model.add(Dense(32, activation='tanh',kernel_initializer= init));
model.add(Dense(32,activation='tanh',kernel_initializer= init));
model.add(Dense(1,activation=None));
optimizer_adam = optimizers.Adam(lr=0.005,decay = 0.01)
model.compile(loss='mean_absolute_error', optimizer=optimizer_adam)
model.summary()

# Train the model
# call back
early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience = 30,restore_best_weights = True)
hist=model.fit(x_train,y_train,epochs =1000, batch_size =30, validation_split=0.2,callbacks=[early_stopping_cb])
mse_test = model.evaluate(x_test, y_test)

# plot loss function 
plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Test on the test set
y_pred = model.predict(x_test)

plt.figure()
plt.plot(y_pred)
plt.plot(y_test)
plt.title('prediction and test')
plt.ylabel('Y')
plt.xlabel('configurations')
plt.legend(['y_pred', 'y_test'], loc='upper left')
plt.show()

# save the model 
model.save("Potential_model.h5")





