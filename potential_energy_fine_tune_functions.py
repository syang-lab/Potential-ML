#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 01:40:11 2020

@author: shanyang
"""

import numpy as np 
from sklearn.model_selection import train_test_split;
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import initializers 
from keras import optimizers
import matplotlib.pyplot as plt
from keras.models import load_model


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
    

def build_model(n_hidden, n_neuron, total_atoms=96):
    model = Sequential();
    init = initializers.RandomNormal(mean=0, stddev=1, seed=None);
    model.add(Dense(n_neuron, input_shape=(total_atoms*3,), activation='tanh',kernel_initializer= init));
    
    for layer in range(n_hidden):
        model.add(Dense(n_neuron, activation='tanh',kernel_initializer= init));
    
    model.add(Dense(1,activation=None)); 
    optimizer_adam = optimizers.Adam(lr=0.005,decay = 0.01)
    model.compile(loss='mean_absolute_error', optimizer= optimizer_adam)
    return model


def model_parameters(x_train,y_train,n_hidden,n_neuron,total_atoms,input_epochs):
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience = 30,restore_best_weights = True)
    model = build_model(n_hidden, n_neuron, total_atoms)
    hist=model.fit(x_train,y_train,epochs=input_epochs,batch_size =100, validation_split=0.2, callbacks=[early_stopping_cb])    
    plt.figure()
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()   
    return model 


def model_predict(x_test,y_test,model):
    mse_test = model.evaluate(x_test, y_test)
    print(mse_test)
    y_pred = model.predict(x_test)
    plt.figure()
    plt.plot(y_pred)
    plt.plot(y_test)
    plt.title('prediction and test')
    plt.ylabel('Y')
    plt.xlabel('configurations')
    plt.legend(['y_pred', 'y_test'], loc='upper left')
    plt.show()
    
    
def process_original_data_and_save(XDATCAR,Energy):
    data_loader = load_data();
    x_original=data_loader.read_xdatcar(XDATCAR);
    y_original=data_loader.read_energy(Energy);
    x = x_original ;
    y = (y_original-y_original[0])*1000; # to meV
    np.savetxt('x',x)
    np.savetxt('y',y)
    return data_loader.get_total_atoms();
    
   
def fine_tune_model(total_atoms,input_epochs,x_train,y_train,x_val,y_val):
    #total_atoms = process_original_data_and_save('/Users/shanyang/Desktop/Potential-ML/XDATCAR', '/Users/shanyang/Desktop/Potential-ML/Energy')
    #total_atoms = 96;
    #input_epochs = 3000;
    #x=np.loadtxt('x')
    #y=np.loadtxt('y')
    #x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, test_size=0.2, shuffle = True ,random_state=42)
    #x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.1, shuffle = True ,random_state=42)
    for n_hidden in np.array([0,1,2,3,4]):
        for n_neuron in np.array([16,32,64,96]):
            model = model_parameters(x_train,y_train,n_hidden,n_neuron,total_atoms,input_epochs)
            model.evaluate(x_val,y_val)
            model.save(str(n_hidden)+str(n_neuron)+"Potential_model.h5")


def load_model_on_validate(x_val,y_val,model_name):
    #total_atoms = 96;
    #input_epochs = 3000;
    #x=np.loadtxt('x')
    #y=np.loadtxt('y')
    #x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, test_size=0.1, shuffle = True ,random_state=42)
    #x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.1, shuffle = True ,random_state=42)
    # load model
    model = load_model(model_name)
    mse_val = model.evaluate(x_val, y_val)
    print(mse_val)
    y_pred = model.predict(x_val)
    plt.figure()
    plt.plot(y_pred)
    plt.plot(y_val)
    plt.title('prediction on validation set')
    plt.ylabel('Y')
    plt.xlabel('configurations')
    plt.legend(['y_pred', 'y_val'], loc='upper left')
    plt.show()    
    return model 

def main():
    #preprocess of XDATCAR and Energy data, and save the x (features) and y (lables)
    XDATCAR = 'path_of_XDATCAR'
    Energy = 'path_of_Energy'
    process_original_data_and_save(XDATCAR,Energy)
    # reload the preprocesssed data 
    x=np.loadtxt('x')
    y=np.loadtxt('y')
    # train validation and test set
    x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, test_size=0.1, shuffle = True ,random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.1, shuffle = True ,random_state=42)
    # this is fixed, which is based on your proprocess of your data 
    total_atoms = 96; # this is fixed 
    input_epochs = 3000; 
    # early stopping is implemented in the model
    # training different models
    fine_tune_model(total_atoms,input_epochs,x_train,y_train,x_val,y_val)

    # check every model and find the best one
    model_name = '496Potential_model.h5';
    model = load_model_on_validate(x_val,y_val,model_name)
    
    # select the best model and applied to the test set 
    model_predict(x_test,y_test,model)

if __name__=='__main__':
    main()
    
    
    
    
    
    
    
    
    