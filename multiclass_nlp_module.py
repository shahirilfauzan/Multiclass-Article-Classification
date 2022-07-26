# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 12:25:32 2022

@author: Shah
"""

from tensorflow.keras.layers import LSTM,Dense,Dropout,Embedding,Bidirectional
from tensorflow.keras import Input,Sequential 
import matplotlib.pyplot as plt

class ModelDevelopment:
    def simple_dl_model(self,input_shape,nb_class,vocab_size,
                        out_dim=128,nb_node=128,dropout_rate=0.3):
        '''
        

        Parameters
        ----------
        input_shape : TYPE
            DESCRIPTION.
        nb_class : TYPE
            DESCRIPTION.
        vocab_size : TYPE
            DESCRIPTION.
        out_dim : TYPE, optional
            DESCRIPTION. The default is 128.
        nb_node : TYPE, optional
            DESCRIPTION. The default is 128.
        dropout_rate : TYPE, optional
            DESCRIPTION. The default is 0.3.

        Returns
        -------
        model : TYPE
            DESCRIPTION.

        '''
        model=Sequential()
        model.add(Input(shape=(input_shape)))
        model.add(Embedding(vocab_size,out_dim))
        model.add(Bidirectional(LSTM(nb_node,return_sequences=True)))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(nb_node)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(nb_class,activation='softmax'))
        model.summary()
        
        return model


class ModelEvaluation:
    def plot_hist_graph(self,hist,key,a=0,b=2,c=1,d=3):
        plt.figure()
        plt.plot(hist.history[key[a]])
        plt.plot(hist.history[key[b]])
        plt.legend(['training_'+ str(key[a]), key[b]])
        plt.show()
        
        plt.figure()
        plt.plot(hist.history[key[c]])
        plt.plot(hist.history[key[d]])
        plt.legend(['training_'+ str(key[c]), key[d]])
        plt.show()
        