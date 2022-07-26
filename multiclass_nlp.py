# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 09:13:58 2022

@author: Shah
"""

import os
import re
import json
import pickle
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from multiclass_nlp_module import ModelDevelopment,ModelEvaluation
from sklearn.metrics import confusion_matrix, classification_report
#%%Constant
TOKENIZER_SAVE_PATH=os.path.join(os.getcwd(),'model','tokenizer.json')
OHE_SAVE_PATH=os.path.join(os.getcwd(),'model','ohe.pkl')
MODEL_SAVE_PATH=os.path.join(os.getcwd(),'model','model.h5')
#%%STEP 1 Data Loading
df=pd.read_csv('https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv')
#%%STEP 2 Data Inspection
df.info()
df.describe().T
df.head(10)
#category = target
df.isna().sum()
#no NaNs

print(df['text'][4])
print(df['text'][10])
print(df['text'][20])
print(df['text'][30])

sns.countplot(x=df.category)
plt.title('Category')
plt.show

#imbalance target, model can improve more if target more balance
#%%STEP 3 Data Cleaning

#removing unecessary symbol and spacing
text=df['text']
category=df['category']

text_backup=text.copy()
category_backup=category.copy()

for index, texts in enumerate(text):
    text[index]=re.sub('<.*?>','',texts)
    text[index]=re.sub('[^a-zA-Z]',' ',texts).lower().split()
    
print(text[12])
print(df['text'][12])
#%%STEP 4 Features Selection

#no feature selection because we only have 1 feature
#%% STEP 5 Data Preprocessing

vocab_size=10000
oov_token='<OOV>'

tokenizer=Tokenizer(num_words=vocab_size,oov_token=oov_token)

tokenizer.fit_on_texts(text) #to learn
word_index=tokenizer.word_index

print(dict(list(word_index.items())[0:10]))

text_int=tokenizer.texts_to_sequences(text) #to convert to number
text_int[100] #to check all convert to number

max_len=np.median([len(text_int[i])for i in range(len(text_int))])

padded_text=pad_sequences(text_int,
                          maxlen=int(max_len),
                          padding='post',
                          truncating='post')

#%% One Hot Encoder
# Y target
ohe=OneHotEncoder(sparse=False)
category=ohe.fit_transform(np.expand_dims(category,axis=-1))


X_train,X_test,y_train,y_test=train_test_split(padded_text,
                                                category,
                                                test_size=0.3,
                                                random_state=(123))

#%%Model Development
input_shape=np.shape(X_train)[1:]
nb_class=len(np.unique(category,axis=0))
out_dim = 128

md=ModelDevelopment()
model=md.simple_dl_model(input_shape,nb_class,vocab_size,
                    out_dim=128,nb_node=128,dropout_rate=0.3)

plot_model(model,show_shapes=(True))

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['acc'])
#%% Callbacks

LOGS_PATH=os.path.join(os.getcwd(),'logs',datetime.datetime.now().
                          strftime('%Y%m%d-%H%M%S'))

tensorboard_callback=TensorBoard(log_dir=LOGS_PATH,histogram_freq=1)

#%% Model Training

hist=model.fit(X_train,y_train,
               epochs=5,
               callbacks=[tensorboard_callback],
               validation_data=(X_test,y_test))

#%%Model Evaluation
print(hist.history.keys())
key=list(hist.history.keys())

#plot_hist_graph
me=ModelEvaluation()
plot_hist=me.plot_hist_graph(hist,key,a=0,b=2,c=1,d=3)
#%%model analysis

y_pred=np.argmax(model.predict(X_test),axis=1)
y_actual=np.argmax(y_test,axis=1)
labels=['tech','business','sport','entertainment','politics']

cr=classification_report(y_actual,y_pred,target_names=labels)
print(cr)

cm=confusion_matrix(y_actual, y_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
disp.plot(cmap='Blues')
plt.rcParams['figure.figsize']=[9, 9]
plt.show()

#%%model saving
#TOKENIZER
token_json=tokenizer.to_json()
with open(TOKENIZER_SAVE_PATH,'w')as file:
    json.dump(token_json,file)

# OHE
with open(OHE_SAVE_PATH,'wb') as file:
    pickle.dump(ohe,file)
    
#MODEL
model.save(MODEL_SAVE_PATH)
