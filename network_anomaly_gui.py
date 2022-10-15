# IMPORTING LIBRARIES 

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers,models
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Activation
from sklearn.preprocessing import LabelEncoder
import warnings 
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score

st.title('MLSA Event')
st.subheader('Upload the Anomaly detection dataset (.csv)')
# creating a side bar 
st.sidebar.info("Created By : Deepthi Sudharsan")
# Adding an image to the side bar
st.sidebar.header("Anomaly Detection") 
st.sidebar.image("data:image/png;base64,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", width=None)
st.sidebar.subheader("Contact Information : ")
col1, mid, col2 = st.sidebar.columns([1,1,20])
with col1:
	st.sidebar.subheader("LinkedIn : ")
with col2:
	st.sidebar.markdown("[![Linkedin](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTLsu_X_ZxDhuVzjTHvk4eZOmUDklreUExhlw&usqp=CAU)](https://www.linkedin.com/in/deepthi-sudharsan/)")

col3, mid, col4 = st.sidebar.columns([1,1,20])
with col3:
	st.sidebar.subheader("Github : ")
with col4:
	st.sidebar.markdown("[![Github](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQJGtP-Pq0P67Ptyv3tB7Zn2ZYPIT-lPGI7AA&usqp=CAU)](https://github.com/DeepthiSudharsan)")

# For user to upload the data
file = st.file_uploader('Upload CSV Dataset')
flag = False

if file is not None:

	data = pd.read_csv(file)
	# flag is set to true as data has been successfully read
	flag = "True"
	st.header('Anomaly Data successfuly loaded')
	st.dataframe(data.head())
	# Since normal datas are labelled as "normal" but attacks are given by the attack type which we aren't concerned 
	# about as of now as we need to know if it is an attack or not so all the labels apart from normal we are changing it to 
	# "attack"
	data['attack'].loc[data['attack']!='normal']='attack'
	# label encoding non-numerical attributes
	le=LabelEncoder()
	data['protocoltype'] = le.fit_transform(data['protocoltype'])
	data['service'] = le.fit_transform(data['service'])
	data['flag'] = le.fit_transform(data['flag'])
	data['attack'] = le.fit_transform(data['attack'])
	# All the features apart from Attack are what we are going to use to predict the attack status of the data
	# attack = 1 (normal/not an attack) and attack = 0 (attack)
	X = data.drop(['attack'],axis=1).to_numpy()
	Y = data['attack'].to_numpy()
	# Splitting X and y testing and training data
	# we are taking 20% of the data for testing and 80% of the data for training 
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)
	# reshaping y test and train array
	y_train = y_train.reshape(len(y_train),1)
	y_test = y_test.reshape(len(y_test),1)
	# Getting input and output layer dimensions
	input_dim = X_train.shape[1]
	output_dim = y_train.shape[1]

	# TESING AND TRAINING OUR NEURAL NETWORK MODEL

	# Basic Neural Network
	#Defining a sequential model with 3 layers
	model = tf.keras.models.Sequential([
	Dense(units = 16, input_shape =(input_dim,),activation = 'relu'),
	Dense(units = 16, activation = 'relu'),
	Dense(units = 2, activation = 'softmax')])
	# Compiling the NN with adam optimizer and sparse_categorical_crossentropy loss
	model.compile(optimizer = "adam",loss = "sparse_categorical_crossentropy",metrics=["accuracy"])
	# Fitting the training data 
	model.fit(X_train,y_train, epochs = 10,batch_size=32)
	# Predicting 
	prediction = model.predict(X_test,verbose = 0)
	# since our predictions here will be probabilites, the class with the maximum probability 
	# will be the prediction     
	y_pred = np.argmax(prediction,axis = -1)
	# Confusion matrix heatmap
	conf_matrix = confusion_matrix(y_test, y_pred)
	fig, ax = plt.subplots()
	ax = sns.heatmap(conf_matrix, annot=True, fmt="d")
	# The top and bottom of heatmap gets trimmed off so to prevent that we set ylim
	bottom, top = ax.get_ylim()
	ax.set_ylim(bottom + 0.5, top - 0.5)
	st.write("CONFUSION MATRIX")
	st.pyplot(fig)
	pred = y_pred.astype(int)
	st.write("ACCURACY :",accuracy_score(y_test,pred))
	# model accuracy
	st.success("Code Execution Successful")
else:
	st.warning("No file has been chosen yet")































