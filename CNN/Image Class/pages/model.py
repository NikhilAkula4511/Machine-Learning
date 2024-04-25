import numpy as np
import streamlit as st
import pandas as pd
import keras 
from keras.models import Sequential
from keras.layers import Dense,MaxPool2D,Dropout,InputLayer,Conv2D,Flatten
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings("ignore")
labels=pd.read_json(r'C:\Users\Hello\Deep Learning\CNN\data_1\data\class_dict_10.json')
import os
import cv2
fv=[]
cv=[]
for  folder,subfolder,files in os.walk(r'C:\Users\Hello\Deep Learning\CNN\data_1\data\class_10_train'):
    for file in files:
        if file.endswith(".JPEG"):
            path = os.path.join(folder,file)
            fv.append(cv2.imread(path))
            label = folder.split("\\")[-2]
            cv.append(labels.loc['class'][label]) 
            
labels_val = pd.read_json(r'C:\Users\Hello\Deep Learning\CNN\data_1\data\val_class_dict_10.json')
import os
import cv2
fv2=[]
cv22 = []
for  folder,subfolder,files in os.walk(r'C:\Users\Hello\Deep Learning\CNN\data_1\data\class_10_val\val_images'):
    for file in files:
        if file.endswith(".JPEG") & ~(file.endswith("checkpoint.JPEG")):
            path = os.path.join(folder,file)
            fv2.append(cv2.imread(path))
            label = labels_val.loc['class'][file]
            cv22.append(labels.loc['class'][label])

X_train = np.asarray(fv)
X_valid = np.asarray(fv2)
from sklearn.preprocessing import LabelEncoder
encode =LabelEncoder()
y_train = encode.fit_transform(cv)
y_valid = encode.transform(cv22)

model = Sequential()

model.add(InputLayer(input_shape=(64,64,3)))
model.add(Conv2D(filters=10,kernel_size=(3,3),strides=(1,1),padding='valid',activation="relu"))
model.add(Conv2D(filters=10,kernel_size=(3,3),strides=(1,1),padding='valid',activation="relu"))

model.add(MaxPool2D(pool_size=(2, 2),strides=None,padding='valid'))
model.add(Conv2D(filters=10,kernel_size=(3,3),strides=(1,1),padding='valid',activation="relu"))
model.add(Conv2D(filters=10,kernel_size=(3,3),strides=(1,1),padding='valid',activation="relu"))

model.add(MaxPool2D(pool_size=(2, 2),strides=None,padding='valid'))
model.add(Flatten())
model.add(Dense(units=50,activation="relu"))
model.add(Dense(units=20,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=10,activation="sigmoid"))

model.compile(optimizer="adam",loss='sparse_categorical_crossentropy',metrics=["accuracy"])

st.header("Model layers")

st.write(model.layers)

history = model.fit(X_train,y_train,epochs=20,validation_data=(X_valid,y_valid))


fig, ax = plt.subplots()
ax.plot(range(1, 21), history.history['loss'], label='train')
ax.plot(range(1, 21), history.history['val_loss'], label='test')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training and Validation Loss')
ax.legend()

# Display the plot in Streamlit

st.header("Train vs test loss ")
st.pyplot(fig)
            
