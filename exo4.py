import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import *
from keras.layers import *

##Load the data set and make a preprocessing

df = pd.read_csv(r'C:\Users\Corentin BATUT\Documents\COURS UEF\AI\household_power_consumption.txt')
df.dropna(inplace=True)
df = df.drop(['Date'], axis=1)


##Transform the dataset to 7 columns of the variables at (t-1) and 1 column of the Global_active_power at time (t)

X = np.array(df.drop(["Time"],axis=1))
Y = X[:-1,:]
C = X[1:,0]
A = np.zeros((np.shape(Y)[0],8))
for i in range(np.shape(Y)[0]):
    l0 = X[i]
    c0 = np.array([C[i]],dtype=object)
    A[i] = np.concatenate((l0,c0),axis=0)

## Split the data to train and validation sets

X_train, X_test, y_train, y_test = train_test_split(A[:,:-1], A[:,-1], test_size=0.2)
X_train = X_train.reshape(1639423, 1, 7)
y_train = y_train.reshape(1639423,1,1)
X_test = X_test.reshape(409856, 1, 7)
y_test = y_test.reshape(409856,1,1)

## LSTM network
model = Sequential()

model.add(keras.layers.LSTM(1,dropout=0.2, return_sequences=True,input_shape=(1, 7)))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train, y_train, epochs=2,validation_data=(X_test, y_test),verbose=0)

At = model.predict(X_test)
average_error = np.mean(At-y_test)


## results

# for epochs = 2
# a = average_error = -0.46388119861018823
# m =mean(y_test) = 1.0921343447454714
# a/m = about -42% ==> the error is not negligible

# for epochs = 5
# a = average_error = -0.44618045872993584
# m =mean(y_test) = 1.0921343447454714
# a/m = about -40.8% ==> the error is lower but not negligible
# It seems that we can obtain good results by increasing epochs but it's too complex for my computer.y