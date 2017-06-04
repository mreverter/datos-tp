from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

train = pd.read_csv('trip_train.csv')
train = train.values

#train_X = train[:,[0,4,7,8]]
train_X = train[:,[4,7]]
train_Y = train[:,1]

np.random.seed(0)

model = Sequential()
model.add(Dense(2,input_dim=2,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='mean_absolute_percentage_error',optimizer='adam',metrics=['accuracy'])
model.fit(train_X,train_Y,epochs=20,batch_size=5)

scores = model.evaluate(train_X,train_Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

test = pd.read_csv('trip_test.csv')
test = test.values
#test_X = test[:,[0,3,6,7]]
test_X = test[:,[3,6]]

predictions = model.predict(test_X)

rounded = [round(x[0]) for x in predictions]
print(rounded)