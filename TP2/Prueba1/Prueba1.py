from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

#Parametros
epoch = 1
batch = 32
umbral = 0.500

#Cargamos los datos
train = pd.read_csv('train.csv')
train['duration'] = train['duration'].map(lambda x: bin(x)[2:].zfill(32))

for i in range(0,32):
    train['bit'+str(i)] = train['duration'].map(lambda x: x[i])
    
train = train.values

train_X = train[:,[0,1,2,3,4,5,6,7,9,10]]
train_Y = train[:,11:43]

np.random.seed(7)

input_layer = train_X.shape[1]
output_layer = train_Y.shape[1]

print("")
print("Entrenando modelo...")
print("")

model = Sequential()
model.add(Dense(input_layer,input_dim=input_layer))
model.add(Dense(18,activation='relu'))
model.add(Dense(output_layer,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(train_X,train_Y,epochs=epoch,batch_size=batch,verbose=1)

scores = model.evaluate(train_X,train_Y)

print("")
print("Precision del modelo:")
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("")
print("Entrenando modelo... OK")

print("")
print("Salvando el modelo en archivo")

model.save('model_train.h5')

print("")
print("Salvando el modelo en archivo... OK")

test = pd.read_csv('test.csv')
test = test.values

test_X = test[:,[0,1,2,3,4,5,6,7,8,9]]

print("")
print("Prediciendo valores...")

predictions = model.predict(test_X)

#Creo el csv para las predicciones (lo que hay que subir a kaggle)
predictions_df = pd.DataFrame(predictions)
predictions_df = predictions_df.applymap(lambda x: 1 if (x >= umbral) else 0)

print("")
print("Prediciendo valores... OK")

print("")
print("Transformando el formato de duraciones predecidas...")

durations = [];
for i in range(0,predictions_df.shape[0]):
    binary = ''
    for j in range(0,predictions_df.shape[1]):
        binary = binary + str(predictions_df.iloc[i,j])
    durations.append(int(binary,2))

print("")
print("Transformando el formato de duraciones predecidas... OK")

print("")
print("Generando csv de predicciones...")
  
durations_df = pd.DataFrame(durations,columns={'duration'})
test = pd.read_csv('test.csv')
durations_df['id'] = test['id']
durations_df = durations_df[['id','duration']]
durations_df.to_csv('predict.csv',index=False)

print("")
print("Generando csv de predicciones... OK")
print("Procedimiento finalizado")
