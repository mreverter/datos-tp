from keras.models import load_model
import pandas as pd

#Parametros
umbral = 0.500
entrada = [0,1,2,3,4,5,6,7,8,9]

print("Cargando modelo...")

model = load_model('model_train.h5')
test = pd.read_csv('test.csv')
test = test.values

print("")
print("Cargando modelo... OK")

test_X = test[:,entrada]

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
durations_df.to_csv('tpDatos_predictions.csv',index=False)

print("")
print("Generando csv de predicciones... OK")
print("Procedimiento finalizado")
