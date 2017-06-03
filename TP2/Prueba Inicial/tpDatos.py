#Ejecucion del algoritmo con redes neuronales usando Keras y TensorFlow

#Primero importamos las librerias para el modelo y las capas
from keras.models import Sequential
from keras.layers import Dense
import numpy

#1. Cargamos el set de datos
dataset=numpy.loadtxt('pima-indians-diabetes.csv', delimiter=',')

#Luego realizamos una division en el set de datos

#El X va a contener las primeras 7 columnas del set de datos, 
#siendo los valores de entrada

#El Y va a contener la columna 8 del set (siendo la ultima columna),
#que contiene los valores que la red neural tiene que predecir a partir
#de los datos de entrada X

X = dataset[:,0:8]
Y = dataset[:,8]

#2. Definimos el modelo

#Creamos un modelo secuencial donde agregamos las capas
#Debemos asegurarnos que la primera capa de entrada contenga el numero
#correcto de entradas, como tenemos 8 entradas (o sea las columnas que
#tiene X), establecemos en 8 el segundo argumento de Dense con input_dim.

#En cuanto a la cantidad de capas, es algo que es a prueba y error, podemos
#agregar las que queramos (por supuesto que cuanto mas capas, mas lento
#es el entrenamiento). En este caso vamos a usar 3 capas.

#En la funcion Dense definimos la cantidad de neuronas en el 1er argumento,
#la cantidad de entradas en el segundo argumento, y la funcion de activacion
#en el tercer argumento.

#A continuacion definimos el modelo y cada una de las siguientes lineas
#con model.add define una capa, como usamos 3, vamos a tener 3 lineas
#de model.add.

#La primer capa tiene 12 neuronas, la segunda 8 y la tercera 1, que sirve
#para predecir la clase (si el paciente tiene diabetes es 1, sino 0).

#El peso para cada uno de los valores de entrada son asignados con valores
#aletorios entre 0 y 0.05 por keras. Esto puede provocar que por cada
#ejecucion del algoritmo tengamos distintas predicciones, para evitarlo
#agregamos la siguiente linea de codigo.

numpy.random.seed(0)

#Esto hace que cuando usemos funciones random, estas devuelvan siempre el
#mismo valor en cada ejecucion.

#Continuando, creamos el modelo

model = Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))      #1era capa (entrada)
model.add(Dense(8,activation='relu'))                   #2da capa (oculta)
model.add(Dense(1,activation='sigmoid'))                #3ra capa (salida)

#3. Compilar el modelo

#Ahora que el modelo está definido, lo vamos a compilar. Aca entran los
#backends de keras, como las librerías Theano o TensorFlow.

#El backend selecciona automaticamente la mejor forma de representar la red
#para el entrenamiento y hacer que las predicciones se ejecuten en el
#hardware, ya sea en la CPU o en la GPU.

#Cuando compilamos, debemos especificar algunas propiedades adicionales
#que se necesitan para entrenar la red. Recordar que entrenar una red
#significa buscar el mejor conjunto de pesos para hacer predicciones para
#este problema.

#Compilamos el modelo
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#4. Ajustar modelo

#Ahora que definimos y compilamos el modelo. Es hora de ejecutar el modelo
#en algun dataset. Aca podemos entrenar o ajustar nuestro modelo en nuestro
#dataset que cargamos anteriormente usando la funcion fit() en el modelo.

#El proceso de entrenamiento se ejecutara para numero fijo de iteraciones
#a traves del dataset llamado epochs, que debemos especificarlo usando el
#argumento epochs. Tambien podemos establecer el numero de instancias que
#seran evaluadas antes de realizar una actualizacion de pesos en la red,
#llamado batch size (tamanio del lote) y lo establecemos usando el argumento
#batch_size. 

#Tanto el epochs como el batch_size, son argumentos cuyos valores a pasar
#son a prueba y error. Para este problema usaremos un numero chico de
#iteraciones (150) y un numero chico de instancias (10).

#Ajustamos el modelo (aca es donde el trabajo sucede en la CPU o GPU)
model.fit(X,Y,epochs=150,batch_size=10)

#5. Evaluar el modelo

#Ya tenemos entrenado nuetra red neuronal en todo el dataset y podemos
#evaluar el rendimiento de la red con el mismo dataset. Esto nos dara una
#idea de lo bien que hemos modelado el dataset (como por ejemplo, la
#exactitud o presicion del entrenamiento que lo hicimos con el 'accuracy'
#al compilar el modelo), pero no nos da ninguna idea de lo bien que el
#algoritmo puede realizar con los nuevos datos. Hemos hecho esto por
#simplicidad, pero idealmente, usted podria separar sus datos en dos
#datasets: train y test, para entrenar (train) y evaluar (test) su modelo.

#Puede evaluar su modelo en su dataset entrenado usando la funcion
#evaluate() en su modelo y pasarle la misma entrada X y salida Y que fue
#utilizada para entrenar el modelo.

#Esto generara una prediccion para cada par de entrada y salida, y
#obtendra sus puntuaciones (scores), incluyendo la perdida promedio y
#cualquier metrica que haya configurado, como la precision (accuracy)

#Evaluamos el model
scores = model.evaluate(X,Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#Al ajecutar el codigo, al final de la terminal nos va a aparecer un
#acc: 76.17%, que muestra la precision del modelo. Notemos que a mayor
#epochs y mayor batch_size, la precision del modelo es mejor. Dicha
#precision tambien es alterada por la cantidad de capas y neuronas que
#establezcamos al definir el modelo.

#6. Hacer predicciones

#Podemos adaptar el ejemplo anterior y usarlo para generar predicciones
#en el dataset entrenado, pretendiendo que es un nuevo dataset que no
#hemos visto antes.

#Hacer predicciones es tan facil como llamar model.predict(). Como estamos
#usando una funcion sigmoide en la capa de salida, las predicciones
#estarán en el rango entre 0 y 1. Nosotros podemos convertir esto en una
#prediccion binaria (donde devuelva 0 o 1) redondeando el resultado
#por la funcion sigmoidal aplicando round().

#Calculamos predicciones
predictions = model.predict(X)
#Redondeamos las predicciones
rounded = [round(x[0]) for x in predictions]
print(rounded)
