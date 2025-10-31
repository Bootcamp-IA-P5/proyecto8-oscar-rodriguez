# Mushrooms dataset

Este README contiene un resument de los pasos seguidos en el notebook que podemos encontrar en este repositorio

## 1. Intalacion de las librerias
Este apartado solo cubre la instacion de las librerias requeridas. Puedes o bien ejecutar el apartado del notebook donde se hace la carga de las mismas o usar `pip install -r requirements.tzt`

(consultar el notebook para mas detalles)

**NOTA** Asegurate de tener un entorno virtual activado para Pyhton o de estar usando contenedores en visual studio code.

## 2. Exploración y carga de datos

Cargamos el dataset `mushrooms.csv` y realizamos un conteno de variables nulas que arroja que no hay varibles nulas (el dataset esta completo). A continuación pasamos a analizar que variables pueden ser descartadas a priori. Para ello contamos los valores únicos que pueden tomar cada una de las variables del dataset y observamos que algunas de ellas como `veil-type` pueden ser eliminadas porque solo tienen un posible valor para todo el dataset y por tanto pueden ser eliminadas con seguridad.

Para determinar otras variables que pueden ser eliminadas usamos un metodo llamado 'Mutual Exclusion Classifier' que es un valor no negativo que mide la dependencia entre dos variables. Cuanto mas cerca de cero esta ese valor, mas independientes son las variables. Calculamos este valor para todas las variables independientes contra la variable dependiente y descartamos todas aquellas variables que tengan un valor menor de un umbral determinado (que hemos fijado a 0.05). De esta forma, descartamos del dataset inicial las siguientes variables:

```
'cap-color', 'cap-shape', 'ring-number', 'cap-surface', 'veil-color', 'gill-attachment', 'stalk-shape', 'veil-type'
```
y nos quedamos con las siguientes variables:

```
'odor', 'spore-print-color', 'gill-color', 'ring-type', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'gill-size', 'population', 'bruises', 'habitat', 'stalk-root', 'gill-spacing'

```
(consultar el notebook para mas detalles)

**NOTA** Despues de este paso, tenemos dos datasets: `df`, que contiene todas las variables, y `df_final`, que contiene solo las variables que el método 'Mutual Exclusion Classifier' ha determinando como independientes.

## 3. Preprocesamiento del dataset

Buscamos valores faltantes (que no hay), codificamos las variables categoricas con one-hot encoder y separamos el dataset en dos: el formado por las variables independientes (X), y el formado por la variable dependiente (y). 

(consultar el notebook para mas detalles)

**NOTA** En previsión de futuros usos, a partir de ahora en el notebook se van a crear métodos para realizar operaciones repetitivas, como es el caso del one-hot encoder.

```
def one_hot_encoding(df_model: pd.DataFrame, target: str) -> pd.DataFrame, pd.DataFrame:

```
## 4. PCA

Con PCA vamos a reducir el conjunto total de variables independientes a dos. Como en el caso anterior, creamos un método que se encarga de reducir un modelo a dos variables independientes. Tambien creamos un grafico con la distribución de nuestra variable dependiente de acuerdo a las dos variables independientes obtenidas al usar PCA.

(consultar el notebook para mas detalles)

```
def pca(X: pd.DataFrame, pca_value: float) -> PCA, pd.DataFrame:
```

## 5. Random Forest

Una vez que hemos visto todo lo anterior, pasamos a entrenar nuestro dataset con el modelo de Random Forest. Para hacer una comparativa, usamos varios tipos de datasets (la funcion `train_random_forest` se encarga de hacerlo por nosotros).

- Dataset completo.
- Dataset despues de reducir variables usando el metodo 'Mutual Exclusion Classifier`.
- Caso especial 1. Tomando solo la variable `odor`como variable independiente.
- Caso especial 2. Eliminando la variable `odor` de la lista de variables independientes y dejando las demas.
- Dataset obtenido al aplicar PCA.

Los resultados muestran que en los dos primeros casos, la precisión se mantuvo en 1, en el tercero bajó a 0.99, en el cuarto volvió a subir a 1, y en el último bajo hasta el 0.77.

(consultar el notebook para mas detalles)

## 6. KNN

Como ultimo paso, evaluamos el dataset usando KNN. Lo primero que hacemos es determinar cuantos clusters queremos usar. En buena lógica y al tratarse de un dataset para predecir un resultado binario (seta comestible o venenosa), consideramos que el tamaño de cluster ideal es 2.

Los resultados muestran que la configuración no es mala, pero queremos ver si podemos determinar con precisión el numero de clusters. Para ello usamos un metodo llamado `shiluette` que en teoría nos va a ayudar a encontar el tamaño de cluster ideal. Despues de ejecutar el proceso, nos dice que el tamaño de cluster ideal para nuestro dataset es 9, que esta muy lejos de lo que habiamos pensado (2). Esto no quiere decir que el tamaño de cluster 2 es malo, sino que por la misma naturaleza del dataset, al intentar encontrar el numero optimo de clusteres, el proceso ha encontado estructuras complejas dentro de los resultados esperados (comestible o venenosa) que se pueden ajustar mejor para el entrenamiento de modelos.

(consultar el notebook para mas detalles)