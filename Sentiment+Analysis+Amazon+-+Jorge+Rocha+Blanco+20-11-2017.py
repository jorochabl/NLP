
# coding: utf-8

# # Análisis de sentimientos con reviews de productos de Amazon España (opcional)

# Si has hecho ya el ejercicio de web scraping con `Requests` y `BeautifulSoup` habrás visto cómo extraer datos de una página web.
# 
# El dataset que utilizarás en este ejercicio (que no es obligatorio entregar) lo he generado utilizando `Scrapy` y `BeautifulSoup`, y contiene unas $700.000$ entradas con dos columnas: el número de estrellas dadas por un usuario a un determinado producto y el comentario sobre dicho producto; exactamente igual que en el ejercico de scraping.
# 
# Ahora, tu objetivo es utilizar técnicas de procesamiento de lenguaje natural para hacer un clasificador que sea capaz de distinguir (¡y predecir!) si un comentario es positivo o negativo.
# 
# Es un ejercicio MUY complicado, más que nada porque versa sobre técnicas que no hemos visto en clase. Así que si quieres resolverlo, te va a tocar estudiar y *buscar por tu cuenta*; exactamente igual que como sería en un puesto de trabajo. Dicho esto, daré un par de pistas:
# 
# + El número de estrellas que un usuario da a un producto es el indicador de si a dicho usuario le ha gustado el producto o no. Una persona que da 5 estrellas (el máximo) a un producto probablemente esté contento con él, y el comentario será por tanto positivo; mientras que cuando una persona da 1 estrella a un producto es porque no está satisfecha... 
# + Teniendo el número de estrellas podríamos resolver el problema como si fuera de regresión; pero vamos a establecer una regla para convertirlo en problema de clasificación: *si una review tiene 4 o más estrellas, se trata de una review positiva; y será negativa si tiene menos de 4 estrellas*. Así que probablemente te toque transformar el número de estrellas en otra variable que sea *comentario positivo/negativo*.
# 
# Y... poco más. Lo más complicado será convertir el texto de cada review en algo que un clasificador pueda utilizar y entender (puesto que los modelos no entienden de palabras, sino de números). Aquí es donde te toca investigar las técnicas para hacerlo. El ejercicio se puede conseguir hacer, y obtener buenos resultados, utilizando únicamente Numpy, pandas y Scikit-Learn; pero siéntete libre de utilizar las bibliotecas que quieras.
# 
# Ahora escribiré una serie de *keywords* que probablemente te ayuden a saber qué buscar:
# 
# `bag of words, tokenizer, tf, idf, tf-idf, sklearn.feature_extraction, scipy.sparse, NLTK (opcional), stemmer, lemmatizer, stop-words removal, bigrams, trigrams`
# 
# No te desesperes si te encuentras muy perdido/a y no consigues sacar nada. Tras la fecha de entrega os daré un ejemplo de solución explicado con todo el detalle posible.
# 
# ¡Ánimo y buena suerte!

# # Carga de Datos y Volcado en Dataframe
# Lo primero que haremos es cargar los datos, para así poder vectorizar y tratar los diferentes comentarios.

# In[1]:


import numpy as n
import pandas as pd

import os
#Almacenamiento a disco de modelos
from sklearn.externals import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

#Una de las cosas que hace falta es detectar el lenguaje
from langdetect import detect

#Librerías de Natural Language ToolKit
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, download
from nltk.data import load
from nltk.stem import SnowballStemmer
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

#Importo este paquete para poder pasar como parámetro (y guardar) las funciones de tokenización
import pickle
import dill


# ## Funciones Auxiliares
# Vamos a crear varias funciones para tratar los datos:
# - Tokenizador: función `tokenizador(texto, stemmer, simbolos_excl, idioma = 'es')`. Se le pasa un texto (comentario o lo que queramos), el `stemmer` a aplicar, la lista de símbolos a excluir y el idioma en codificación estándar (la que devuelve la función `detect()`, por ejemplo.
# - Filtrado de DataSet por idiomas. Función `getDS_por_idioma(df, campo_idioma = "idioma")`, toma un DF con un campo *idioma* y devuelve una lista de DataSets con la mism estructura, pero sin ese campo y organizados en forma de Diccionario `{"<idioma>": df}`.
# - Cálculo de modelos de Machine Learning para un idioma concreto. `getModeloML_SentimentAnalysis(df_x, y, stop_words, parametros, funcionML = RandomForest(...), tokenizador(<texto>), imprimirResumenModelo = True, scoring = "f1")`, devuelve el modelo que mejor funciona. Los parámetros para el modelo deben llevar el prefijo `funcionML__` y los parámetros y valores que mejor apliquen para dicho modelo, a juicio del usuario. El resto de parámetros, para el `CountVectorizer()`, van implicitos en la función. El tokenizador que se le pasa debe aceptar 1 parámetro de tipo *texto*; si la función utilizada usa en realidad más parámetros, se puede crear una **lambda** que le indique el resto de parámetros.

# In[29]:


#Función para "tokenizar" los textos. Toma la lista de "stems" para el idioma en el que esté el texto.
#El usuario debe identificar qué idioma es ese, o tomará "es" por defecto.
def tokenizador(texto, stemmer, simbolos_excl, idioma = "es"):
#    stemmer = dic_stemmers[idioma]
    txt_resultado = ''.join([car for car in texto if car not in simbolos_excl])  #quita los caracteres a excluir
    tokens =word_tokenize(txt_resultado)    #nltk, divide en palabras
    try:
        stems_texto = [stemmer.stem(item) for item in tokens]
    except Exception as e:
        stems_texto = ['']
        print(texto + ", excepcion" + e)
#    print(stems_texto)
    return stems_texto

#Función auxiliar para el ejercicio. Toma un df con un campo de idioma y devuelve dicho DF troceado en un diccionario
#donde cada idioma tiene una etiqueta "<idioma>"
def getListaDF_idiomas(df, col_idioma = "idioma", imprimir_idiomas = True):
    df_aux_idiomas = df.groupby(by=col_idioma).count()
    lista_idiomas = df_aux_idiomas.index    #agrupado por idioma, los índices serán la lista única de idiomas
    if imprimir_idiomas:
        print("Listado de idiomas en el DF:\n" + str(lista_idiomas))
    dicc_dfs = {}
    for idioma in lista_idiomas:
        try:
            dicc_dfs[idioma] = df[df[col_idioma] == idioma]   #filtrar las filas para el idioma en cuestión
        except Exception as e:
            print("Error en " + str(idioma))
            continue
    return dicc_dfs

#Función para entrenar modelos en cada idioma. Se le pasa el DF, las stop_words, los parámetros de la función
#de machine learning y la propia función de machine learning, y si se imprime o no el scoring del mejor modelo.
#También la función de evaluación del modelo que se quiere usar, por defecto "f1".
def getModeloML_SentimentAnalysis(df_x, y, stop_words, parametros, tokenizador = None, 
                                  funcionML = LinearSVC(), 
                                  imprimirResumenModelo = True, scoring = "f1", n_jobs = 4, verbose = 1):
    #Antes de crear el CountVectorizer, tenemos que "reconstruir" (usando dill) la función de tokenizador, serializada:
#    if tokenizador != None:
#        f_tokenizador = dill.loads(tokenizador)
#    else:
#        f_tokenizador = None
    #Creamos el Vectorizer
    vectorizador = CountVectorizer(
            analyzer = 'word',
 #           tokenizer = f_tokenizador,    #función recién deserializada, ojo
            tokenizer = tokenizador,
            lowercase = True,
            stop_words = stop_words
    )
    pipeline = Pipeline(
            [("vect", vectorizador),
            ("funcion_ML", funcionML)])      #como indicábamos arriba, le pasamos la función recibida y se etiqueta así
    parametros_GS = {
            'vect__max_df': [0.5, 1.0],
            'vect__binary': [True, False],
            'vect__ngram_range': [(1,1), (1,2)]
    }
    parametros_GS.update(parametros)         #no deja concatenar los dos diccionarios en una línea
##############
#    parametros_GS = {}    #para pruebas
##############
    if imprimirResumenModelo:
        print("Parámetros para el GridSearchCV (etiquetas):\n----------------------------")
        print(parametros_GS.keys())
        print("----------------------------")
    #Creamos ya el GridSearchCV
    gridsearch = GridSearchCV(pipeline, parametros_GS, n_jobs=n_jobs, scoring=scoring, verbose=verbose)
    print("Modelo con GridSearchCV() creado")
    #Al parámetro X hay que pasarle exclusivamente las "features" que vaya a usar, si hay que quitar campos, hacerlo antes.
    gridsearch.fit(X=df_x, y=y)
    print("fit() sobre el modelo ejecutado")
    #Devolvemos el mejor modelo, e imprimimos el scoring si está indicado en los parámetros
    if imprimirResumenModelo:
        print("Mejor scoring:")
        print(gridsearch.best_score_)
        print("Mejores parámetros:")
        print(gridsearch.best_estimator_)
    return gridsearch      #devolvemos el objeto GridSearchCV(), y así podemos obtener luego el scoring y el best_model


# Cargamos los datos en un DataFrame, para a partir de ahí pasar a procesarlos. Convertimos la clasificación de 
# los comentarios, con un campo nuevo, en **bueno** (4, 5) y **malo** (1, 2, 3)

# In[3]:


df_comentarios = pd.read_csv("amazon_es_reviews.csv", sep=";")


# In[4]:


#Lo separamos en otra celda, para evitar recargar cada vez el DF
df_comentarios.head()


# In[5]:


lista_calificaciones = [int(r >= 4) for r in df_comentarios["estrellas"]]
df_comentarios["resultado"] = lista_calificaciones
del lista_calificaciones
df_comentarios.head()


# Vamos a usar `nltk` para apoyar el procesamiento previo de las cadenas de texto:
# - Eliminar **stop words** (signos y palabras que no *aportan* significado, tales como puntuación, conjunciones, etc.
# - *Tokenizar* las cadenas de texto. Sobre todo, separar en palabras.
# 

# In[6]:


#Cargamos el soporte para las "stop_words" de los diferentes idiomas
#nltk.download()    #una vez cargado, mejor no ejecutarlo, o abre un cuadro de diálogo


# In[7]:


#Para cada fila, hay que detectar el idioma y apuntarlo.
#ESTA OPERACIÓN ES PESADA, son > 700.000 registros. No he conseguido encontrar parámetro u opción para que use 4 CPU
lista_idiomas = []
count_errores = 0
for i, fila in df_comentarios.iterrows():
    try:
        lista_idiomas.append(detect(fila["comentario"]))
    except Exception as e:     #ignoramos cualquier excepción que haya y, en ese caso, insertamos ('-')
        count_errores += 1
        lista_idiomas.append('-')
        pass
    if (i+1)%1000 == 0:
        print("\t" + str(i+1), end="")
    if (i+1)%10000 == 0:
        print("\n" + str(i+1), end="")
print(lista_idiomas[:100])
#Añadimos los datos al Dataframe de comentarios, y sacamos un ranking de comentarios por idioma
df_comentarios["idioma"] = lista_idiomas


# Para realizar pruebas más rápidamente, ya que la detección de idiomas toma bastante tiempo (varias horas para el DF completo), vamos a grabar el resultado en disco y recuperarlo bajo demanda. De esta forma, cuando haya que parar y relanzar el proceso, bastará con recargar el detaset y "saltarse" el paso previo.

# In[8]:


#Este código debe ejecutarse sólo una vez, después del cálculo del idioma en el dataset anterior.
df_comentarios.to_csv("df_comentario_idioma.csv", sep=";", encoding="utf_8")


# In[ ]:


#Este código es para poder recuperar el dataset con los idiomas ya puestos, saltándonos el análisis a efectos de testing.
df_comentarios = pd.read_csv("df_comentario_idioma.csv", sep=";", encoding="utf_8")


# *Fin código auxiliar testing*

# In[9]:


#Creamos una lista de diccionarios con los idiomas y nos quedamos sólo con los principales: 
# "es", "en", "ca", "fr", "it", "de", "pt"
dicc_df_aux = getListaDF_idiomas(df_comentarios, imprimir_idiomas=False)
dicc_df_comentarios = {}
for idioma in dicc_df_aux:   #iteramos sobre el diccionario auxiliar, para poder recuperar clave y valor
    if idioma in ["es", "en", "pt"]:   #he tenido que quitar la mayoría, porque daban error
        #reducimos los elementos del diccionario, y nos quedamos sólo con los elelementos de estos idiomas
        dicc_df_comentarios[idioma] = dicc_df_aux[idioma]
        print("Idioma: " + str(idioma))
        print(dicc_df_aux[idioma].head())
del dicc_df_aux   #para liberar memoria
print(dicc_df_comentarios.keys())


# Vamos a crear un dataset de entrenamiento y otro de testing, para todas las pruebas que hay que llevar a cabo. El DF de entrenamiento será del 80%.

# In[10]:


#Creamos split de datos: train y test, para cada idioma, y los guardamos en un diccionario
dicc_df_train = {}
dicc_df_test = {}
for idioma in dicc_df_comentarios:
    dicc_df_train[idioma], dicc_df_test[idioma] = train_test_split(dicc_df_comentarios[idioma],
                                                                   test_size=0.2, train_size=0.8)
    print("Tamaño del DF de entrenamiento " + idioma + ": " + str(len(dicc_df_train[idioma])))
    print("Tamaño del DF de testing " + idioma + ": " + str(len(dicc_df_test[idioma])))


# Crearemos los datos de `stop_words` y `stems` para cada idioma. El problema es que en este caso no vale el código abreviado de idioma, sino que precisa una palabra completa. Ej: `"es" --> "spanish", "en" --> "english"`, así que tengo que crearme otro diccionario más para traducir. Se llamará `yad`, en inglés...

# In[11]:


#Hay que traducir los códigos de idioma para que los reconozca, así que no queda otra que crear oootro diccionario
yad = {"es": "spanish", "en": "english", "fr": "french", "it": "italian", "pt": "portuguese", "de": "german"}

#Primero, vamos a buscar aquellas palabras y signos que debemos ignorar: puntuación, números, y las "stop words" 
#que identifica NLTK para cada idioma
lista_idiomas = dicc_df_comentarios.keys()
#Ahora incluimos aquello que no son símbolos que forman parte de palabras (puntuación, números)
simbolos_a_excluir = list(punctuation)
simbolos_a_excluir.extend(map(str, range(10)))
simbolos_a_excluir.extend(["¿","¡"])  #números y signos adicionales

#Usamos el diccionario "yad" para pasar a las funciones, pero volvemos a guardarlo bien (código 2 letras x idioma)
#Creamos y rellenamos diccionarios de "stems" y de "stop_words"
dicc_stopwords = {} 
dicc_stems = {}
for idioma in lista_idiomas:   #stopwords para cada idioma
    dicc_stopwords[idioma] = stopwords.words(yad[idioma])
    dicc_stems[idioma] = SnowballStemmer(yad[idioma])

print(lista_idiomas)
print("")
print(dicc_stopwords["es"])
print("")
print(simbolos_a_excluir)
print("")
print(dicc_stems["es"])


# ## Generación de un diccionario, por Idiomas, con un Diccionario de modelos
# Partiendo de los diccionarios anteriores, voy a lanzar el entrenamiento de los modelos y almacenar sus resultados, para después hacer scoring. Haremos un bucle en el que recorreremos todos los países de los diccionarios, y para cada uno de ellos lanzaremos los diferentes entrenamientos de modelos, para ver cuál es el mejor.  
#   
# Los resultados de cada país se almacenarán, igualmente, en un diccionario, con una `clave` para cada país, y otro diccionario de modelos:
# - `Best`. El modelo con mejor scoring, guardamos la etiqueta para recoger luego el modelo concreto por dicha etiqueta.
# - `SVC`. Modelo de SVC, con los hiperparámetros con mejor *scoring*.
# - `RFC`. Modelo de Random Forest, con los hiperparámetros con mejor *scoring*.
# - `RNC`. Red Neuronal Convolucional, idem.

# In[18]:


###Prueba de tokenizador  -- no es parte de la ejecución, sólo son pruebas
func_tokeniz_prueba = lambda texto: tokenizador(texto, stemmer=dicc_stems['es'], simbolos_excl=simbolos_a_excluir,
                                                 idioma="es")
#tokens_prueba = tokenizador(
#    "El perro de San Roque no tiene rabo, porque Ramón Rodríguez se lo ha cortado. Bueno, bondad, buenismo.",
#                           dicc_stems, simbolos_a_excluir, idioma="es")
tokens_prueba = func_tokeniz_prueba(
    "El ¡perro! de San ¿Roque? no tiene rabo, porque; Ramón Rodrí1guez 20$ 233 se lo ha cortado. Bueno, bondad, buenismo.")

tokens_prueba = func_tokeniz_prueba(
    "El perro de San Roque no tiene rabo, porque Ramón Rodríguez se lo ha cortado. Bueno, bondad, buenismo.")

### Fin pruebas


# ### En esta parte, hacemos ya las llamadas a la función genérica de ML. Pasos:
# - Crear diccionario de funciones `lambda` para cambiar la morfología y parámetros del tokenizador, paramétricamente.
# - Invocar la función genérica de ML parametrizada, y almacenar `metadiccionario' de resultados:`
#     + **1er nivel** --> *clave* **idioma**.
#         + **2º nivel** --> *clave* modelo (**Best, SVC, KN-Neighbors, Red Neuronal**).
#         + En el 2º nivel el valor de *Best* se puede usar como clave para obtener el propio modelo con hiperparámetros.

# In[30]:


#Creamos las funciones lambda para "es", "en" y "pt", que serán recuperadas de un diccionario para el algoritmo
#de Machine Learning. Se pueden crear como funciones Lambda o de ámbito global
def tokenizador_es(texto):
    return tokenizador(texto,stemmer=dicc_stems["es"],simbolos_excl=simbolos_a_excluir,idioma="es")
def tokenizador_en(texto):
    return tokenizador(texto,stemmer=dicc_stems["en"],simbolos_excl=simbolos_a_excluir,idioma="en")
def tokenizador_pt(texto):
    return tokenizador(texto,stemmer=dicc_stems["pt"],simbolos_excl=simbolos_a_excluir,idioma="pt")
dicc_tokenizadores = {}
dicc_tokenizadores["es"]=lambda texto:tokenizador(texto,stemmer=dicc_stems["es"],
                                                  simbolos_excl=simbolos_a_excluir,idioma="es")
dicc_tokenizadores["en"]=tokenizador_en
dicc_tokenizadores["pt"]=lambda texto:tokenizador(texto,stemmer=dicc_stems["pt"],
                                                  simbolos_excl=simbolos_a_excluir,idioma="pt")


# In[31]:


#Primero definimos constantes, para usarlas con las etiquetes de tanto diccionario.
BEST = 'Best'
BEST_PCT = 'Best_pct'
SVC_ = 'SVC'
LRC = 'Logistic Regression'
RNC = 'Red Neuronal'

#Parámetros para los modelos de SVC, RandomForest y Red Neuronal
param_SVC = {
    "funcion_ML__class_weight": ["balanced", None],
    "funcion_ML__C":[0.01, 0.1, 1],
    "funcion_ML__dual":[True, False],
    "funcion_ML__fit_intercept":[True, False]
}
param_LRC = {
    "funcion_ML__C": [0.0001, 0.001, 0.01, 0.1, 1, 10],
    "funcion_ML__dual": [True, False],
    "funcion_ML__class_weight": [None, "balanced"],
    "funcion_ML__fit_intercept": [True, False]
}
param_RNC = {
    "funcion_ML__alpha":[0.0001, 0.01, 1],
    "funcion_ML__hidden_layer_sizes":[4,8,12,16,20,24],
#    "funcion_ML__learning_rate":["constant", "adaptive"],
    "funcion_ML__activation":["relu", "logistic"],
    "funcion_ML__shuffle":[True, False]
}

dicc_modelos_resultado = {}
for pais in dicc_df_train:   #para los diccionarios de training
    print("Idioma: " + str(pais))
    stop_words = dicc_stopwords[pais]    #necesitamos que se pase una copia, no referencia a fichero
    df_train_pais = dicc_df_train[pais]
    print("\nIdioma: " + pais + "\t\tModelo ML: LinearSVC")
    modelo_SVC = getModeloML_SentimentAnalysis(df_train_pais["comentario"], df_train_pais["resultado"],
                                               stop_words = stop_words, parametros = param_SVC, scoring="f1",
#                                              tokenizador = None, 
                                               tokenizador = dicc_tokenizadores[pais], 
                                               funcionML = LinearSVC(), n_jobs=4, verbose = 2)
    print("\nIdioma: " + pais + "\t\tModelo ML: LogisticRegression (Clasificación)")
    modelo_LRC = getModeloML_SentimentAnalysis(df_train_pais["comentario"], df_train_pais["resultado"],
                                               stop_words = stop_words, parametros = param_LRC, scoring="f1",
#                                              tokenizador = None, 
                                               tokenizador = dicc_tokenizadores[pais], 
                                               funcionML = LogisticRegression(), n_jobs=4, verbose = 2)
    print("\nIdioma: " + pais + "\t\tModelo ML: DecissionTreeClassifier")
    modelo_RNC = getModeloML_SentimentAnalysis(df_train_pais["comentario"], df_train_pais["resultado"],
                                               stop_words = stop_words, parametros = param_RNC, scoring="f1",
#                                              tokenizador = None, 
                                               tokenizador = dicc_tokenizadores[pais],  
                                               funcionML = LinearSVC(), n_jobs=4, verbose = 2)
#    break;    #por ahora en pruebas
    #Identificamos qué modelo tiene mejor scoring para guardarlo en el diccionario. Best score está en <modelo>[1]
    if (modelo_SVC.best_score_ >= modelo_LRC.best_score_) and (modelo_SVC.best_score_ >= modelo_LRC.best_score_):
        mejor_modelo = SVC_
    elif (modelo_LRC.best_score_ >= modelo_SVC.best_score_) and (modelo_LRC.best_score_ >= modelo_RNC.best_score_):
        mejor_modelo = LRC
    else:
        mejor_modelo = RNC
    #En el diccionario interno, va el mejor modelo global y los tres probados con hiperparámetros.
    dic_modelo = {BEST: mejor_modelo, 
                  SVC_: (modelo_SVC.best_score_, modelo_SVC.best_estimator_), 
                  LRC: (modelo_LRC.best_score_, modelo_LRC.best_estimator_), 
                  RNC: (modelo_RNC.best_score_, modelo_RNC.best_estimator_)}
    #Lanzamos el predict() con el modelo ganador contra el dataset de test, y comparamos % de aciertos
    df_test_pais = dicc_df_test[pais]
    y_test_real = df_test_pais["resultado"]
    mejor_modelo_tupla = dic_modelo[dic_modelo[BEST]]   #obtenemos la tupla de score / modelo con mejor puntuación
    y_test_predict = mejor_modelo_tupla[1].predict(df_test_pais["comentario"])  #en la tupla, el modelo es [1]
    y_diff = [abs(yr-y_test_predict[i]) for i, yr in enumerate(y_test_real)]
    y_pct = 100.0 - (sum(y_diff)*100.0/len(y_diff))  #% de resultados correctos (dan 0) vs total
    dic_modelo[BEST_PCT] = y_pct      #Añadimos como complemento el % de acirtos 
    #En el diccionario externo, el país y el primer diccionario, para seleccionar lo que se quiera.
    dicc_modelos_resultado[pais] = {dic_modelo}       #tenía una etiqueta de país de más, ojo
    #Borramos variables para liberar memoria (df_test_pais y y_test_real)
    df_test_pais = None
    y_test_real = None
    y_test_predict = None
#Y ya fuera del bucle, borramos también el diccionario de test y traing por países, es el dataset completo
del dicc_df_train
del dicc_df_test


# Ahora vamos a hacer una ejecución análoga, para ver si pasando el dataset de training completo y con su función de tokenización concreta se mejoran, empeoran o más o menos se mantienen los resultados obtenidos separando por idioma.  
#   
# Como parámetros (tokenizador y stop_words) vamos a usar estrictamente los de **es**, ya que es el idioma mayoritario en el Dataset.

# In[ ]:


#Primero, tomamos el DS completo y volvemos a dividir entre train / test
df_train_coment_completo, df_test_coment_completo = train_test_split(df_comentarios, 
                                                                     test_size=0.2, train_size=0.8)
stop_words = dicc_stopwords["es"]    #necesitamos que se pase una copia, no referencia a fichero

#Parámetros para pruebas
param_SVC={}
param_LRC={}
param_RNC={}
print("\nIdioma: todos\t\tModelo ML: LinearSVC")
modelo_SVC = getModeloML_SentimentAnalysis(df_train_coment_completo["comentario"], df_train_coment_completo["resultado"],
                                           stop_words = stop_words, parametros = param_SVC, scoring="f1",
                                           tokenizador = None, 
#                                          tokenizador = tokenizador_es,
                                           funcionML = LinearSVC(), n_jobs=4, verbose = 1)
print("\nIdioma: todos\t\tModelo ML: LogistincRegression (Clasificación)")
modelo_LRC = getModeloML_SentimentAnalysis(df_train_coment_completo["comentario"], df_train_coment_completo["resultado"],
                                           stop_words = stop_words, parametros = param_LRC, scoring="f1",
                                           tokenizador = None, 
#                                          tokenizador = tokenizador_es, 
                                           funcionML = LogisticRegression(), n_jobs=4, verbose = 1)
print("\nIdioma: todos\t\tModelo ML: DecissionTreeClassifier")
modelo_RNC = getModeloML_SentimentAnalysis(df_train_coment_completo["comentario"], df_train_coment_completo["resultado"],
                                           stop_words = stop_words, parametros = param_RNC, scoring="f1",
                                           tokenizador = None, 
#                                          tokenizador = tokenizador_es,
#                                          funcionML = MLPClassifier(), n_jobs=4, verbose = 1)
                                           funcionML = LinearSVC(), n_jobs=4, verbose = 1)
#Identificamos qué modelo tiene mejor scoring para guardarlo en el diccionario. Best score está en <modelo>[1]
if (modelo_SVC.best_score_ >= modelo_LRC.best_score_) and (modelo_SVC.best_score_ >= modelo_LRC.best_score_):
    mejor_modelo = SVC_
elif (modelo_LRC.best_score_ >= modelo_SVC.best_score_) and (modelo_LRC.best_score_ >= modelo_RNC.best_score_):
    mejor_modelo = LRC
else:
    mejor_modelo = RNC
#En el diccionario interno, va el mejor modelo global y los tres probados con hiperparámetros.
dic_modelo = {BEST: mejor_modelo, 
              SVC_: (modelo_SVC.best_score_, modelo_SVC.best_estimator_),
              LRC: (modelo_LRC.best_score_, modelo_LRC.best_estimator_), 
              RNC: (modelo_RNC.best_score_, modelo_RNC.best_estimator_)}
#Lanzamos el predict() con el modelo ganador contra el dataset de test, y comparamos % de aciertos
y_test_real = df_test_coment_completo["resultado"]
mejor_modelo_tupla = dic_modelo[dic_modelo[BEST]]   #a partir de la etiqueta de mejor modelo (BEST) obtenemos la tupla
y_test_predict = mejor_modelo_tupla[1].predict(df_test_coment_completo["comentario"])  #el best_estimator_ está en [1]
y_diff = [abs(yr-y_test_predict[i]) for i, yr in enumerate(y_test_real)]
y_pct = 100.0 - (sum(y_diff)*100.0/len(y_diff))  #% de resultados correctos (dan 0) vs total
dic_modelo[BEST_PCT] = y_pct      #Añadimos como complemento el % de acirtos 
#En el diccionario externo, el país ("nolang" para distinguir) y el primer diccionario, para seleccionar lo que se quiera.
dicc_modelos_resultado["nolang"] = {dic_modelo}     #tenía una etiqueta de país de más, ojo

#Borramos df y demás auxiliares (ocupan mucha memoria)
del df_train_coment_completo
del df_test_coment_completo


# ## Análisis de resultados
# Tenemos ahora 4 entradas en el diccionario de 1er nivel:
# - **"es"**
# - **"en"**
# - **"pt"**
# - **"nolang"**
#   
# Para cada una de ellas, vamos a ver el mejor modelo y el "Best Score", e imprimir resultados. Adicionalmente, guardaremos el mejor modelo en disco para cada idioma, para ahorrarnos el entrenamiento a futuro.

# In[ ]:


#Bucle para recorrer idiomas, y variables para almacenar el "best of the best" agregado
best_lang = ""           #etiqueta de idioma, para lookup en metadiccionario 1er nivel
best_model_str = ""      #etiqueta de modelo, para lookup en diccionario 2º nivel
best_modelo = None       #el modelo con el mejor rendimiento global, par almacenar e imprimir hiperparámetros
best_score = 0
best_pct_y = 0
modelo_tupla = None       #el modelo en sí, porque a veces a los diccionarios no les gusta la doble index. (paso a paso)

print("Claves de países almacenadas en el diccionario de resultados:")
print(dicc_modelos_resultado.keys())
print("\n\n")

for pais in dicc_modelos_resultado:
    if pais == "nolang":     #por errores al rellenar el diccionario, borrar en código definitivo (para evitar reejecutar)
        dicc_modelos_pais = dicc_modelos_resultado[pais]["pt"]
    else:
        dicc_modelos_pais = dicc_modelos_resultado[pais][pais]
    print("Procesamos el país: " + pais)
    print(type(dicc_modelos_pais))
    print(dicc_modelos_pais.keys())
    best = dicc_modelos_pais[BEST]
    modelo_tupla = dicc_modelos_pais[best]
    pct_y = dicc_modelos_pais[BEST_PCT]
    print("Idioma: " + pais)
    print("\tMejor modelo: " + best)
    print("\tScoring: " + str(modelo_tupla[0]))
    print("\t% acierto en testing: " + str(pct_y) + "\n")    #en las tuplas gardadas, scoring o en el [0], modelo en el [1]
    if modelo_tupla[0] > best_score:
        best_lang = pais
        best_model_str = best           #nombre del modelo en cuestión (cadena de texto)
        best_model = modelo_tupla[1]    #modelo con hiperparámetros para guardar / imprimir
        best_score = modelo_tupla[0]    #scoring del modelo
        best_pct_y = pct_y              #el % de acierto del .predict() para ese modelo
    #En cualquier caso, guardamos en disco el mejor modelo de cada lenguage
    directorio_actual = os.path.abspath(os.curdir)
    joblib.dump(modelo_tupla[1], os.path.join(directorio_actual, "mejor_modelo_SentAnalysis_" + pais + ".pkl"))
#Ahora que hemos acabado, mostramos los parámetros en detalle del "best of the best"
print("\n\n\nY el mejor modelo de todos tiene estos datos:")
print("Lenguaje: " + best_lang)
print("\tModelo: " + best_model_str + "\tScoring: " + str(best_score) + "\n% acierto: " + str(best_pct_y))
print("\tParámetros del modelo:\n")
print(best_model)

