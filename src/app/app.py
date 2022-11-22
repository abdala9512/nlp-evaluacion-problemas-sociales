import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from ast import literal_eval
from nlp_utils import process_text, basic_cleaning

# Data preprocessing
MAX_LEN = 7


data = pd.read_csv("./data/app_data.csv", sep=";")
data["text_tokenized_list"] = data["text_tokenized_list"].apply(lambda x: " ".join(literal_eval(x)))
label_binarizer = LabelBinarizer()
y = label_binarizer.fit_transform(data["target"])

tf_tokenizer = Tokenizer()
fit_text = [" ".join(data["text_tokenized_list"])]
tf_tokenizer.fit_on_texts(fit_text)


nn_model = tf.keras.models.load_model('./src/app/LSTM_prod/')

def text_to_index(text):
    return [ tf_tokenizer.word_index[word] for word in text.split(" ")]

st.title("Analisis de problemas sociales - NLP")
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Javeriana.svg/1200px-Javeriana.svg.png",  width=100)

st.markdown("""
## Integrantes

* Omar Balcero
* Miguel Arquez
* Leonardo Espitia
* Laura Peñaranda


## Inferencia  Modelo NLP
""")
input_text = st.text_input("Texto de entrada", "Prueba de texto ")


def make_prediction(
    text: str, model, 
    prediction_threshold: float = 0.35
    ) -> str:
    """Make prediction for Selected neural network
    """
    tokenized = " ".join([
        word for word in process_text(text.lower()).split(" ")
        if word in list(tf_tokenizer.word_index.keys())
    ])
    
    vector_ = tf.keras.preprocessing.sequence.pad_sequences( 
        np.array(text_to_index(tokenized)).reshape(1,-1),  maxlen=MAX_LEN
    )
    
    probabilities = np.array(model.predict(vector_))
    predictions = {
        label_binarizer.classes_[i]: probabilities[0][i]
        for i in range(3)
    }
    
    if any([prob > prediction_threshold for prob in list(predictions.values())]):
        return max(predictions, key=predictions.get)
    return "Predicciones no superan el umbral para seleccionar almenos una categoria"

st.markdown(f"""
La prediccion del modelo es: **{make_prediction(input_text, model=nn_model)}**
""" )