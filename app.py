import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

model = load_model('next_word_lstm.keras')

with open('tokenizer.pickle','rb') as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(model,tokenizer,text,max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
   # print('Token_List',token_list)
    if(len(token_list)>=max_sequence_len):
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
    #print('After pad sequences',token_list)
    predicted = model.predict(token_list,verbose=0)
    #print(predicted)
    predicted_word_index = np.argmax(predicted,axis=1)
    for word,index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

st.title('Next Word Prediction using LSTM')
input_text = st.text_input("Enter sequence of words ",'To be or not to be')
if st.button('Predict Next Word'):
    max_sequence_len = model.input_shape[1]+1
    next_word = predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f"Next word Predicted : {next_word}")


