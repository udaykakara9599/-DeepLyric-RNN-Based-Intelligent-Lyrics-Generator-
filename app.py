import streamlit as st
import numpy as np
import pickle
from keras.models import load_model

# Load model
model = load_model("Lyrics_Generator.h5")

# Load mappings
with open("mapping.pkl", "rb") as f:
    mapping = pickle.load(f)

with open("reverse_mapping.pkl", "rb") as f:
    reverse_mapping = pickle.load(f)

L_symb = len(mapping)

# Title
st.title("🎵 Lyrics Generator using RNN (LSTM)")
st.write("Generate song lyrics using your trained model")

# User input
starter = st.text_input("Enter starting text:", "love is")
length = st.slider("Select length of lyrics:", 50, 500, 200)

# Generator function
def Lyrics_Generator(starter, Ch_count):
    generated = starter
    
    for i in range(Ch_count):
        try:
            seed = [mapping[char] for char in starter]
        except:
            return "⚠️ Please use known characters only"

        x_pred = np.reshape(seed, (1, len(seed), 1))
        x_pred = x_pred / float(L_symb)

        prediction = model.predict(x_pred, verbose=0)[0]

        prediction = np.asarray(prediction).astype('float64')
        prediction = np.log(prediction + 1e-8)
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)

        index = np.argmax(prediction)
        next_char = reverse_mapping[index]

        generated += next_char
        starter = starter[1:] + next_char

    return generated

# Button
if st.button("Generate Lyrics"):
    result = Lyrics_Generator(starter.lower(), length)
    
    st.subheader("🎶 Generated Lyrics:")
    st.write(result)