import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

MODEL_PATH = os.path.join('model', 'plastic_rice_model.h5')
IMG_SIZE = (224, 224)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title('Plastic Rice Detection')
st.write('Upload an image of cooked or raw rice to detect if it is real or plastic.')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    label = 'Plastic Rice' if prediction > 0.5 else 'Real Rice'
    st.write(f'**Prediction:** {label} ({prediction:.2f})') 