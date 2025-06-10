import os
from tensorflow import keras
import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
# Step 1: Just store the path (as a string)
model_path = r'C:\Users\ACER\OneDrive\Documents\AIML TASKS\PROJECT\plant_model.keras'
model = keras.models.load_model(model_path)



# REPLACE WITH YOUR ACTUAL CLASS NAMES
class_names = [
    'Pepper__bell___Bacterial_spot',
    'Potato___healthy',
    'Tomato_Leaf_Mold',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato_Bacterial_spot',
    'Tomato_Septoria_leaf_spot',
    'Tomato_healthy',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato_Early_blight',
    'Tomato__Target_Spot',
    'Pepper__bell___healthy',
    'Potato___Late_blight',
    'Tomato_Late_blight',
    'Potato___Early_blight',
    'Tomato__Tomato_mosaic_virus'
]
  

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    prediction = model.predict(img_array)
    return class_names[np.argmax(prediction[0])]

st.title('Plant Disease Detection')
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(uploaded_file)
    prediction = predict_disease(temp_path)
    st.success(f"Predicted Disease: {prediction}")
    os.remove(temp_path)  # Clean up