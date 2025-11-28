
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the saved model (assuming it's in the same path as during training)
model = tf.keras.models.load_model('/content/drive/MyDrive/garbage_classifier_model.h5')

# Get image dimensions and class mapping from previous execution
img_height = 128
img_width = 128
class_indices = {
    'cardboard': 0, 'glass': 1, 'metal': 2,
    'paper': 3, 'plastic': 4, 'trash': 5
}
idx_to_class = {v: k for k, v in class_indices.items()}

def predict_garbage_class(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image or NumPy array.")

    image = image.resize((img_width, img_height))
    image_array = np.asarray(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    predicted_class_idx = np.argmax(predictions)
    return idx_to_class[predicted_class_idx]

st.set_page_config(page_title="Garbage Classification App")
st.title('Garbage Classification App')
st.write('Upload an image of garbage (cardboard, glass, metal, paper, plastic, or trash) and the model will predict its class.')

# Add a file uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Call the prediction function
    predicted_class = predict_garbage_class(image)

    # Display the predicted garbage class
    st.success(f"Predicted Class: {predicted_class}")
else:
    st.info("Please upload an image to get a prediction.")
