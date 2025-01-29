import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from fileReader import load_data

def main():
    st.title("Lung Cancer Detection")

    model = tf.keras.models.load_model("models/model.h5")

    dataset_dir = 'lung_image_sets'
    _, _, class_names = load_data(dataset_dir)

    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Go to", ["Upload Image"])

    if option == "Upload Image":
        st.subheader("Select the Actual Class and Upload an Image")
        actual_class = st.selectbox("Select the actual class of the uploaded image:", class_names)

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None and actual_class:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            st.write(f"Prediction: {prediction}")
            predicted_class = np.argmax(prediction, axis=1)[0]
            st.write(f"Predicted Class Index: {predicted_class}")

            st.subheader("Prediction")
            predicted_class_name = class_names[predicted_class]

            if actual_class == predicted_class_name:
                st.markdown(f"<h3 style='color: green;'>Actual: {actual_class} | Predicted: {predicted_class_name}</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color: red;'>Actual: {actual_class} | Predicted: {predicted_class_name}</h3>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()