import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from fileReader import load_data
import os

def main():
    st.title("Lung Cancer Detection")

    model = tf.keras.models.load_model("models/model.h5")

    dataset_dir = 'lung_image_sets'
    _, _, class_names = load_data(dataset_dir)

    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Go to", ["Upload Image"])

    if option == "Upload Image":
        st.subheader("Upload an Image")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Extract the class label from the file name
            file_name = os.path.basename(uploaded_file.name)
            actual_class = file_name.split('_')[0]  # Assuming the class label is the first part of the file name

            # Normalize the actual class to match the format of class names
            actual_class = actual_class.lower().replace(" ", "_")

            # Map the actual class to the correct format if necessary
            class_mapping = {
                'lungn': 'lung_n',
                'lungaca': 'lung_aca',
                'lungscc': 'lung_scc'
            }
            actual_class = class_mapping.get(actual_class, actual_class)

            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            st.write(f"Prediction: {prediction}")
            predicted_class = np.argmax(prediction, axis=1)[0]
            st.write(f"Predicted Class Index: {predicted_class}")

            st.subheader("Prediction")
            predicted_class_name = class_names[predicted_class]

            # Display the actual class and predicted class
            st.markdown(f"<h3>Actual: {actual_class} | Predicted: {predicted_class_name}</h3>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()