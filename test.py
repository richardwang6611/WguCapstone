import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from fileReader import load_data

def main():
    model = tf.keras.models.load_model("models/model.h5")

    dataset_dir = 'lung_image_sets'  # Replace with the actual path to your dataset
    _, test_dataset, class_names = load_data(dataset_dir)

    true_labels = []
    predicted_probs = []
    for images, labels in test_dataset:
        predictions = model.predict(images)
        predicted_classes = np.argmax(predictions, axis=1)
        true_labels.extend(labels.numpy())
        predicted_probs.extend(predicted_classes)

    # Generates the confusion matrix
    cm = confusion_matrix(true_labels, predicted_probs)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    # Plot and save the confusion matrix as an image
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    main()