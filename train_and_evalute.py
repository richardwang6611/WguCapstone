import tensorflow as tf
import os
import matplotlib.pyplot as plt
from model import build_model
from fileReader import load_data
from sklearn.utils import class_weight
import numpy as np

class PrintValidationMetrics(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}: Validation Loss: {logs['val_loss']:.4f}, Validation Accuracy: {logs['val_accuracy']:.4f}")

def main():

    # Load the dataset
    dataset_dir = 'lung_image_sets'  # Replace with the actual path to your dataset
    train_dataset, val_dataset, class_names = load_data(dataset_dir)

    # Check class distribution
    class_counts = {class_name: 0 for class_name in class_names}
    all_labels = []
    for _, labels in train_dataset:
        all_labels.extend(labels.numpy())
        for label in labels.numpy():
            class_counts[class_names[label]] += 1
    print("Class distribution in training set:", class_counts)

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}
    print("Class weights:", class_weights)

    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    model = build_model(input_shape=(224, 224, 3), num_classes=len(class_names))

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=10, callbacks=[PrintValidationMetrics()], class_weight=class_weights)

    loss, accuracy = model.evaluate(val_dataset)
    print(f"Final Validation Loss: {loss:.4f}")
    print(f"Final Validation Accuracy: {accuracy * 100:.2f}%")

    os.makedirs('models', exist_ok=True)

    model.save('models/model.h5')

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()