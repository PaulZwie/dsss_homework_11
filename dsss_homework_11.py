import os
import sys

import numpy as np
import tensorflow as tf
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from tensorflow.keras import models, layers, datasets


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText('\n\n Drop Image Here \n\n')

        self.setStyleSheet('''
            QLabel{
                border: 4px dashed #aaa
            }
        ''')

    def setPixmap(self, image):
        super().setPixmap(image)


class GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(400, 400)
        self.setAcceptDrops(True)

        mainLayout = QVBoxLayout()

        self.photoViewer = ImageLabel()
        mainLayout.addWidget(self.photoViewer)

        self.resultLabel = QLabel()
        self.resultLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mainLayout.addWidget(self.resultLabel)

        self.setLayout(mainLayout)

        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path='cnn_model.tflite')
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Class names for Fashion MNIST
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            filepath = event.mimeData().urls()[0].toLocalFile()
            self.set_image(filepath)
            self.run_inference(filepath)
            event.accept()
        else:
            event.ignore()

    def set_image(self, filepath):
        pixmap = QPixmap(filepath)
        scaled_pixmap = pixmap.scaled(self.photoViewer.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
        self.photoViewer.setPixmap(scaled_pixmap)

    def run_inference(self, filepath):
        # Load and preprocess the image
        image = QImage(filepath)
        image = image.convertToFormat(QImage.Format.Format_Grayscale8)
        image = image.scaled(28, 28, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)

        # Convert QImage to numpy array
        ptr = image.bits()
        ptr.setsize(image.width() * image.height())
        image_array = np.array(ptr).reshape((28, 28, 1)).astype('float32') / 255.0

        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], [image_array])
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        predicted_class = np.argmax(output_data)

        # Update result label with class name
        self.resultLabel.setText(f'Predicted Class: {self.class_names[predicted_class]}')


def create_model(filepath):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))


    return model

def convert_to_tflite(model, model_path, tflite_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {tflite_path}")


def main():
    # GUI Implementation
    #app = QApplication(sys.argv)

    #window = GUI()
    #window.show()


    # Model Implementation
    # MNIST dataset
    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255

    # paths
    cwd = os.getcwd()

    model_path = os.path.join(cwd, 'cnn_model.h5')
    tflite_path = os.path.join(cwd, 'cnn_model.tflite')

    #if not os.path.exists(tflite_path):
    model = create_model(model_path)
    model = train_model(model, x_train, y_train, x_test, y_test)
    convert_to_tflite(model, model_path, tflite_path)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
