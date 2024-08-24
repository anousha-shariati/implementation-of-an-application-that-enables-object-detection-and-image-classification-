import sys,ssl
import cv2 ,os
import numpy as np
import PIL.Image
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont, QColor
import tensorflow as tf
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog

# Initialize global variables
frame = None
image_label = None  # Label for the selected image
video_label = None  # Label for the video capture
cap = None
ret = False

# SSL context for unverified HTTPS requests
ssl._create_default_https_context = ssl._create_unverified_context

# Load a pre-trained ResNet50 model from Keras
model = tf.keras.applications.ResNet50(weights='imagenet')

def preprocess_image(image_path):
    " Preprocess the image for classification. "
    try:
        image = PIL.Image.open(image_path)
        image = image.resize((224, 224))
        image = np.array(image)
        image = tf.keras.applications.resnet50.preprocess_input(image)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        print(f"Error opening image: {e}")
        raise

def classify_image(image_path):
    """ Classify the image using the pre-trained model and return class label with confidence. """
    try:
        image = preprocess_image(image_path)
        predictions = model.predict(image)
        decoded_predictions = tf.keras.applications.resnet50.decode_predictions(predictions, top=1)[0][0]
        class_name, confidence = decoded_predictions[1], decoded_predictions[2]
        return class_name, confidence
    except Exception as e:
        print(f"Error classifying image: {e}")
        return "Classification Error", 0


def frame_update():
    """ Update the video frame in the GUI. """
    global video_label, cap, ret, frame
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(convert_to_Qt_format).scaled(640, 480, Qt.KeepAspectRatio)
        video_label.setPixmap(pixmap)


def capture_and_classify():
    """ Capture image from webcam, classify, and display. """
    global cap, frame
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)  # Ensure the camera is opened here
    ret, frame = cap.read()
    if ret:
        cv2.imwrite('captured_photo.jpg', frame)
        class_name, confidence = classify_image('captured_photo.jpg')
        display_classified_image('captured_photo.jpg', class_name, confidence)
    else:
        print("Error: Could not capture an image.")
    cap.release()
    cap = None  # Reset cap to ensure it's clean for next use



def display_classified_image(image_path, class_name, confidence):
    """ Display the image with aspect ratio maintained and classification with confidence on the UI. """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image from {image_path}")
        return

    # Define the target display size
    target_width, target_height = 640, 480

    # Calculate the aspect ratio of the image
    height, width, channels = image.shape
    scaling_factor = min(target_width / width, target_height / height)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)

    # Resize the image to maintain aspect ratio
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a black background to paste the resized image onto
    background = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Calculate centering position
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # Place the resized image onto the center of the background
    background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image

    # Convert the background to QImage for display
    rgb_image = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    qt_image = QImage(rgb_image.data, target_width, target_height, rgb_image.strides[0], QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qt_image)

    # Prepare to draw the classification result and confidence on the image
    painter = QPainter(pixmap)
    painter.setPen(QColor(255, 255, 255))
    painter.setFont(QFont('Arial', 20))

    # Display text format
    display_text = f"{class_name} - {confidence*100:.2f}"
    text_rect = QtCore.QRect(0, pixmap.height() - 50, pixmap.width(), 50)
    painter.drawText(text_rect, Qt.AlignCenter, display_text)
    painter.end()

    # Set the pixmap to the QLabel to display on the UI
    image_label.setPixmap(pixmap)




def open_file():
    """ Open a file dialog to select an image, display it, and classify it. """
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    file_name, _ = QFileDialog.getOpenFileName(None, "Open Image File", "", "Image Files (*.png *.jpg *.bmp *.jpeg)", options=options)
    if file_name:
        classify_and_display(file_name)


def classify_and_display(image_path):
    """ Display the selected image and classify it, showing the class label and confidence on the image. """
    global image_label
    try:
        class_name, confidence = classify_image(image_path)
        display_classified_image(image_path, class_name, confidence)
    except Exception as e:
        print(f"Error displaying image: {e}")




if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = QWidget()
    MainWindow.setObjectName("MainWindow")
    MainWindow.resize(1480, 1000)
    MainWindow.setStyleSheet("background-color: rgb(170, 170, 255);")

    # Setup labels
    video_label = QLabel(MainWindow)
    video_label.resize(640, 480)
    video_label.move(790, 50)
    
    image_label = QLabel(MainWindow)
    image_label.resize(640, 480)
    image_label.move(100, 50)  # Adjust the position to not overlap with video_label

    # Setup buttons
    video_btn = QPushButton("Capture Video", MainWindow)
    video_btn.setGeometry(QtCore.QRect(1025, 580, 150, 50))
    video_btn.setStyleSheet("background-color: rgb(221, 222, 255);")
    video_btn.clicked.connect(capture_and_classify)

    open_file_btn = QPushButton("Open File", MainWindow)
    open_file_btn.setGeometry(QtCore.QRect(1025, 640, 150, 50))
    open_file_btn.setStyleSheet("background-color: rgb(221, 222, 255);")
    open_file_btn.clicked.connect(open_file)

    # Timer to update video frames
    timer = QTimer(MainWindow)
    timer.start(1000//30)  # Update at about 30fps
    timer.timeout.connect(frame_update)

    MainWindow.show()
    sys.exit(app.exec_()) 