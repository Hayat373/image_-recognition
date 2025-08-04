import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained MobileNet model from TF Hub
mobilenet_layer = hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5")

# Load and preprocess an image
img_path = 'cat.jpg'  # Replace with your image path
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

# Make a prediction using the layer directly
predictions = mobilenet_layer(img_array)
predicted_class = np.argmax(predictions[0])

# Load labels
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
labels = np.array(open(labels_path).read().splitlines())
print(f"Predicted: {labels[predicted_class]}")