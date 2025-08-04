import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape images to include a channel dimension (required for CNN)
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Step 2: Build the CNN model
model = models.Sequential([
    # First convolutional layer: 32 filters, 3x3 kernel, ReLU activation
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),  # Pooling to reduce size
    # Second convolutional layer: 64 filters
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # Flatten the output for dense layers
    layers.Flatten(),
    # Dense layer with 64 neurons
    layers.Dense(64, activation='relu'),
    # Output layer: 10 neurons (one for each digit), softmax for classification
    layers.Dense(10, activation='softmax')
])

# Step 3: Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 4: Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

# Step 5: Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# Step 6: Visualize some predictions
predictions = model.predict(test_images[:5])
for i in range(5):
    plt.figure(figsize=(2, 2))
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[i])}, True: {test_labels[i]}")
    plt.axis('off')
    plt.show()