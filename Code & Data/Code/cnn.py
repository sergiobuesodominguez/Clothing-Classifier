import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical


train_data = np.load('Data/fashion_train.npy')  
train_images = train_data[:, :-1]  
train_labels = train_data[:, -1]  


train_images = train_images.reshape(-1, 28, 28).astype('float32') / 255.0
train_images = np.expand_dims(train_images, axis=-1)

num_classes = len(np.unique(train_labels))
train_labels = to_categorical(train_labels, num_classes)

test_data = np.load('Data/fashion_test.npy')  
test_images = test_data[:, :-1]  
test_labels = test_data[:, -1]  

test_images = test_images.reshape(-1, 28, 28).astype('float32') / 255.0

# Add a channel dimension for grayscale
test_images = np.expand_dims(test_images, axis=-1)

# Convert testing labels to categorical
test_labels = to_categorical(test_labels, num_classes)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_accuracy:.4f}')
