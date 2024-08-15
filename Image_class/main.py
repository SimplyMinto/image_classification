import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models


(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255, testing_images / 255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])


plt.show()

# Data slicing
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

model = models.load_model('image_classifier.keras')

# Load the image
img = cv.imread("E:\Image_class\car3.jpeg")

# Convert the image from BGR to RGB
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Display the image
plt.imshow(img)
plt.axis('off')  # Turn off axis numbers and ticks
plt.show()

# Preprocess the image: scale pixel values and add batch dimension
img = np.array([img]) / 255.0

# Make prediction
prediction = model.predict(img)
index = np.argmax(prediction)

# Assuming you have a list of class names corresponding to your model's output classes
print(f'Prediction is: {class_names[index]}')
