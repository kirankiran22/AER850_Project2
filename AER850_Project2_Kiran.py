#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:17:51 2024
@author: kiranpatel

Semester (Term, Year): Fall 2024
Course Code: AER850
Section: 01
Course Title: Intro to Machine Learning 
Course Instructor: Dr. Reza Faieghi

Title: AER850 - Project 2: Deep Convolution Neural Networks
By: Kiran Patel
Student#: XXXXXX568

"""
#Purpose - To Design a CNN to correctly classify Images as either 
#           Cracks, missing screw heads, or paint-off


#Import Packages for the Project 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np 
from keras import layers
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------


# STEP 1: Data Processing
#-----------------------------------------------------------------------------

#define the input image shape which is required to be (500,500, 3) which 
#is the desired width, height and channel of the image for model training.

img_height = 500
img_width = 500
channels = 3
input_shape = (img_height, img_width, channels)


#Initializing File pathways to access image dataset
# Establish the train and validation data directory (use relative paths). 
# The data is split into 3 folders - Train, Validation and Test which contain 
# 1942, 431 and 539 images respectively

train_path = './Data/train'
val_path = './Data/valid'
test_path ='./Data/test'

#Print Statement Verifiers for myself to ensure Code can access images from 
#Data folder and that Tensor Flow installed with the version#
print("\nImporting Data")
print("\n")
print(tf.__version__)


# Data augmentation for the training image set
# such as re-scaling, shear range and zoom range by using
# packages such as Keras’ image preprocessing methods to format the dataset
train_data_augment = ImageDataGenerator(
    rescale = 1.0/255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

# Rescale validation set images
# only apply re-scaling for validation
val_data_augment = ImageDataGenerator(
    rescale = 1.0 / 255
    )

# Train and validation data generators
# Create the train and validation generator using Keras’s built-in imagedatasetfromdirectory
#function which takes in the data directory, image target size, batch size (32), and
# class mode (categorical

train_generator = train_data_augment.flow_from_directory(
    train_path,
    target_size = (img_height, img_width),
    batch_size = 32,
    class_mode = 'categorical',
    color_mode = 'rgb',
    shuffle = True
)

val_generator = val_data_augment.flow_from_directory(
    val_path,
    target_size = (img_height, img_width),
    batch_size = 32,
    class_mode = 'categorical',
    color_mode = 'rgb',
    shuffle = True
)

# Verify the data generators are working
print("Train classes: ", train_generator.class_indices)
print("Validation classes: ", val_generator.class_indices)
#-----------------------------------------------------------------------------


# STEP 2: Neural Network Architecture Design
#-----------------------------------------------------------------------------

# CNN Model
model = Sequential([
    layers.Conv2D(128, (5, 5), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(128, (5, 5), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.3),
    
    layers.Conv2D(128, (5, 5), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.3),
    
    layers.Conv2D(128, (5, 5), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.4),
    
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(
    optimizer = optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[early_stopping]
)

# Display model architecture
model.summary()

# Save the trained model
model.save("model2_kiran.h5")

# Plot training and validation accuracy/loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# Accuracy plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy', color='red')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss', color='red')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


