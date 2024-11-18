#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 19:41:13 2024

Semester (Term, Year): Fall 2024
Course Code: AER850
Section: 01
Course Title: Intro to Machine Learning 
Course Instructor: Dr. Reza Faieghi

Title: AER850 - Project 2: Deep Convolution Neural Networks - Step 5
By: Kiran Patel
Student#: XXXXXX568
"""


#STEP 5: MODEL Testing
#-----------------------------------------------------------------------------

#Importing Required Packages 
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Load the Model from first project file of the saved CNN
Model = load_model("model2_kiran.h5")

# Define Input Image Shape
Img_width = 500
Img_height = 500

# Define Class Labels to be Identified
class_labels = ['Crack', 'Missing-head', 'Paint-off']

# Data Preprocessing
def preprocess(image_path, target_size=(Img_width, Img_height)):
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image)
    image_array = image_array / 255
    image_array = np.expand_dims(image_array, 0)
    return image_array

# Data Prediction
def predict(image_array, model):
    predictions = model.predict(image_array)
    return predictions

# Data Display
def display(image_path, predictions, true_label, class_labels):
    predicted_label = class_labels[np.argmax(predictions)]
    fig, ax = plt.subplots(figsize=(6, 6))
    img = plt.imread(image_path)
    plt.imshow(img)
    ax.axis('off')
    
    # Add title with True and Predicted Labels
    plt.title(
        f"True Crack Classification Label: {true_label}\n"
        f"Predicted Crack Classification Label: {predicted_label}",
        fontsize=12,
        fontweight='bold' #bold font for title
    )

    # Display predictions in the bottom left corner with light blue background
    sorted_labels = sorted(zip(class_labels, predictions[0]))
    text_lines = "\n".join(
        [f"{label}: {percentage * 100:.2f}%" for label, percentage in sorted_labels]
    )
    ax.text(
        0.02, 0.02,  # Bottom-left corner
        f"STEP 5: MODEL Testing\n\n{text_lines}",
        transform=ax.transAxes,
        fontsize=10,
        color='black',
        bbox=dict(facecolor="lightblue", edgecolor="none", alpha=0.8),
        verticalalignment='bottom',
        horizontalalignment='left'
    )
    
    plt.tight_layout()
    plt.show()

# Test Images File path
test_images = [
    ("./Data/test/crack/test_crack.jpg", "Crack"),
    ("./Data/test/missing-head/test_missinghead.jpg", "Missing Head"),
    ("./Data/test/paint-off/test_paintoff.jpg", "Paint-Off")
]

for image_path, true_label in test_images:
    img_preprocess = preprocess(image_path)
    predictions = predict(img_preprocess, Model)
    display(image_path, predictions, true_label, class_labels)
