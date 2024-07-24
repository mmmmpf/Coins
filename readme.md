# Israeli Coin Recognition Using Deep Learning

## Overview

This project explores the efficacy of three object detection deep learning models: VGG16, ResNet50, and YOLOv5. The objective is to develop an intelligent coin recognition system capable of identifying denominations of Israeli Shekel coins (1, 2, 5, 10 ILS) within images. By leveraging the power of deep learning, particularly convolutional neural networks (CNNs), the system is trained to accurately classify coins based on labeled data with bounding boxes indicating the location and value of each coin.

This project is divided into three parts:
1. Introduction and technical details
2. Code
3. Comprehensive comparison of the performance of the three models for the coin recognition task. This comparison will consider factors like accuracy, speed, and computational efficiency to determine the most suitable model for real-world applications.

## Key Features

- **Data Annotation and Extraction**: A custom function parses JSON files containing annotations for coin images. These annotations include bounding boxes and labels for each coin, which are then combined with the corresponding image paths.
- **Image Preprocessing**: Functionality to load and preprocess images by extracting regions of interest (ROIs) based on bounding box coordinates, resizing them to the required input dimensions for the neural network, and normalizing pixel values.
- **Deep Learning Models**: The project investigates multiple deep learning models for coin recognition, including:
  - **VGG16**: A well-established pre-trained CNN architecture that will be fine-tuned for coin classification.
  - **ResNet50**: Another powerful CNN architecture known for its depth and performance, also fine-tuned for coin recognition.
  - **YOLOv5**: A state-of-the-art real-time object detection model trained to identify and localize coins within images.
- **Training with Early Stopping**: All models incorporate early stopping to prevent overfitting. This technique monitors validation loss and halts training when the loss stops improving.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- PyTorch
- OpenCV
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Israeli-Coin-Recognition.git
   cd Israeli-Coin-Recognition
}

## Usage

1. Data Preparation:
* Ensure your dataset is properly annotated and organized.
* Update the paths in the notebook accordingly.

2. Training:
* Load and preprocess the images.
* Choose the desired model architecture (VGG16, ResNet50, or YOLOv5).
* Train the model with the training dataset.
* Evaluate the model on the validation dataset using early stopping to prevent overfitting.

3. Inference:
* Use the trained model to perform coin recognition on new images.
* Visualize the detected coins with bounding boxes and predicted labels.

## Results

The project includes a comprehensive comparison of the performance of the three models for the coin recognition task. This comparison considers factors like accuracy, speed, and computational efficiency to determine the most suitable model for real-world applications.
