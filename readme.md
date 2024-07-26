# Israeli Coin Recognition Using Deep Learning

## Overview

In this project, I explored the efficacy of three object detection deep learning models; VGG16, ResNet50 and YOLOv5. The objective was to develop an intelligent coin recognition system capable of identifying denominations of Israeli Shekel coins (1, 2, 5, 10 ILS) within images. By leveraging the power of deep learning, particularly convolutional neural networks (CNNs), the system is trained to accurately classify coins based on labeled data with bounding boxes indicating the location and value of each coin.

![labeled](https://github.com/user-attachments/assets/b1c6ec9e-d16e-440d-8529-880395f03b72)


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

### Results for VGG16 and ResNet50 models
We can assess the models' ability to recognize and classify different coin denominations by using confusion metrics and other performance parameters. The training process is visualized with plots that illustrate accuracy improvements over epochs, and early stopping is employed to retain the best model. 
**Although both models achieved only mediocre accuracy on the validation set, their performance on the test set was notably poor.**

Next, we will shift focus to a different object recognition model—YOLOv5—and evaluate its capability to recognize Israeli Shekel coins.

## Results for YOLOv5 model
![image](https://github.com/user-attachments/assets/4a7f2ee0-5d3e-4248-a319-21b011fcb2d8)
F1 score for YOLOv5 model

![image](https://github.com/user-attachments/assets/80c08455-60e3-4e9c-892a-82bced0b6227)
PR score for YOLOv5 model

# Comprehensive Model Comparison:
This section is the heart of the project analysis. It presents a comparison of the performance of VGG16, ResNet50, and YOLOv5 on the coin recognition task. Factors considered in this comparison will include:

* Accuracy: The primary metric of success, measured by the percentage of correctly classified coins across all denominations. This section will compare the accuracy achieved by each model on the test set.
* Speed: Time taken for each model to perform inference on an image. This is crucial for real-world applications where a balance between accuracy and speed is essential.
* Computational Efficiency: The amount of computational resources (e.g., memory, processing power) required by each model for training and inference. This can be estimated by the number of parameters in the model - fewer parameters generally require less computation for both training and inference. This will be particularly relevant if the system is intended to be deployed on resource-constrained devices.

This comprehensive approach not only showcases the development of a coin recognition system but also provides valuable insights into the strengths and weaknesses of different deep learning models for this specific task.

While YOLOv5 has proven to be effective in this project, it's valuable to consider alternative object detection models like ResNet50 and VGG16. Here's a comparative analysis:

Model            | VGG16 | ResNet50 | YOLOv5
 ----------------|-------|----------|-------
 Advantages |Well-established architecture, widely used for transfer learning | Strong performance on image classification tasks | Real-time inference capability
 Disadvantages |May not be as efficient as YOLOv5 for object detection, especially with smaller datasets | Can be computationally expensive for real-time object |May require more training data for complex object detection tasks
 Time [Sec] |622 |646 |1206  
 Accuracy |Low | Low | 0.81
 No. #Parameters |27e+6|74e+6 | 7e+6
 Epochs |44|50|100


The choice between these models often hinges on the specific project requirements. If real-time inference is crucial, YOLOv5 excels. For image classification tasks with a large dataset, ResNet50 could be a suitable option. VGG16 is a solid choice for transfer learning scenarios.

# Deploying the Model:
I utilized Vertex AI to train and deploy a model for real-world coin detection, integrating it into a web application and mobile devices. Vertex AI facilitated both the training and deployment processes, resulting in satisfactory results. This platform demonstrates the model's performance and ensuring its effective operation in practical scenarios.

![1](https://github.com/user-attachments/assets/f86a4523-0836-4f0d-b293-c49a6376a6d8)

![stat](https://github.com/user-attachments/assets/0abe4a7a-6211-4d85-9667-6928b0cb7f6e)
Vertex AI model prediction accuracy


# Conclusion

This project successfully demonstrated the effectiveness of YOLOv5 for object detection. The trained model achieved promising results in identifying coins within images. The comparison with ResNet50 and VGG16 provided valuable insights into the strengths and trade-offs associated with different models.



