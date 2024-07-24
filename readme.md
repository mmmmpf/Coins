{\rtf1\ansi\ansicpg1252\cocoartf2580
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Israeli Coin Recognition Using Deep Learning\
\
## Overview\
\
This project explores the efficacy of three object detection deep learning models: VGG16, ResNet50, and YOLOv5. The objective is to develop an intelligent coin recognition system capable of identifying denominations of Israeli Shekel coins (1, 2, 5, 10 ILS) within images. By leveraging the power of deep learning, particularly convolutional neural networks (CNNs), the system is trained to accurately classify coins based on labeled data with bounding boxes indicating the location and value of each coin.\
\
This project is divided into three parts:\
1. Introduction and technical details\
2. Code\
3. Comprehensive comparison of the performance of the three models for the coin recognition task. This comparison will consider factors like accuracy, speed, and computational efficiency to determine the most suitable model for real-world applications.\
\
## Key Features\
\
- **Data Annotation and Extraction**: A custom function parses JSON files containing annotations for coin images. These annotations include bounding boxes and labels for each coin, which are then combined with the corresponding image paths.\
- **Image Preprocessing**: Functionality to load and preprocess images by extracting regions of interest (ROIs) based on bounding box coordinates, resizing them to the required input dimensions for the neural network, and normalizing pixel values.\
- **Deep Learning Models**: The project investigates multiple deep learning models for coin recognition, including:\
  - **VGG16**: A well-established pre-trained CNN architecture that will be fine-tuned for coin classification.\
  - **ResNet50**: Another powerful CNN architecture known for its depth and performance, also fine-tuned for coin recognition.\
  - **YOLOv5**: A state-of-the-art real-time object detection model trained to identify and localize coins within images.\
- **Training with Early Stopping**: All models incorporate early stopping to prevent overfitting. This technique monitors validation loss and halts training when the loss stops improving.\
\
## Requirements\
\
- Python 3.x\
- TensorFlow\
- Keras\
- PyTorch\
- OpenCV\
- NumPy\
- Matplotlib\
\
## Installation\
\
1. Clone the repository:\
   ```bash\
   git clone https://github.com/yourusername/Israeli-Coin-Recognition.git\
   cd Israeli-Coin-Recognition\
}