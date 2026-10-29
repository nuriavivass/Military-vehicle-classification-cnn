# Military-vehicle-classification-cnn
CNN-based military vehicle classification using satellite imagery and deep learning.


# Military Vehicle Classification in Satellite Images Using CNNs

A deep learning project focused on classifying military vehicles from satellite imagery using a **Convolutional Neural Network (CNN)** built and trained from scratch.

The project explores the challenges of image classification with limited training data, particularly **data scarcity, overfitting, and model generalization**.

## Project Overview

The objective is to develop a CNN capable of classifying military vehicles into **11 different classes**, including:

- **Aircraft (6 classes):** F-35, MiG-29, Eurofighter, A400M, and others.
- **Ground vehicles (5 classes):** 3 tanks, 1 artillery vehicle, and 1 armored personnel carrier.

The model processes RGB images resized to **120 × 120 pixels** and predicts the corresponding vehicle class.

## Dataset & Preprocessing

A custom dataset was created from scratch using satellite imagery collected from Google Earth and online resources such as Virtual Globetrotting.

Due to the limited availability of satellite images of military ground vehicles, some synthetic images from simulation environments such as Arma 3 were also incorporated.

The original dataset contained approximately **20 images per class**.

To address data scarcity, several data augmentation techniques were applied:

- Random rotations
- Horizontal and vertical flips
- Random zoom
- Horizontal and vertical translations

An additional online augmentation pipeline was incorporated during training to improve generalization and reduce overfitting.

## CNN Architecture

The model consists of three convolutional blocks followed by fully connected layers.

| Layer | Configuration |
|---|---|
| Convolutional Block 1 | Conv2D (32 filters, 3×3) + ReLU + MaxPooling |
| Convolutional Block 2 | Conv2D (64 filters, 3×3) + ReLU + MaxPooling |
| Convolutional Block 3 | Conv2D (128 filters, 3×3) + ReLU + MaxPooling |
| Flatten | Flatten feature maps |
| Dense | 256 neurons + ReLU |
| Dropout | 0.5 |
| Output | 11 neurons + Softmax |

### Training Configuration

- **Optimizer:** Adam
- **Loss Function:** Categorical Cross-Entropy
- **Dataset Split:** 70% training / 30% validation
- **Early Stopping:** To prevent unnecessary training
- **Learning Rate Scheduling:** Reduction based on validation loss

## Results

After multiple training iterations and adjustments to the data augmentation pipeline, the model achieved the following results:

| Metric | Training | Validation |
|---|---|---|
| Accuracy | 72.73% | 61.44% |
| Loss | 1.02 | 1.26 |

The model demonstrated progressive improvement in classification accuracy, reaching approximately **61–63% validation accuracy**.

Although moderate overfitting remains, the results indicate that the CNN learned meaningful visual patterns despite the limited dataset.

## Challenges & Lessons Learned

One of the main challenges was achieving good generalization with a relatively small dataset.

Initial experiments showed significant overfitting. Several approaches were explored, including:

- Architectural adjustments
- Learning rate scheduling
- Dropout regularization
- Early stopping
- Additional data augmentation

The most effective improvement came from introducing additional data augmentation during training, which helped reduce validation loss and narrow the generalization gap.

This project provided practical experience in **CNN design, image preprocessing, model optimization, and deep learning evaluation under data constraints**.

## Future Improvements

Potential improvements include:

- **Transfer Learning:** Fine-tuning pretrained architectures such as ResNet or VGG.
- **Dataset Expansion:** Increasing the number and diversity of real satellite images.
- **Batch Normalization:** Improving training stability.
- **Global Average Pooling:** Reducing the number of trainable parameters and limiting overfitting.

## Test Images

The [`test_images/`](test_images/) directory contains sample images of military vehicles used to demonstrate the classification task.

## Technologies & Concepts

- Python
- Deep Learning
- Convolutional Neural Networks (CNNs)
- Computer Vision
- Image Classification
- Data Augmentation
- Satellite Imagery


