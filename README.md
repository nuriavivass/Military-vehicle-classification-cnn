# Military-vehicle-classification-cnn
CNN-based military vehicle classification using satellite imagery and deep learning.


# Military Vehicle Classification Using CNNs

This project focuses on classifying military vehicles from satellite images using a Convolutional Neural Network (CNN) trained from scratch.

The main goal was to explore how well a CNN could perform with a relatively small dataset, while dealing with common image classification challenges such as different viewing angles, lighting conditions and overfitting.

## Dataset

I built my own dataset using images collected from Google Earth and websites such as Virtual Globetrotting. Finding aircraft was relatively straightforward, but collecting images of ground vehicles was more challenging, so I also used some synthetic images from Arma 3.

The dataset includes 11 different military vehicle classes: 6 aircraft and 5 ground vehicles (3 tanks, an artillery vehicle and an armored personnel carrier).

Initially, I had around 20 images per class, which was not enough to train a CNN properly. To increase the amount of training data, I applied data augmentation techniques such as rotations, zooming and flipping.

All images were resized to 120 × 120 pixels.

## Model Architecture

I designed a CNN with three convolutional blocks followed by fully connected layers.

| Layer | Configuration |
|---|---|
| Conv Block 1 | Conv2D (32, 3×3) + ReLU + MaxPooling |
| Conv Block 2 | Conv2D (64, 3×3) + ReLU + MaxPooling |
| Conv Block 3 | Conv2D (128, 3×3) + ReLU + MaxPooling |
| Flatten | Flatten |
| Dense | 256 neurons + ReLU |
| Dropout | 0.5 |
| Output | 11 neurons + Softmax |

For training, I used the Adam optimizer and Categorical Cross-Entropy as the loss function. The dataset was split into 70% training and 30% validation.

I also implemented Early Stopping and learning rate reduction based on validation loss.

## Training and Challenges

The main problem I faced during training was overfitting. The model was learning the training images quite well, but its performance on validation images was significantly worse.

I tried different approaches, including changing the number of neurons, adjusting the learning rate and experimenting with the number of epochs.

What helped the most was adding another data augmentation process directly into the training pipeline, with random rotations, translations, flips and zooming.

This reduced the validation loss and improved generalization, although the difference between training and validation accuracy was still noticeable.

## Results

After several experiments, the final model achieved the following results:

| Metric | Training | Validation |
|---|---|---|
| Accuracy | 72.73% | 61.44% |
| Loss | 1.02 | 1.26 |

The model reached around 61–63% validation accuracy, which shows that it was able to learn useful features despite the limited amount of data.

There is still some overfitting, but the results improved considerably after introducing more data augmentation.

## Possible Improvements

There are several things I would like to explore to improve the model:

- **Transfer Learning:** Using a pretrained model such as ResNet or VGG instead of training a CNN from scratch.
- **Larger Dataset:** Collecting more real satellite images for each vehicle class.
- **Global Average Pooling:** Reducing the number of parameters to help prevent overfitting.
- **Batch Normalization:** Improving training stability.

I think Transfer Learning would be particularly interesting, since the main limitation of this project was the small dataset.

## Test Images

Some example images are available in the [test_images](test_images/) folder.


