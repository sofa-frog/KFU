# Lab 3: Convolutional Neural Network for Image Classification

**Course:** Neural Networks and Their Applications  
**Student:** Bichurina S.P., Group 09-261, 3rd year  
**Variant:** 4  
**University:** Kazan (Volga Region) Federal University, Institute of Computational Mathematics and Information Technologies, Department of Information Systems

---

## Task Description

Build a convolutional neural network (CNN) using PyTorch, train it on an image dataset. The dataset from Lab 2 could be reused, or a new one could be generated or downloaded.

## Mathematical Foundations

1. **NLL Loss** — Negative Log-Likelihood loss function for computing the error between predicted and target class labels.
2. **Linear transformation** — y = xW^T + b, a fully connected layer operation.
3. **ReLU activation** — ReLU(x) = max(0, x).
4. **Conv2d** — 2D convolution layer that applies learnable filters (kernels) to input images to extract features such as edges, textures, and patterns. Input shape: (N, C, H, W) — batch size, channels, height, width.
5. **MaxPool2d** — Max pooling layer that reduces the spatial dimensions of the input by selecting the maximum value within each pooling window.

## Training Data

The dataset consisted of 200 images of size 64×64 pixels (50 per class) depicting four handwritten letters: **K, L, M, N**.

## Network Architecture

The CNN includes:

- **First convolutional layer:** 32 filters, extracting primary features from the input images
- **Second convolutional layer:** 64 filters, extracting higher-level features
- **Max pooling** after each convolutional layer to reduce spatial dimensions
- **First fully connected layer:** maps extracted features to a 256-dimensional vector
- **Output layer:** 4 neurons (one per class) with LogSoftmax activation

The Adam optimizer was used with a learning rate of 0.0005. Loss was computed using NLL Loss. Training ran for 50 epochs.

## Results

During training, the loss decreased to **0.0584**, and the classification accuracy on the training set reached **96.5%**. Testing on a separate dataset confirmed correct predictions for all tested samples of letters K, L, M, and N.

The final loss on training data dropped to **0.0023**, and test accuracy reached **94%**.

Compared to the simple feedforward network from Lab 2, the CNN trained faster and achieved **6.5% higher accuracy**.

## Conclusion

The developed convolutional neural network successfully classifies handwritten letter images. The reduction in loss and high accuracy on both training and test data confirm the model's quality. Compared to the simple neural network, training is faster and the classification accuracy is noticeably higher. Further improvement could be achieved by expanding the dataset or using more complex architectures.
