# Lab 2: Simple Neural Network for Image Classification

**Course:** Neural Networks and Their Applications  
**Student:** Bichurina S.P., Group 09-261, 3rd year  
**Variant:** 4  
**University:** Kazan (Volga Region) Federal University, Institute of Computational Mathematics and Information Technologies, Department of Information Systems

---

## Task Description

Build a simple feedforward neural network with one hidden layer using Python and PyTorch. Train and use the network for image recognition. Requirements include: direct connections between neurons, at least 4 classes, at least 2 training samples per class, and user-defined training parameters.

## Mathematical Foundations

1. **NLL Loss** — Negative Log-Likelihood loss function, used to compute the discrepancy between predicted and actual class labels, weighted by batch size.
2. **Linear transformation** — y = xW^T + b, where x is the input tensor, W is the weight matrix, b is the bias vector.
3. **ReLU activation** — ReLU(x) = max(0, x), introducing non-linearity into the model.
4. **LogSoftmax** — Computes the logarithm of the softmax function, converting raw outputs into log-probabilities for classification.

## Training Data

The dataset consisted of 200 grayscale images (50 per class) of four handwritten letters: **K, L, M, N**.

## Network Architecture

The network has a simple fully connected architecture with three layers:

- **Input layer:** accepts preprocessed grayscale images, outputs 128 features
- **Hidden layer:** expands the representation to 256 dimensions with ReLU activation
- **Output layer:** 4 neurons (one per class) with LogSoftmax activation

Training was performed using the Adam optimizer with a learning rate of 0.0005. The loss function used was NLL Loss.

## Results

During training, the loss function decreased steadily down to **0.0091**, indicating effective learning. Accuracy increased correspondingly over the epochs.

Testing was performed on a separate dataset not used during training. The model predicted the most probable class for each test image by computing class probabilities. All test samples for each letter (K, L, M, N) were classified correctly.

The test accuracy reached **87.5%**.

## Conclusion

The trained neural network successfully handles the image classification task. The decrease in loss to 0.0091 demonstrates that the model learned to classify the data correctly. Testing on a separate dataset showed 87.5% accuracy. Future improvements could include increasing the dataset size or employing more complex neural network architectures.
